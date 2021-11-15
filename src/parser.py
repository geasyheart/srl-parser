# -*- coding: utf8 -*-
#
import math
from typing import Optional, Union, List, Dict, Any

import torch
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed, AutoTokenizer

from src.canvas import render_graph
from src.config import TRAIN_PATH, MODEL_PATH, DATA_PATH
from src.metric import Metric
from src.model import SpanBIOSemanticRoleLabelingModel
from src.transform import CoNLL2012SRLFile, CoNLL2012SRLDataSet
from src.utils import logger, get_entities, build_optimizer_for_pretrained


class SpanBIOParser(object):

    def __init__(self):
        self.model: Optional[SpanBIOSemanticRoleLabelingModel, None] = None

        self.tokenizer = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.labels = CoNLL2012SRLFile(file_path=TRAIN_PATH).get_labels()
        self.id_labels = {v: k for k, v in self.labels.items()}

        # self.tags = get_tags()

    def build_model(self, transformer):
        self.model = SpanBIOSemanticRoleLabelingModel(model=transformer, n_out=len(self.labels))
        self.model.to(self.device)
        logger.info(self.model)
        return self.model

    def build_tokenizer(self, pretrained_model_name: str):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def build_criterion(self):
        raise NotImplementedError

    def build_optimizer(
            self,
            warmup_steps: Union[float, int],
            num_training_steps: int,
            lr=1e-3, weight_decay=0.01,
            transformer_lr=1e-4,
    ):
        optimizer = build_optimizer_for_pretrained(
            model=self.model, pretrained=self.model.transformer, lr=lr,
            weight_decay=weight_decay, transformer_lr=transformer_lr
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def build_dataloader(self, path, transformer, batch_size, shuffle):
        return CoNLL2012SRLDataSet(path=path, pretrained_name_or_path=transformer, device=self.device).to_dataloader(
            batch_size=batch_size,
            shuffle=shuffle
        )

    def fit(self, train_path, dev_path, epoch=100, lr=1e-3, transformer_lr=1e-4,
            pretrained_model_name=None, batch_size=32,
            warmup_steps=0.1, ):
        set_seed(seed=123231)

        self.build_tokenizer(pretrained_model_name=pretrained_model_name)

        train_dataloader = self.build_dataloader(
            path=train_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=True
        )
        dev_dataloader = self.build_dataloader(
            path=dev_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=False
        )

        self.build_model(transformer=pretrained_model_name)

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epoch,
            lr=lr, transformer_lr=transformer_lr
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, optimizer=optimizer,
                             scheduler=scheduler)

    def fit_loop(self, train, dev, epoch, optimizer, scheduler):
        # loss
        min_train_loss, min_dev_loss = math.inf, math.inf
        # metric
        max_dev_metric = 0

        for _epoch in range(1, epoch + 1):
            train_loss = self.fit_dataloader(
                train=train,
                optimizer=optimizer,
                scheduler=scheduler
            )
            if train_loss < min_train_loss:
                logger.info(f'Epoch:{_epoch} save min train loss:{train_loss} model')
                min_train_loss = train_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'train_loss_{train_loss:.4e}.pt'))
                )

            dev_loss, dev_metric = self.evaluate_dataloader(dev)

            if dev_loss < min_dev_loss:
                logger.info(f'Epoch:{_epoch} save min dev loss:{dev_loss} model')
                min_dev_loss = dev_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_loss_{dev_loss:.4e}.pt'))
                )

            if dev_metric > max_dev_metric:
                logger.info(f'Epoch:{_epoch} save max dev metric:{dev_metric.f1} model')
                max_dev_metric = dev_metric
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_metric_{dev_metric.f1:.4e}.pt'))
                )

            logger.info(
                f'Epoch:{_epoch} lr: {scheduler.get_last_lr()[0]:.4e} train loss: {train_loss} ' + \
                f'dev loss: {dev_loss} ' + \
                f'dev metric: {dev_metric}'
            )

    def fit_dataloader(self, train, optimizer, scheduler):
        self.model.train()
        total_loss = 0.
        # metric = Metric()
        for batch in tqdm(train, desc='fit_dataloader'):
            pred = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                word_index=batch['word_index']
            )
            word_mask = batch['word_attention_mask']
            mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
            loss = self.model.loss(pred, batch['srl_matrix'], mask)
            total_loss += loss.item()
            loss.backward()

            # for trues assert
            # trues = batch['srl_matrix'].flatten(end_dim=1)[mask.flatten(end_dim=1)[:, 0]]
            # assert trues.size(0) == sum([len(u) for u in batch['batch_tokens']])

            #
            # pred = self.model.decode(pred, mask)
            #
            # result1 = self.decode_output(pred, batch)
            # result2 = self.decode_output2(pred, batch)
            # assert result1 == result2
            # metric.step(y_preds=result1, y_trues=batch['srl_sets'])

            self._step(optimizer=optimizer, scheduler=scheduler)
        # logger.info(f'train metric: {metric}')
        total_loss /= len(train)
        return total_loss

    def decode_output(self, pred, batch):
        offset = 0
        results = []
        for sent in batch['batch_tokens']:
            results.append([])
            for token in sent:
                token_pred = get_entities([self.id_labels[i] for i in pred[offset]])
                results[-1].append(token_pred)
                offset += 1
        return results

    def decode_output2(self, pred, batch):
        pred = sum(pred, [])
        pred = [self.id_labels[x] for x in pred]
        results = []
        offset = 0
        for sent in batch['batch_tokens']:
            results.append([])
            for token in sent:
                tags_per_token = pred[offset:offset + len(sent)]
                srl_per_token = get_entities(tags_per_token)
                results[-1].append(srl_per_token)
                offset += len(sent)
        assert offset == len(pred)
        # assert results == naive
        return results

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()

        total_loss, metric = 0, Metric()

        for batch in tqdm(dev, desc='evaluate_dataloader'):
            pred = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                word_index=batch['word_index']
            )
            word_mask = batch['word_attention_mask']
            mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
            loss = self.model.loss(pred, batch['srl_matrix'], mask)
            total_loss += loss.item()

            # for trues assert
            trues = batch['srl_matrix'].flatten(end_dim=1)[mask.flatten(end_dim=1)[:, 0]]
            assert trues.size(0) == sum([len(u) for u in batch['batch_tokens']])

            #
            pred = self.model.decode(pred, mask)
            #
            result1 = self.decode_output(pred, batch)
            result2 = self.decode_output2(pred, batch)
            assert result1 == result2
            metric.step(y_preds=result1, y_trues=batch['srl_sets'])

        total_loss /= len(dev)

        return total_loss, metric

    loaded = False

    def load(self, pretrained_model_name, model_path: str, device='cpu'):
        self.device = torch.device(device)
        if not self.loaded:
            self.build_tokenizer(pretrained_model_name=pretrained_model_name)
            self.build_model(transformer=pretrained_model_name)
            self.load_weights(save_path=model_path)
            self.loaded = True

    @torch.no_grad()
    def predict(self, test):
        self.model.eval()

        results = []

        for batch in tqdm(test, desc='predict_dataloader'):
            pred = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                word_index=batch['word_index']
            )
            word_mask = batch['word_attention_mask']
            mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)

            pred = self.model.decode(pred, mask)
            prediction = self.decode_output(pred, batch)
            batch_pred_result = [i for i in self.prediction_to_result(prediction, batch)]
            # for picture
            if len([_ for _ in DATA_PATH.joinpath('imgs').iterdir()]) <= 9:
                render_graph(tokens=batch['batch_tokens'][0], srl_set=batch['srl_sets'][0], suffix='true')
                render_graph(tokens=batch['batch_tokens'][0], srl_set=batch_pred_result[0], suffix='pred')
            results.extend(batch_pred_result)
        return results

    def prediction_to_result(self, prediction: List, batch: Dict[str, Any], delimiter='') -> List:
        for matrix, tokens in zip(prediction, batch['batch_tokens']):
            # result = []
            srls = []
            for i, arguments in enumerate(matrix):
                if arguments:
                    # pas = [(delimiter.join(tokens[x[1]:x[2]]),) + x for x in arguments]
                    # pas.insert(bisect([a[1] for a in arguments], i), (tokens[i], 'PRED', i, i + 1))
                    # result.append(pas)
                    srls.extend([(i, start, end, label) for (label, start, end) in arguments])

            yield srls

    def _step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def save_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.module.state_dict(), save_path)

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path))
        else:
            self.model.module.load_state_dict(torch.load(save_path))
