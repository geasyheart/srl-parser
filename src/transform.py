# -*- coding: utf8 -*-
#
import json
from typing import List, Set, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import DATA_PATH, TRAIN_PATH


def group_pa_by_p_(srl):
    grouped_srl = {}
    for p, b, e, l in srl:
        bel = grouped_srl.get(p, None)
        if not bel:
            bel = grouped_srl[p] = set()
        bel.add((b, e, l))
    return grouped_srl


class CoNLL2012SRLFile(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_file(self, doc_level_offset=True):
        num_docs, num_sentences = 0, 0
        with open(self.file_path, 'r') as f:
            for doc_line in tqdm(f, desc='load file'):
                doc = json.loads(doc_line)
                num_tokens_in_doc = 0
                num_docs += 1
                for sid, (sentence, srl) in enumerate(zip(doc['sentences'], doc['srl'])):
                    if doc_level_offset:
                        srl = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2] - num_tokens_in_doc, x[3])
                               for x in srl]
                    else:
                        srl = [(x[0], x[1], x[2], x[3]) for x in srl]

                    for x in srl:
                        if any([o < 0 for o in x[:3]]):  # 判断长度是否小于0
                            raise ValueError(f'Negative offset occurred, maybe doc_level_offset=False')
                        if any([o >= len(sentence) for o in x[:3]]):  # 判断长度是否大于句子长度
                            raise ValueError('Offset exceeds sentence length, maybe doc_level_offset=True')
                    # 去重
                    deduplicated_srl = set()
                    pa_set = set()
                    for p, b, e, l in srl:
                        pa = (p, b, e)
                        if pa in pa_set:
                            continue
                        pa_set.add(pa)
                        deduplicated_srl.add((p, b, e, l))
                    yield self.build_sample(sentence, deduplicated_srl, doc, sid)
                    num_sentences += 1
                    num_tokens_in_doc += len(sentence)

    def build_sample(self, tokens: List[str], deduplicated_srl: Set, doc: Dict, sid: int):
        """
        返回bio格式的sample，用序列标注的方式来做semantic role labeling
        :param tokens:
        :param deduplicated_srl:
        :param doc:
        :param sid:
        :return:
        """
        # 注意这里，把谓词给忽略掉了，市面上对于谓词，一般会单独分出一个二分类任务来做
        # 那能不能做？
        # 我觉得可以，可以在biaffine那一层来学习这个规律
        # 但是在crf那层没啥子希望，为啥子？
        # 因为谓词是单个词。。。
        # 额外插个话题进来，amr貌似是近来的热点，谓词可以不一定为词，也可以为多个词组成的，边界更具有语义一些。
        deduplicated_srl = set((x[0], x[1], x[2] + 1, x[3]) for x in deduplicated_srl if x[3] != 'V')
        labels = [['O'] * len(tokens) for _ in range(len(tokens))]
        srl = group_pa_by_p_(deduplicated_srl)
        for p, args in sorted(srl.items()):
            labels_per_p = labels[p]
            for start, end, label in args:
                assert end > start
                assert label != 'V'  # We don't predict predicate（谓词）
                labels_per_p[start] = 'B-' + label
                for j in range(start + 1, end):
                    labels_per_p[j] = 'I-' + label
        sample = {
            'token': tokens,
            'srl': labels,
            'srl_set': deduplicated_srl,
        }
        if 'pos' in doc:
            sample['pos'] = doc['pos'][sid]
        return sample

    def get_labels(self):
        label_file_path = DATA_PATH.joinpath('labels.json')
        if label_file_path.exists():
            with open(label_file_path, 'r') as f:
                return json.load(f)

        labels = {'[PAD]': 0, 'O': 1}
        for sample in self.load_file():
            srl_matrix: List[List[str]] = sample['srl']
            for srl_vector in srl_matrix:
                for srl in srl_vector:
                    labels.setdefault(srl, len(labels))

        with open(label_file_path, 'w') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        return labels


def encoder_texts(texts: List[List[str]], tokenizer, add_cls_token: bool = True, add_sep_token: bool = False):
    # 统计句子中最大的词长度
    fix_len = max([max([len(word) for word in text]) for text in texts])

    matrix = []
    for _text in texts:
        vector = []
        text = [*_text]
        if add_cls_token:
            text.insert(0, tokenizer.cls_token)
        if add_sep_token:
            text.append(tokenizer.sep_token)

        input_ids = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
        )['input_ids']

        for _input_ids in input_ids:
            # 修复例如: texts = [['\ue5f1\ue5f1\ue5f1\ue5f1']] 这种情况
            _input_ids = _input_ids or [tokenizer.unk_token_id]
            vector.append(_input_ids + (fix_len - len(_input_ids)) * [tokenizer.pad_token_id])
        matrix.append(torch.tensor(vector, dtype=torch.long))
    return pad_sequence(matrix, batch_first=True)


class CoNLL2012SRLDataSet(dataset.Dataset):
    def __init__(self, path, pretrained_name_or_path: str, device: torch.device = 'cpu'):
        super(CoNLL2012SRLDataSet, self).__init__()
        self.device = device
        # self.pretrained_name_or_path = pretrained_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path) if isinstance(
            pretrained_name_or_path,
            str) else pretrained_name_or_path

        self.lines = [i for i in CoNLL2012SRLFile(file_path=path).load_file() if len(i['token']) < 100]
        self.labels = CoNLL2012SRLFile(file_path=path).get_labels()

    def __getitem__(self, item) -> Dict:
        line = self.lines[item]
        srl_ids = []
        for srl_vector in line['srl']:
            srl_ids.append([])
            for srl in srl_vector:
                srl_id = self.labels[srl]
                srl_ids[-1].append(srl_id)
        line['srl_ids'] = torch.tensor(srl_ids, dtype=torch.long)
        return line

    def __len__(self):
        return len(self.lines)

    def collate_fn(self, batch: List[Dict]):

        batch_srls = [_['srl_ids'] for _ in batch]
        batch_tokens = [_['token'] for _ in batch]
        srl_matrix = torch.zeros(
            len(batch_srls),
            max(i.size(0) for i in batch_srls),
            max(i.size(0) for i in batch_srls),
            dtype=torch.long
        )
        for index, srl in enumerate(batch_srls):
            w = srl.size(0)
            srl_matrix[index][:w, :w] = srl

        return {
            'batch_srls': batch_srls,
            'batch_tokens': batch_tokens,
            'srl_sets': [_['srl_set'] for _ in batch],
            'subwords': encoder_texts(batch_tokens, tokenizer=self.tokenizer).to(self.device),
            'srl_matrix': srl_matrix.to(self.device)
        }

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = False):
        return dataloader.DataLoader(
            dataset=self, batch_size=batch_size,
            shuffle=shuffle, collate_fn=self.collate_fn
        )


if __name__ == '__main__':
    for sample in CoNLL2012SRLDataSet(TRAIN_PATH, 'hfl/chinese-electra-180g-small-discriminator').to_dataloader():
        print(sample)
