# -*- coding: utf8 -*-
#
import json
from typing import List
from typing import Set, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import DATA_PATH, TRAIN_PATH


def tokenize(form: List[List[str]], tokenizer: PreTrainedTokenizerBase, max_length: int, char_base: bool = False):
    """

    Args:
        form:
        tokenizer:
        max_length:
        char_base: 这里指的是form[即 word]是否是字级别的

    Returns:

    """
    res = tokenizer.batch_encode_plus(
        form,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
    )
    result = res.data
    # 可用于长度大于指定长度过滤, overflow指字长度大于指定max_length，如果有cls,sep，那么就算上这个
    result['overflow'] = [len(encoding.overflowing) > 0 for encoding in res.encodings]
    if not char_base:
        word_index = []
        for encoding in res.encodings:
            word_index.append([])

            last_word_idx = -1
            current_length = 0
            for word_idx in encoding.word_ids[1:-1]:
                if word_idx != last_word_idx:
                    word_index[-1].append(current_length)

                current_length += 1
                last_word_idx = word_idx
        result['word_index'] = word_index
        result['word_attention_mask'] = [[True] * len(index) for index in word_index]
    return result


def group_pa_by_p_(srl):
    grouped_srl = {}
    for p, b, e, l in srl:
        bel = grouped_srl.get(p, None)
        if not bel:
            bel = grouped_srl[p] = set()
        bel.add((b, e, l))
    return dict(sorted(grouped_srl.items(), key=lambda x: x[0]))


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

        labels = {'O': 0}
        for sample in self.load_file():
            srl_matrix: List[List[str]] = sample['srl']
            for srl_vector in srl_matrix:
                for srl in srl_vector:
                    labels.setdefault(srl, len(labels))

        with open(label_file_path, 'w') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        return labels


class CoNLL2012SRLDataSet(dataset.Dataset):
    def __init__(self, path, pretrained_name_or_path: str, device: torch.device = 'cpu'):
        super(CoNLL2012SRLDataSet, self).__init__()
        self.device = device
        # self.pretrained_name_or_path = pretrained_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path) if isinstance(
            pretrained_name_or_path,
            str) else pretrained_name_or_path

        # self.lines = sorted(
        #     [i for i in CoNLL2012SRLFile(file_path=path).load_file() if len(i['token']) < 100],
        #     key=lambda x: len(x['token'])
        # )
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
        token_result = tokenize(batch_tokens, tokenizer=self.tokenizer, max_length=512)
        assert any(token_result['overflow']) is False  # 如果这里报错就改self.lines那里
        input_ids = pad_sequence([torch.tensor(input_ids) for input_ids in token_result['input_ids']], batch_first=True)
        token_type_ids = pad_sequence(
            [torch.tensor(token_type_ids) for token_type_ids in token_result['token_type_ids']],
            batch_first=True)
        attention_mask = pad_sequence(
            [torch.tensor(attention_mask) for attention_mask in token_result['attention_mask']],
            batch_first=True)
        word_index = pad_sequence([torch.tensor(word_index) for word_index in token_result['word_index']],
                                  batch_first=True)

        word_attention_mask = pad_sequence(
            [torch.tensor(word_attention_mask) for word_attention_mask in token_result['word_attention_mask']],
            batch_first=True)

        result = {
            # 'batch_srls': batch_srls,
            'batch_tokens': batch_tokens,
            'srl_sets': [_['srl_set'] for _ in batch],
            'srl_matrix': srl_matrix.to(self.device),
            'input_ids': input_ids.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'word_index': word_index.to(self.device),
            'word_attention_mask': word_attention_mask.to(self.device)
        }
        return result

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = False):
        return dataloader.DataLoader(
            dataset=self, batch_size=batch_size,
            shuffle=shuffle, collate_fn=self.collate_fn
        )


if __name__ == '__main__':
    for sample in CoNLL2012SRLDataSet(TRAIN_PATH, 'hfl/chinese-electra-180g-small-discriminator').to_dataloader():
        print(sample)
