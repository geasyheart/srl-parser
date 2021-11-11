# -*- coding: utf8 -*-
#
from unittest import TestCase

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from src.transform import tokenize


class TestSample(TestCase):
    def test_max_length(self):
        """
        测试max_length overflow情况
        :return:
        """
        pass

    def test_sample(self):
        form = [
            ['我', '呀'],
            ['我', '小明', '呀']
        ]

        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-electra-180g-small-discriminator')
        result = tokenize(form, tokenizer, 6)
        model = AutoModel.from_pretrained('hfl/chinese-electra-180g-small-discriminator')

        input_ids = pad_sequence([torch.tensor(input_ids) for input_ids in result['input_ids']], batch_first=True)
        token_type_ids = pad_sequence([torch.tensor(token_type_ids) for token_type_ids in result['token_type_ids']],
                                      batch_first=True)
        attention_mask = pad_sequence([torch.tensor(attention_mask) for attention_mask in result['attention_mask']],
                                      batch_first=True)

        # tensor([[ 101, 2769, 1435,  102,    0,    0],
        #         [ 101, 2769, 2207, 3209, 1435,  102]])

        # 1. 获取bert output.
        bert_out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        seq_out = bert_out[0]

        word_index = pad_sequence([torch.tensor(word_index) for word_index in result['word_index']], batch_first=True)

        # 2. 获取词首字向量，包括cls开头
        word_out = torch.cat([seq_out[:, :1, :], torch.gather(
            seq_out[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, seq_out.size(-1))
        )], dim=1)

        word_attention_mask = pad_sequence(
            [torch.tensor(word_attention_mask) for word_attention_mask in result['word_attention_mask']],
            batch_first=True)

        # 这里方便view
        # 1. ['我', '呀']
        self.assertTrue((seq_out[0][0] == word_out[0][0]).all())  # cls
        self.assertTrue((seq_out[0][1] == word_out[0][1]).all())  # 我
        self.assertTrue((seq_out[0][2] == word_out[0][2]).all())  # 呀

        self.assertTrue((word_out[0][1] == word_out[0][3]).all())  # 填充位

        # 2. ['我', '小明', '呀']
        self.assertTrue((seq_out[1][0] == word_out[1][0]).all())  # cls
        self.assertTrue((seq_out[1][1] == word_out[1][1]).all())  # 我
        self.assertTrue((seq_out[1][2] == word_out[1][2]).all())  # 小明
        self.assertTrue((seq_out[1][4] == word_out[1][3]).all())  # 呀

        # 3. Note: word_out的时候concat了seq_out[:, :1, :](cls)，所以word_out的长度比word_attention_mask大1
        self.assertEqual(word_out.size(1), word_attention_mask.size(1) + 1)

        # 4. 获取每个词对应的向量
        result = word_out[:, 1:, :][word_attention_mask]
        result2 = result.split(word_attention_mask.sum(1).tolist())
        self.assertEqual(len(result2[0]), 2)
        self.assertEqual(len(result2[1]), 3)
