# -*- coding: utf8 -*-
#


from torch import nn
from transformers import AutoModel


class TransformerEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super(TransformerEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

        self.n_out = self.transformer.config.hidden_size
        self.dropout_prob = self.transformer.config.hidden_dropout_prob

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids, token_type_ids, attention_mask):
        seq_out = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]
        return self.dropout(seq_out)

    def __repr__(self):
        s = f'n_out={self.n_out},dropout={self.dropout.p}'
        return f'{self.__class__.__name__}({s})'
