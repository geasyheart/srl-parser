# -*- coding: utf8 -*-
#
import torch
from torch import nn
from torch.functional import F

from src.layers.affine import Biaffine
from src.layers.crf import CRF
from src.layers.mlp import MLP
from src.layers.transformer import TransformerEncoder


class SpanBIOSemanticRoleLabelingModel(nn.Module):
    def __init__(self, model: str, n_out, n_mlp=300, dropout=0.1):
        super(SpanBIOSemanticRoleLabelingModel, self).__init__()

        self.transformer = TransformerEncoder(pretrained_model_name_or_path=model)
        self.s_layer = MLP(
            self.transformer.n_out,
            n_mlp,
            dropout=self.transformer.dropout_prob
        )
        self.e_layer = MLP(
            self.transformer.n_out,
            n_mlp,
            dropout=self.transformer.dropout_prob
        )
        self.biaffine = Biaffine(n_in=n_mlp, n_out=n_out)

        self.crf = CRF(num_tags=n_out)

    def forward(self, input_ids, token_type_ids, attention_mask, word_index):
        seq_out = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        word_out = torch.gather(
            seq_out[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, seq_out.size(-1))
        )

        out1 = self.s_layer(word_out)
        out2 = self.e_layer(word_out)
        out = self.biaffine(out1, out2).permute(0, 2, 3, 1)
        return F.log_softmax(out, dim=-1)

    def loss(self, preds, trues, mask):
        preds = preds.flatten(end_dim=1)
        mask = mask.flatten(end_dim=1)
        trues = trues.flatten(end_dim=1)
        # ValueError: mask of the first timestep must all be on
        first_mask = mask[:, 0]
        preds = preds[first_mask]
        mask = mask[first_mask]
        trues = trues[first_mask]
        return - self.crf(preds, trues, mask, reduction='mean')

    def decode(self, preds, mask):
        preds = preds.flatten(end_dim=1)
        mask = mask.flatten(end_dim=1)
        first_mask = mask[:, 0]
        preds = preds[first_mask]
        mask = mask[first_mask]
        return self.crf.decode(preds, mask)
