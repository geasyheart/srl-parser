# -*- coding: utf8 -*-
#
from functools import total_ordering


@total_ordering
class Metric(object):
    def __init__(self):
        self.correct = 0
        self.p = 0
        self.r = 0

    def step(self, y_preds, y_trues):
        for y_pred, y_true in zip(y_preds, y_trues):
            srl = set()
            for token_index, args in enumerate(y_pred):
                srl.update((token_index, start, end, label) for (label, start, end) in args)
            self.correct += len(srl & y_true)
            self.p += len(srl)
            self.r += len(y_true)

    def score(self):
        precision = self.correct / self.p if self.p != 0 else 0
        recall = self.correct / self.r if self.r != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        return precision, recall, f1

    @property
    def f1(self):
        _, _, f1 = self.score()
        return f1

    def __repr__(self):
        precision, recall, f1 = self.score()
        return f' Precision: {precision}, Recall: {recall}, F1: {f1} '

    def __lt__(self, other):
        precision, recall, f1 = self.score()
        return f1 < other

    def __le__(self, other):
        precision, recall, f1 = self.score()
        return f1 <= other

    def __gt__(self, other):
        precision, recall, f1 = self.score()
        return f1 > other

    def __ge__(self, other):
        precision, recall, f1 = self.score()
        return f1 >= other
