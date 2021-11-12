# -*- coding: utf8 -*-
#
import logging
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AdamW

from src.config import DATA_PATH


class TqdmHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def get_logger():
    logger = logging.getLogger('con-parser')
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'
    file_handler = logging.FileHandler(filename=DATA_PATH.joinpath('run.log'))
    tqdm_handler = TqdmHandler()
    file_handler.setFormatter(logging.Formatter(fmt))
    tqdm_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    logger.addHandler(tqdm_handler)

    return logger


logger = get_logger()


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
      prev_tag: previous chunk tag.
      tag: current chunk tag.
      prev_type: previous type.
      type_: current type.

    Returns:
      chunk_end: boolean.

    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
      prev_tag: previous chunk tag.
      tag: current chunk tag.
      prev_type: previous type.
      type_: current type.

    Returns:
      chunk_start: boolean.

    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
      seq(list): sequence of labels.
      suffix:  (Default value = False)

    Returns:
      list: list of (chunk_type, chunk_start, chunk_end).
      Example:

    >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-2]
        else:
            tag = chunk[0]
            type_ = chunk[2:]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def build_optimizer_for_pretrained(model: torch.nn.Module,
                                   pretrained: torch.nn.Module,
                                   lr=1e-5,
                                   weight_decay=0.01,
                                   eps=1e-8,
                                   transformer_lr=None,
                                   transformer_weight_decay=None,
                                   no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
                                   **kwargs):
    if transformer_lr is None:
        transformer_lr = lr
    if transformer_weight_decay is None:
        transformer_weight_decay = weight_decay
    params = defaultdict(lambda: defaultdict(list))
    pretrained = set(pretrained.parameters())
    if isinstance(no_decay, tuple):
        def no_decay_fn(name):
            return any(nd in name for nd in no_decay)
    else:
        assert callable(no_decay), 'no_decay has to be callable or a tuple of str'
        no_decay_fn = no_decay
    for n, p in model.named_parameters():
        is_pretrained = 'pretrained' if p in pretrained else 'non_pretrained'
        is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
        params[is_pretrained][is_no_decay].append(p)

    grouped_parameters = [
        {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
        {'params': params['pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': transformer_lr},
        {'params': params['non_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
        {'params': params['non_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
    ]

    return AdamW(
        grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        **kwargs)
