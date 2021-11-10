# -*- coding: utf8 -*-
#
import pathlib

DATA_PATH = pathlib.Path('.').parent.joinpath('data')

TRAIN_PATH = DATA_PATH.joinpath('train.chinese.conll12.jsonlines')
DEV_PATH = DATA_PATH.joinpath('development.chinese.conll12.jsonlines')
TEST_PATH = DATA_PATH.joinpath('test.chinese.conll12.jsonlines')

MODEL_PATH = DATA_PATH.joinpath('savepoints')
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(exist_ok=True)