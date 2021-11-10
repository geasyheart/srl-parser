# -*- coding: utf8 -*-
#
from src.config import TRAIN_PATH, DEV_PATH
from src.parser import SpanBIOParser

m = SpanBIOParser()
m.fit(
    train_path=TRAIN_PATH,
    dev_path=DEV_PATH,
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',

)
