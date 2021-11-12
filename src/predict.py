# -*- coding: utf8 -*-
#
from src.config import MODEL_PATH, TEST_PATH
from src.parser import SpanBIOParser

m = SpanBIOParser()
m.load(
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    model_path=str(MODEL_PATH.joinpath('dev_metric_7.1880e-01.pt'))
)
dataloader = m.build_dataloader(path=TEST_PATH, transformer=m.tokenizer, batch_size=3, shuffle=False)
m.predict(dataloader)
