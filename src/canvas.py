# -*- coding: utf8 -*-
#
import uuid
from typing import List, Set, Tuple

from graphviz import Digraph

from src.config import DATA_PATH


def render_graph(token: List[str], srl_set: Set[Tuple[int, int, int, str]], suffix=''):
    g = Digraph(format='png')

    for (cur_token, start, end, label) in srl_set:
        sg = Digraph()
        sg.attr(rank='same')
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())
        sg.node(name=from_id, label=token[cur_token])
        sg.node(name=to_id, label="".join(token[start:end]))

        sg.edge(from_id, to_id, label=label)

        g.subgraph(sg)

    save_dir = DATA_PATH.joinpath('imgs')
    if not save_dir.exists():
        save_dir.mkdir()
    g.view(filename=f'{"".join(token)[:10]}.{suffix}.gv', directory=save_dir)


if __name__ == '__main__':
    line = {
        'token': ['各位', '好', '，', '欢迎', '您', '收看', '国际', '频道', '的', '今日', '关注', '。'],
        'srl': [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['B-ARG0', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'B-ARG1', 'B-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']],
        'srl_set': {(1, 0, 1, 'ARG0'), (3, 5, 11, 'ARG2'), (5, 6, 11, 'ARG1'), (3, 4, 5, 'ARG1')},
        'pos': ['PN', 'VA', 'PU', 'VV', 'PN', 'VV', 'NN', 'NN', 'DEG', 'NT', 'NN', 'PU']}
    render_graph(line['token'], line['srl_set'])
