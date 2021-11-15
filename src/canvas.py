# -*- coding: utf8 -*-
#
import uuid

from graphviz import Digraph

from src.config import DATA_PATH
from src.transform import group_pa_by_p_


def render_graph(tokens, srl_set, pos=None, suffix=''):
    g = Digraph(format='png')
    # 分词
    for index, token in enumerate(tokens):
        if pos:
            node = f'{token}/{pos[index]}'
        else:
            node = token
        g.node(name=f'node-{index}', label=node)

    group_srls = group_pa_by_p_(srl_set)
    for token_index, srls in group_srls.items():
        # add new level description

        g.node(name=f'label-{token_index}', label=f'谓词:\n{tokens[token_index]}', style='filled', color='lightblue2')
        g.edge(f'node-{token_index}', f'label-{token_index}', style="invis")

        for start, end, label in srls:
            uni_id = str(uuid.uuid4())
            g.node(name=uni_id, label="".join(tokens[start:end]))
            g.edge(f'label-{token_index}', uni_id, label=label)

    save_dir = DATA_PATH.joinpath('imgs')
    if not save_dir.exists():
        save_dir.mkdir()
    g.view(filename=f'{"".join(tokens)[:10]}.{suffix}.gv', directory=save_dir)


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

    render_graph(line['token'], line['srl_set'], pos=line['pos'])
