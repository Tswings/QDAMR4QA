#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zden658 on 1/08/21

import copy
import json
import pandas as pd
import amrlib
import penman
from penman import layout
from penman.model import Model
from collections import Counter
import string
punc = string.punctuation

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
props = {'annotators': 'pos,lemma',
         'pipelineLanguage': 'en',
         'outputFormat': 'json'}

verb_label = ['VB', 'VBZ', 'VBG', 'VBN', 'VBP', 'VBD']
noun_label = ['NN', 'NNS', 'NNP', 'NNPS', 'WP', 'WDT', 'amr-unknown']


def VerbNounInSent(amr_graph, sent, node_dict_rev, node_fn_dict):
    all_nodes_list0 = [ins.target for ins in amr_graph.instances()]
    all_nodes_stem = [ps.stem(node) for node_l in all_nodes_list0 for node in node_l.split("-")]
    count_nodes_stem = Counter(all_nodes_stem)
    verb_tmp, noun_tmp = [], []
    verb_tmp_list, noun_tmp_list = [], []

    sent_pos = nlp.annotate(sent, props)
    parsed_sent = json.loads(sent_pos)

    for tok in parsed_sent['sentences'][0]['tokens']:
        tok_stem = ps.stem(tok['lemma'])
        if tok_stem in all_nodes_stem and tok['pos'] in verb_label: verb_tmp.append(tok_stem)
        if tok_stem in all_nodes_stem and tok['pos'] in noun_label: noun_tmp.append(tok_stem)

    for node in all_nodes_list0:
        node_stem = [ps.stem(n) for n in node.split('-')][0]
        if count_nodes_stem[node_stem] > 1:
            for verb in verb_tmp:
                if verb in node_stem and node in node_fn_dict.keys():
                    verb_tmp_list.append(node_dict_rev[node])
            for noun in noun_tmp:
                if noun in node_stem and node not in node_fn_dict.keys():
                    noun_tmp_list.append(node_dict_rev[node])
        else:
            for verb in verb_tmp:
                if verb in node_stem:
                    verb_tmp_list.append(node_dict_rev[node])
            for noun in noun_tmp:
                if noun in node_stem:
                    noun_tmp_list.append(node_dict_rev[node])

    return verb_tmp_list, noun_tmp_list


def VerbNounInNode(node, verb_list, noun_list, node_dict_rev):
    node_sp = node_dict_rev[node]      # node
    nodes_pos = nlp.annotate(' '.join(node.split('-')), props)
    parsed_node = json.loads(nodes_pos)

    nodeIsverb = nodeIsnoun = False
    first_node_pos = parsed_node['sentences'][0]['tokens'][0]['pos']
    # ex: have-degree
    if first_node_pos in verb_label and node_sp not in noun_list:    nodeIsverb = True
    if first_node_pos in noun_label and node_sp not in verb_list:    nodeIsnoun = True

    return nodeIsverb, nodeIsnoun


def get_candidate_secondary_unknown(amr_info):
    candidate_secondary_unknown = []

    # other relations
    # rel = [ln.split()[0].split('-')[0] for ln in amr_info['graph'].split("\n") if '-of' in ln]
    # for h, r, t in amr_info['amr_graph'].edges():
    #     if ((":ARG" in r) or (r in rel)) and h in amr_info['verb_list'] and t in amr_info['noun_list']:
    #         candidate_secondary_unknown.append((t, r, h))  # secUnknown-rel-verb
    #     if ((":ARG" in r) or (r in rel)) and t in amr_info['verb_list'] and h in amr_info['noun_list']:
    #         candidate_secondary_unknown.append((h, r, t))  # secUnknown-rel-verb

    for h, r, t in amr_info['amr_graph'].edges():
        if h in amr_info['verb_list'] and t in amr_info['noun_list']:
            candidate_secondary_unknown.append((t, r, h))  # secUnknown-rel-verb
        if t in amr_info['verb_list'] and h in amr_info['noun_list']:
            candidate_secondary_unknown.append((h, r, t))  # secUnknown-rel-verb

    subQ_amr_graph_tmp = []
    for cand in candidate_secondary_unknown:
        for var, role, target in amr_info['sub_graphs']:
            if cand[1] in role and type(target) == tuple and (
                    (cand[0], cand[2]) == (var, target[0]) or (cand[0], cand[2]) == (target[0], var)):
                sec_unknown_graph = penman.format(target)
                sec_amr_graph = penman.decode(sec_unknown_graph)
                sec_amr_graph = alignment_amr_graph(sec_amr_graph)
                subQ_length = len([n.source for n in sec_amr_graph.instances()])

                subQ_amr_graph_tmp.append(
                    {'sec_unknown_graph': sec_unknown_graph, "sec_amr_graph": sec_amr_graph, "var": var, "role": role,
                     "target": target, "subQ_length": subQ_length, "cand": cand})

    if subQ_amr_graph_tmp:
        subQ_length_list = [tmp['subQ_length'] for tmp in subQ_amr_graph_tmp]
        secondary_unknown_pos = subQ_length_list.index(max(subQ_length_list))
        return subQ_amr_graph_tmp[secondary_unknown_pos]
    else:
        return dict()


def get_candidate_secondary_unknown1(amr_info):
    sec_unk_rel = []
    candidate_secondary_unknown = []

    graph_str = penman.encode(amr_info['amr_graph'])
    # sec_unk_rel = [c for c in graph_str.split() if '-of' in c]
    # sec_unk_rel = [(ln.split()[0].split('-')[0], amr_info['node_dict_rev'][ln.split()[-1]]) for ln in
    #                graph_str.split("\n") if '-of' in ln]
    # for ln in graph_str.split("\n"):
    #     if '-of' in ln:
    #         ln_list = ln.split()
    #         rel = ln_list[0].split('-')[0]
    #         verb = amr_info['node_dict_rev'][ln_list[-1]] if ln_list[-1][-1] != ')' else amr_info['node_dict_rev'][
    #             ln_list[-1][:-1]]
    #         sec_unk_rel.append((rel, verb))
    for h, r, t in amr_info['amr_graph'].triples:
        if (r, h) in sec_unk_rel and h in amr_info['verb_list']:
            candidate_secondary_unknown.append((t, r, h))  # secUnknown-rel-verb
        if (r, t) in sec_unk_rel and t in amr_info['verb_list']:
            candidate_secondary_unknown.append((h, r, t))  # secUnknown-rel-verb

    # for h, r, t in amr_info['amr_graph'].triples:
    #     if (r, h) in sec_unk_rel and h in amr_info['verb_list']:
    #         candidate_secondary_unknown.append((t, r, h))  # secUnknown-rel-verb
    #     if (r, t) in sec_unk_rel and t in amr_info['verb_list']:
    #         candidate_secondary_unknown.append((h, r, t))  # secUnknown-rel-verb

        # if r in sec_unk_rel:     # if r in sec_unk_rel:
        #     if h in amr_info['verb_list']: # and amr_info['node_list'].index(t) < amr_info['node_list'].index(h):  #
        #         candidate_secondary_unknown.append((t, r, h))   # secUnknown-rel-verb
        #     if t in amr_info['verb_list']: # and amr_info['node_list'].index(h) < amr_info['node_list'].index(t):   #
        #         candidate_secondary_unknown.append((h, r, t))   # secUnknown-rel-verb

    subQ_amr_graph_tmp = []
    for cand in candidate_secondary_unknown:
        for var, role, target in amr_info['sub_graphs']:
            if cand[1] in role and cand[2] == target[0]:
                subQ_length = len(target[1]) if type(target) != str else 1
                sec_unknown_graph = penman.format(target)
                subQ_amr_graph_tmp.append(
                    {'sec_unknown_graph': sec_unknown_graph, "var": var, "role": role, "target": target,
                     "subQ_length": subQ_length, "cand": cand})
                break

    if subQ_amr_graph_tmp:
        subQ_length_list = [tmp['subQ_length'] for tmp in subQ_amr_graph_tmp]
        secondary_unknown_pos = subQ_length_list.index(max(subQ_length_list))
        return subQ_amr_graph_tmp[secondary_unknown_pos]
    else:
        return dict()


def construct_subQ(amr_info, cand, amr_graph_copy_for_Q1, amr_graph_copy_for_Q2, id):
    ori_amr_unknown = amr_info['node_dict_rev']['amr-unknown'] if 'amr-unknown' in amr_info['node_dict_rev'].keys() else 'a10'
    subQ1_ins_s = [ins.source for ins in amr_graph_copy_for_Q1.instances()]
    subQ1_ins_t = [ins.target for ins in amr_graph_copy_for_Q1.instances()]

    subQ2_ins_s = [ins.source for ins in amr_graph_copy_for_Q2.instances()]

    # add edges for subQ1
    if 'amr-unknown' not in subQ1_ins_t:
        unknown_char1 = 'a' if 'a' not in subQ1_ins_s else 'a0'
        _trip0 = (unknown_char1, ':instance', 'amr-unknown')
        amr_graph_copy_for_Q1.triples.append(_trip0)
        amr_graph_copy_for_Q1.epidata[_trip0] = []
        #
        i, j = 0, 0
        for edge in amr_graph_copy_for_Q1.edges():
            if cand[0] in edge: i += 1
            if cand[2] in edge: j += 1
        add_rel_to_node = cand[0] if i < j else cand[2]
        #
        if add_rel_to_node == cand[2]:
            verbInsubQ1 = [ins.source for ins in amr_graph_copy_for_Q1.instances() if ins.source in amr_info['verb_list']]
            if len(verbInsubQ1) > 1:
                edges = [edge for edge in amr_graph_copy_for_Q1.edges() if (edge == cand or edge == tuple(reversed(cand)))]
                pos = edges[0].index(add_rel_to_node)
                _trip = list(edges[0])
                _trip[pos] = unknown_char1
                _trip = tuple(_trip)

                amr_graph_copy_for_Q1.epidata.pop(edges[0])
                amr_graph_copy_for_Q1.triples.remove(edges[0])
                del_edge_ins = (add_rel_to_node, ":instance", amr_info['node_dict'][add_rel_to_node])
                amr_graph_copy_for_Q1.epidata.pop(del_edge_ins)
                amr_graph_copy_for_Q1.triples.remove(del_edge_ins)
            else:
                add_edge = [edge for edge in amr_graph_copy_for_Q2.edges() if add_rel_to_node in edge and edge != cand
                            and tuple(list(reversed(edge))) != cand]
                _trip = (add_rel_to_node, ":mod", unknown_char1) if len(add_edge) != 1 \
                    else (add_rel_to_node, add_edge[0].role, unknown_char1)
        else:
            add_edge = [edge for edge in amr_graph_copy_for_Q2.edges() if add_rel_to_node in edge and edge != cand
                                                                              and tuple(list(reversed(edge))) != cand]
            _trip = (add_rel_to_node, ":mod", unknown_char1) if len(add_edge) != 1 \
                                                            else (add_rel_to_node, add_edge[0].role, unknown_char1)

        amr_graph_copy_for_Q1.triples.append(_trip)
        amr_graph_copy_for_Q1.epidata[_trip] = []
    else:
        print("!!!!!flag!!!!!")
        unknown_char1 = subQ1_ins_s[subQ1_ins_t.index('amr-unknown')]

    # add edges for subQ2
    if ori_amr_unknown in subQ2_ins_s:
        i1, j1 = 0, 0
        for edge in amr_graph_copy_for_Q2.edges():
            if cand[0] in edge: i1 += 1
            if cand[2] in edge: j1 += 1
        add_rel_to_node1 = cand[0] if i1 < j1 else cand[2]

        if add_rel_to_node1 == cand[2]:
            verbInsubQ2 = [ins.source for ins in amr_graph_copy_for_Q2.instances() if ins.source in amr_info['verb_list']]
            if len(verbInsubQ2) > 1:
                for trip in list(amr_graph_copy_for_Q2.epidata):
                    if cand[2] in trip:
                        amr_graph_copy_for_Q2.epidata.pop(trip)
                        amr_graph_copy_for_Q2.triples.remove(trip)
    else:
        print("-----Warning!!!-----")
        unknown_char2 = 'a' if 'a' not in subQ2_ins_s else 'a0'
        ori_amr_unknown = unknown_char2
        _trip0 = (unknown_char2, ':instance', 'amr-unknown')
        amr_graph_copy_for_Q2.triples.append(_trip0)
        amr_graph_copy_for_Q2.epidata[_trip0] = []
        #
        i, j = 0, 0
        for edge in amr_graph_copy_for_Q2.edges():
            if cand[0] in edge: i += 1
            if cand[2] in edge: j += 1
        add_rel_to_node = cand[0] if i < j else cand[2]

        if add_rel_to_node == cand[2]:
            verbInsubQ2 = [ins.source for ins in amr_graph_copy_for_Q2.instances() if ins.source in amr_info['verb_list']]
            if len(verbInsubQ2) > 1:
                edges = [edge for edge in amr_graph_copy_for_Q2.edges() if (edge == cand or edge == tuple(reversed(cand)))]
                pos = edges[0].index(add_rel_to_node)
                _trip = list(edges[0])
                _trip[pos] = unknown_char2
                _trip = tuple(_trip)

                amr_graph_copy_for_Q2.epidata.pop(edges[0])
                amr_graph_copy_for_Q2.triples.remove(edges[0])
                del_edge_ins = (add_rel_to_node, ":instance", amr_info['node_dict'][add_rel_to_node])
                amr_graph_copy_for_Q2.epidata.pop(del_edge_ins)
                amr_graph_copy_for_Q2.triples.remove(del_edge_ins)
            else:
                add_edge = [edge for edge in amr_graph_copy_for_Q1.edges() if add_rel_to_node in edge and edge != cand
                            and tuple(list(reversed(edge))) != cand]
                _trip = (add_rel_to_node, ":mod", unknown_char2) if len(add_edge) != 1 \
                    else (add_rel_to_node, add_edge[0].role, unknown_char2)
        else:
            add_edge = [edge for edge in amr_graph_copy_for_Q1.edges() if add_rel_to_node in edge and edge != cand
                                                                              and tuple(list(reversed(edge))) != cand]
            _trip = (add_rel_to_node, ":mod", unknown_char2) if len(add_edge) != 1 \
                                                            else (add_rel_to_node, add_edge[0].role, unknown_char2)

        amr_graph_copy_for_Q2.triples.append(_trip)
        amr_graph_copy_for_Q2.epidata[_trip] = []
    # ----------
    graph_subQ1 = penman.encode(amr_graph_copy_for_Q1, top=unknown_char1)
    subQ1, _ = gtos.generate([graph_subQ1], disable_progress=True)

    graph_subQ2 = penman.encode(amr_graph_copy_for_Q2, top=ori_amr_unknown)
    subQ2, _ = gtos.generate([graph_subQ2], disable_progress=True)

    return {"subQ1": subQ1, "sec_unknown": amr_info['node_dict'][cand[0]], "subQ2": subQ2, "graph_subQ1": graph_subQ1,
            "graph_subQ2": graph_subQ2, "question": amr_info['sent'], 'id': id}


# def generate_bridge_subQ(amr_graph, subQ_amr_graph, cand, node_dict, node_dict_rev, haveUnknown):
def generate_bridge_subQ(amr_info, cand, subQ_amr_graph, haveUnknown, id):
    #
    amr_graph_copy_for_Q1 = copy.deepcopy(amr_info['amr_graph'])
    amr_graph_copy_for_Q2 = copy.deepcopy(amr_info['amr_graph'])

    if amr_info['node_list'].index(cand[0]) < amr_info['node_list'].index(cand[2]):
        for trip in list(amr_info['amr_graph'].epidata):
            # ----- subQ1 -----
                if trip in subQ_amr_graph.triples and cand[0] not in trip and trip[:2] != (cand[2], ":instance"):
                    amr_graph_copy_for_Q1.epidata.pop(trip)
                    amr_graph_copy_for_Q1.triples.remove(trip)

                # ----- subQ2 -----
                if trip not in subQ_amr_graph.triples and cand[2] not in trip and trip[:2] != (cand[0], ":instance"):
                    amr_graph_copy_for_Q2.epidata.pop(trip)
                    amr_graph_copy_for_Q2.triples.remove(trip)
    else:
        for trip in list(amr_info['amr_graph'].epidata):
            # ----- subQ1 -----
            if trip in subQ_amr_graph.triples and cand[2] not in trip and trip[:2] != (cand[0], ":instance"):
                amr_graph_copy_for_Q1.epidata.pop(trip)
                amr_graph_copy_for_Q1.triples.remove(trip)

            # ----- subQ2 -----
            if trip not in subQ_amr_graph.triples and cand[0] not in trip and trip[:2] != (cand[2], ":instance"):
                amr_graph_copy_for_Q2.epidata.pop(trip)
                amr_graph_copy_for_Q2.triples.remove(trip)

    if haveUnknown:
        return construct_subQ(amr_info, cand, amr_graph_copy_for_Q1, amr_graph_copy_for_Q2, id)
    else:
        return construct_subQ(amr_info, cand, amr_graph_copy_for_Q2, amr_graph_copy_for_Q1, id)


def TreetoGraph(tree):
    # from penman import layout
    # from penman.model import Model
    # layout.interpret(parse_secknown_graph, Model())
    return layout.interpret(tree, Model())


def save_test():
    #  ----- Generate SubQ1 -----
    # for trip in subQ_amr_graph.triples:
    #     amr_graph_copy_for_Q1.epidata.pop(trip)
    #     amr_graph_copy_for_Q1.triples.remove(trip)
    #
    # for trip in amr_graph_copy_for_Q1.triples:
    #     if cand[2] in trip:
    #         _pos = trip.index(cand[2])
    #         _trip0 = ('a', ':instance', 'amr-unknown')
    #         amr_graph_copy_for_Q1.triples.append(_trip0)
    #         amr_graph_copy_for_Q1.epidata[_trip0] = []
    #         #
    #         _trip = list(trip)
    #         _trip[_pos] = 'a'
    #         _trip = tuple(_trip)
    #         amr_graph_copy_for_Q1.triples.append(_trip)
    #         amr_graph_copy_for_Q1.triples.remove(trip)
    #
    #         #
    #         key = amr_graph_copy_for_Q1.epidata[trip]
    #         amr_graph_copy_for_Q1.epidata[_trip] = key if cand[2] != key[0].variable else []
    #         amr_graph_copy_for_Q1.epidata.pop(trip)
    #
    # graph_subQ1 = penman.encode(amr_graph_copy_for_Q1)
    # subQ1, _ = gtos.generate([graph_subQ1], disable_progress=True)

    # -----
    # subQ2_triples = []
    # subQ2_triples.append((cand[0], role, cand[2]))
    # subQ2_triples.append((cand[0], "instance", node_dict[cand[0]]))
    # subQ2_triples.extend(subQ_amr_graph.triples)
    #
    # subQ2_g = penman.Graph(subQ2_triples)
    # subQ2_amr = penman.encode(subQ2_g, top=node_dict_rev['amr-unknown'])
    # subQ2, _ = gtos.generate([subQ2_amr], disable_progress=True)
    #
    # subQ1_triples = []
    # subQ1_triples.append(('a', ":instance", 'amr-unknown'))
    # subQ1_triples.append(('a', cand[1], cand[0]))
    # subQ1_triples.extend(
    #     [trip for trip in amr_graph.triples if trip not in subQ_amr_graph.triples and cand[2] not in trip])
    #
    # subQ1_g = penman.Graph(subQ1_triples)
    # subQ1_amr = penman.encode(subQ1_g)
    # subQ1, _ = gtos.generate([subQ1_amr], disable_progress=True)

    # generate bridge sub-question
    # amr_graph_copy_for_Q1 = copy.deepcopy(amr_graph)
    # amr_graph_copy_for_Q2 = copy.deepcopy(amr_graph)
    #
    # if haveUnknown:
    #     for trip in list(amr_graph.epidata):
    #         # ----- subQ1 -----
    #         if trip in subQ_amr_graph.triples and cand[0] not in trip and trip[:2] != (cand[2], ":instance"):
    #             amr_graph_copy_for_Q1.epidata.pop(trip)
    #             amr_graph_copy_for_Q1.triples.remove(trip)
    #
    #         # ----- subQ2 -----
    #         if trip not in subQ_amr_graph.triples and cand[2] not in trip and trip[:2] != (cand[0], ":instance"):
    #             amr_graph_copy_for_Q2.epidata.pop(trip)
    #             amr_graph_copy_for_Q2.triples.remove(trip)
    #
    #     return construct_subQ(cand, amr_graph_copy_for_Q1, amr_graph_copy_for_Q2, node_dict, node_dict_rev, haveUnknown)
    # else:
    #     for trip in list(amr_graph.epidata):
    #         # ----- subQ2 -----
    #         if trip in subQ_amr_graph.triples and cand[0] not in trip and trip[:2] != (cand[2], ":instance"):
    #             amr_graph_copy_for_Q2.epidata.pop(trip)
    #             amr_graph_copy_for_Q2.triples.remove(trip)
    #
    #         # ----- subQ1 -----
    #         if trip not in subQ_amr_graph.triples and cand[2] not in trip and trip[:2] != (cand[0], ":instance"):
    #             amr_graph_copy_for_Q1.epidata.pop(trip)
    #             amr_graph_copy_for_Q1.triples.remove(trip)
    #
    #     return construct_subQ(cand, amr_graph_copy_for_Q1, amr_graph_copy_for_Q2, node_dict, node_dict_rev, haveUnknown)

    # -----
    # for cand in candidate_secondary_unknown:
    #     secknown_graph = penman.encode(amr_graph, top=cand[0])  # make secUnknown as the root
    #     parse_secknown_graph = penman.parse(secknown_graph)
    #     var, branches = parse_secknown_graph.node
    #
    #     for branch in branches[1:]:
    #         role, target = branch
    #         if cand[1] in role and cand[2] == target[0]:  # Find subQ graph
    #             subQ_graph = penman.format(target)
    #             subQ_amr_graph = penman.decode(subQ_graph)
    #             haveUnknown = True if 'amr-unknown' in subQ_graph else False  # True: subQ is the second-hop question
    #
    #             subQ_dict.append(
    #                 generate_bridge_subQ(amr_graph, subQ_amr_graph, cand, node_dict, node_dict_rev, haveUnknown))
    #             break

    # sec_unknown_graph = penman.encode(amr_info['amr_graph'], top=cand[0])
    # sec_unknown_parse_graph = penman.parse(sec_unknown_graph)
    # var, branches = sec_unknown_parse_graph.node
    #
    # for branch in branches[1:]:
    #     role, target = branch
    #     if cand[1] in role and cand[2] == target[0]:
    #         subQ_length = len(target[1]) if type(target) != str else 1
    #         subQ_amr_graph_tmp.append(
    #             {'secknown_graph': sec_unknown_graph, 'parse_secknown_graph': sec_unknown_parse_graph, "role": role,
    #              "target": target, "branch": branch, "subQ_length": subQ_length, "cand": cand})
    #         break

    # if amr_info['node_list'].index(cand[0]) < amr_info['node_list'].index(cand[2]):
    #     add_edge_rel = [edge for edge in amr_graph_copy_for_Q2.edges() if (cand[0] in edge or cand[2] in edge)
    #                     and edge != cand and tuple(list(reversed(edge))) != cand]
    #     _trip = (add_rel_to_node, ":mod", unknown_char) if len(add_edge_rel) != 1 \
    #                                                     else (add_rel_to_node, add_edge_rel[0].role, unknown_char)
    #     # _trip = (cand[0], ":mod", unknown_char)
    # else:
    #     add_edge_rel = [edge for edge in amr_graph_copy_for_Q2.edges() if (cand[0] in edge or cand[2] in edge)
    #                     and edge != cand and tuple(list(reversed(edge))) != cand]
    #     _trip = (add_rel_to_node, ":mod", unknown_char) if len(add_edge_rel) != 1 \
    #                                                     else (add_rel_to_node, add_edge_rel[0].role, unknown_char)
    #     # _trip = (cand[2], ":mod", unknown_char)

    print("test")


def save_subQ_to_excel(subQ_dict):
    pd_list = []

    for samp in subQ_dict:
        pd_list.append([samp['id'], samp['subQ1'], samp['sec_unknown'], samp['subQ2'], samp['question']])

    df = pd.DataFrame(pd_list, columns=['id', 'subQ1', 'sec_unknown', 'subQ2', 'ques'])
    df.to_excel("sub_question_bridge_final_1117.xlsx", encoding='utf-8', index=False)
    print("hello")


def alignment_amr_graph(amr_graph):
    if len(amr_graph.epidata) != len(amr_graph.triples):
        for i, trip in enumerate(list(amr_graph.epidata)):
            if trip != amr_graph.triples[i]:
                del amr_graph.triples[i]
                if trip == amr_graph.triples[i]:
                    print("delete duplicate triple.")

        if len(amr_graph.triples) == len(amr_graph.epidata) + 1:
            del amr_graph.triples[i + 1]

        assert len(amr_graph.epidata) == len(amr_graph.triples)

    return amr_graph


def data_process(graph):
    #
    node_list, edge_list, verb_list, noun_list = [], [], [], []
    node_dict, node_dict_rev, sub_graphs = dict(), dict(), []

    edge_list_nodigit = []
    amr_info = dict()

    sent = graph.split("\n")[0][8:]     # sent
    amr_graph = penman.decode(graph)    # generate AMR_graph

    if len(amr_graph.epidata) != len(amr_graph.triples):
        amr_graph = alignment_amr_graph(amr_graph)

    #
    for ins in amr_graph.instances():
        node_list.append(ins.source)
        node_dict[ins.source] = ins.target
        node_dict_rev[ins.target] = ins.source

    node_fn_dict = dict()
    for source, rel, target in amr_graph.edges():
        if node_dict[source] not in node_fn_dict.keys():
            t = node_dict[target] if target in node_list else target
            node_fn_dict[node_dict[source]] = [t]
        else:
            t = node_dict[target] if target in node_list else target
            node_fn_dict[node_dict[source]].append(t)

    #
    verbInsent, nounInsent = VerbNounInSent(amr_graph, sent, node_dict_rev, node_fn_dict)
    verb_list.extend(verbInsent)
    noun_list.extend(nounInsent)

    for ins in amr_graph.instances():
        nodeIsverb, nodeIsnoun = VerbNounInNode(ins.target, verb_list, noun_list, node_dict_rev)
        if nodeIsverb:  verb_list.append(ins.source)
        if nodeIsnoun:  noun_list.append(ins.source)
    #
    assert list(set(verb_list).intersection(set(noun_list))) == []
    #
    for edge in amr_graph.edges():
        edge_list.append((edge.source, edge.target, 1))
        edge_list_nodigit.append((edge.source[0] + '-' + edge.target[0]))

    def extract_graph(graph):
        parse_graph = penman.parse(graph)
        var, branches = parse_graph.node
        for branch in branches[1:]:
            role, target = branch
            headNodeOfSubTree = [sg[2][0] for sg in sub_graphs]

            if type(target) == str and target in headNodeOfSubTree:
                index = headNodeOfSubTree.index(target)
                sub_graphs.append((var, role, sub_graphs[index][2]))
            else:
                sub_graphs.append((var, role, target))

            if type(target) != str:
                extract_graph(penman.format(target))
        return

    extract_graph(graph)

    amr_info['graph'] = graph
    amr_info['sent'] = sent
    amr_info['sub_graphs'] = sub_graphs
    amr_info['amr_graph'] = amr_graph
    amr_info['node_list'] = node_list
    amr_info['edge_list'] = edge_list
    amr_info['verb_list'] = list(set(verb_list))
    amr_info['noun_list'] = list(set(noun_list))
    amr_info['edge_list_nodigit'] = edge_list_nodigit
    amr_info['node_dict'] = node_dict
    amr_info['node_dict_rev'] = node_dict_rev
    # adj_graph_array = construct_adj_graph(amr_info)
    # amr_info['adj_graph'] = adj_graph_array

    return amr_info


def add_question_mark(question):
    if '?' not in question:
        question = (question[:-1] + '?') if question[-1] in punc else (question + '?')

    return question


if __name__ == '__main__':
    full_file = 'hotpotqa_data/hotpot_train_v1.1.json'
    test_file = 'hotpotqa_data/hotpot_dev_distractor_v1.json'

    bridge_file = []
    bridge_id = []
    comparison_file = []

    with open(test_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)  # [:100]

    for file in full_data:
        if file['type'] == 'bridge':
            question = add_question_mark(file['question'])
            bridge_file.append(question)
            bridge_id.append(file['_id'])

        # if file['type'] == 'comparison':
        #     question = add_question_mark(file['question'])
        #     comparison_file[file['_id']] = question

    questionType = 'bridge'                         #'bridge' or 'comparison':

    stog = amrlib.load_stog_model()
    gtos = amrlib.load_gtos_model()
    graphs = stog.parse_sents(bridge_file)        # bridge_file[:10]
    subQ_dict = []

    for idx, graph in enumerate(graphs):
        # -----
        print("\n *****i=%d*****" % idx)

        # data preprocess

        amr_info = data_process(graph)

        # Find all candidate_secondary_unknown
        secondary_unknown = get_candidate_secondary_unknown(amr_info)
        findSecUnknown = True if secondary_unknown else False
        print("Find second unknown node: %s" % str(findSecUnknown))

        # generate sub-questions based on secondary_unknown
        if secondary_unknown:
            haveUnknown = True if 'amr-unknown' in secondary_unknown['sec_unknown_graph'] else False
            subQ_dict.append(generate_bridge_subQ(amr_info, secondary_unknown['cand'],
                                                  secondary_unknown['sec_amr_graph'], haveUnknown, bridge_id[idx]))
        else:
            print("Can't find secondary unknown nodes!")
            subQ_tmp = {"subQ1": amr_info['sent'], "sec_unknown": '', "subQ2": amr_info['sent'],
                        "graph_subQ1": amr_info['graph'], "graph_subQ2": amr_info['graph'],
                        "question": amr_info['sent'], "id": bridge_id[idx]}
            subQ_dict.append(subQ_tmp)

    # save results
    # save_subQ_to_excel(subQ_dict)
    # print(subQ_dict)


