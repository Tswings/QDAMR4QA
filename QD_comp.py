#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zden658 on 3/08/21
import re
import copy
import json
import pickle
import amrlib
import penman
import pandas as pd
from unsupervised_multihop_QA import T5_QG
t5_qg = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight")
import string
punc = string.punctuation

import spacy
spacy_nlp = spacy.load("en_core_web_sm")
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
props = {'annotators': 'pos,lemma',
         'pipelineLanguage': 'en',
         'outputFormat': 'json'}

verb_label = ['VB', 'VBZ', 'VBG', 'VBN', 'VBP', 'VBD']
noun_label = ['NN', 'NNS', 'NNP', 'NNPS', 'WP', 'WDT', 'amr-unknown']

from nltk.stem import PorterStemmer
ps = PorterStemmer()


def VerbNounInSent(amr_graph, sent):
    all_nodes_stem = [ps.stem(ins.target.split("-")[0])for ins in amr_graph.instances()]
    verb_fn_tmp, noun_fn_tmp = [], []
    verbInsent, nounInsent = [], []

    AndInsent = True if 'and' in all_nodes_stem else False
    AndNode = [ins.source for ins in amr_graph.instances() if ins.target == "and"] if AndInsent else []

    sent_pos = nlp.annotate(sent, props)
    parsed_sent = json.loads(sent_pos)

    for tok in parsed_sent['sentences'][0]['tokens']:
        tok_stem = ps.stem(tok['lemma'])
        if tok_stem in all_nodes_stem and tok['pos'] in verb_label: verb_fn_tmp.append(tok_stem)    #
        if tok_stem in all_nodes_stem and tok['pos'] in noun_label: noun_fn_tmp.append(tok_stem)

    for ins in amr_graph.instances():
        node_stem = ps.stem(ins.target.split("-")[0])

        if node_stem in verb_fn_tmp and ins.source not in verbInsent:
            verbInsent.append(ins.source)
            continue
        if node_stem in noun_fn_tmp and ins.source not in nounInsent:
            nounInsent.append(ins.source)

    return verbInsent, nounInsent, AndNode


def VerbInNode(node):
    node_info = dict()
    node_info['node'] = node

    node_str = nlp.annotate(' '.join(node.split('-')), props)
    parsed_node = json.loads(node_str)

    for tok in parsed_node['sentences'][0]['tokens']:
        # node_info['node'] = node
        # node_info['lemma'] = tok['lemma']
        # node_info['pos'] = tok['pos']

        if tok['pos'] in verb_label:
            node_info['haveVerb'] = True
            return node_info
        else:
            node_info['haveVerb'] = False

    return node_info


def delete_rel_node(amr_graph_copy_for_subQ, trip, type, degreeNode):
    _trip = list(trip)
    if type in ["earlier", "earliest", "first", "last", "later", "latest"]:
        _trip[1] = ":time"
    if type in ["less", "more", "most", "smaller"]:
        _trip[1] = ":quant"
    if type in ["taller", "higher", "larger", "tallest", "highest"]:
        _trip[1] = ":degree"
    if type in ["longer", "shorter", "longest"]:
        _trip[1] = ":duration"
    if type in ["closer", "closest"]:
        _trip[1] = ":polarity"
    if type in ["older", "younger", "oldest"]:
        _trip[1] = ":age"
    if type in ["newer"]:
        _trip[1] = ":be-temporally-at-91"

    _trip = tuple(_trip)

    if degreeNode:
        return amr_graph_copy_for_subQ, _trip[1]
    else:
        amr_graph_copy_for_subQ.triples.remove(trip)
        amr_graph_copy_for_subQ.epidata.pop(trip)
        amr_graph_copy_for_subQ.triples.append(_trip)
        amr_graph_copy_for_subQ.epidata[_trip] = []

        return amr_graph_copy_for_subQ, _trip[1]


def save_draft():
    # retained_verb = list(set(amr_info['verb_list']).difference(set(subQ_nodes)))
    # if (trip[0] in retained_verb or trip[2] in retained_verb) and trip[1] != ":instance":
    #     if trip[0] in retained_verb:
    #         _subQ_verb_top = penman.encode(amr_graph_copy_for_subQ, top=trip[0])
    #     if trip[2] in retained_verb:
    #         _subQ_verb_top = penman.encode(amr_graph_copy_for_subQ, top=trip[2])
    #
    #     _subQ_parse = penman.parse(_subQ_verb_top)
    #     var, branches = _subQ_parse.node
    #
    #     for branch in branches[1:]:
    #         if 'amr-' in penman.format(branch[1]):
    #             role = branch[0].split('-')[0]
    #
    #             if role in trip[1]:
    #                 new_role = ":time"
    #                 _trip = (trip[0], new_role, trip[2])
    #                 key = amr_info['amr_graph'].epidata[trip]
    #                 amr_graph_copy_for_subQ.triples.append(_trip)
    #                 amr_graph_copy_for_subQ.epidata[_trip] = key
    #                 #
    #                 amr_graph_copy_for_subQ.triples.remove(trip)
    #                 amr_graph_copy_for_subQ.epidata.pop(trip)
    #             break
    return 0


class CommonSubStr:
    def maxprefix(self, str1, str2):
        """
        找出相邻的后缀子串的公共子串的长度
        :param str1:
        :param str2:
        :return: 最长公共子串的长度
        """
        common_len = 0
        # 从程序可以看到,相邻后缀子串的第一个字符必须相等
        for i in range(min(len(str1), len(str2))):
            if list(str1)[i] == list(str2)[i]:
                common_len += 1
            else:
                break
        return common_len


    def getMaxCommonStr(self, string):
        longestSubStrLen = 0
        longestSubStr = None
        longestSubStr_list = []

        # 获取后缀数组并排序
        suffixes = [string[i:] for i in range(len(string))]
        suffixes.sort()

        for i in range(1, len(string)):
            common_str_len = self.maxprefix((suffixes[i]), suffixes[i-1])
            longestSubStr_list.append(suffixes[i - 1][:common_str_len])     #
            if common_str_len > longestSubStrLen:
                longestSubStrLen = common_str_len
                longestSubStr = suffixes[i-1][:common_str_len]

        return longestSubStr, longestSubStr_list


def finetuning_edge(amr_graph_copy_for_subQ, amr_info, subQ_unknown_edge, adverb_node, rpl_rel):
    trip_del = subQ_unknown_edge
    new_trip = (trip_del[0], trip_del[1], adverb_node) if trip_del.index(
        amr_info['unknownNode']) else (adverb_node, trip_del[1], trip_del[2])
    amr_graph_copy_for_subQ.triples.remove(trip_del)
    amr_graph_copy_for_subQ.epidata.pop(trip_del)
    amr_graph_copy_for_subQ.triples.append(new_trip)
    amr_graph_copy_for_subQ.epidata[new_trip] = []

    comp_node_trip = (amr_info['unknownNode'], rpl_rel, adverb_node)
    amr_graph_copy_for_subQ.triples.append(comp_node_trip)
    amr_graph_copy_for_subQ.epidata[comp_node_trip] = []

    return amr_graph_copy_for_subQ


def haveSameStartNode(amr_edges, com_substr_pos):
    subQ_nodes = []
    subQ1_nodes = []
    subQ2_nodes = []
    del_substr_instances = []
    del_substr00_instances = []
    del_substr01_instances = []

    for i, com_substr in enumerate(com_substr_pos):
        subQ_nodes_tmp = []
        del_subQ_instances_tmp = []

        for str in com_substr:
            s, r, t = amr_edges[str][0], amr_edges[str][1], amr_edges[str][2]
            if (s, ":instance", amr_info['node_dict'][s]) not in del_subQ_instances_tmp:
                subQ_nodes_tmp.append(s)
                del_subQ_instances_tmp.append((s, ":instance", amr_info['node_dict'][s]))
            if (t, ":instance", amr_info['node_dict'][t]) not in del_subQ_instances_tmp:
                subQ_nodes_tmp.append(t)
                del_subQ_instances_tmp.append((t, ":instance", amr_info['node_dict'][t]))

        subQ_nodes.append(subQ_nodes_tmp)
        del_substr_instances.append(del_subQ_instances_tmp)

    haveSSN = True if del_substr_instances and del_substr_instances[0][0][0] == del_substr_instances[1][0][0] else False
    # com_substr00 = com_substr_pos[0][0]
    # com_substr01 = com_substr_pos[1][0]
    #
    # s0, r0, t0 = amr_edges[com_substr00][0], amr_edges[com_substr00][1], amr_edges[com_substr00][2]
    # s1, r1, t1 = amr_edges[com_substr01][0], amr_edges[com_substr01][1], amr_edges[com_substr01][2]
    #
    # if (s0, ":instance", amr_info['node_dict'][s0]) not in del_substr00_instances:
    #     subQ1_nodes.append(s0)
    #     del_substr00_instances.append((s0, ":instance", amr_info['node_dict'][s0]))
    # if (t0, ":instance", amr_info['node_dict'][t0]) not in del_substr00_instances:
    #     subQ1_nodes.append(t0)
    #     del_substr00_instances.append((t0, ":instance", amr_info['node_dict'][t0]))
    #
    # if (s1, ":instance", amr_info['node_dict'][s1]) not in del_substr01_instances:
    #     subQ2_nodes.append(s1)
    #     del_substr01_instances.append((s1, ":instance", amr_info['node_dict'][s1]))
    # if (t1, ":instance", amr_info['node_dict'][t1]) not in del_substr01_instances:
    #     subQ2_nodes.append(t1)
    #     del_substr01_instances.append((t1, ":instance", amr_info['node_dict'][t1]))
    #
    # subQ_nodes.append(subQ1_nodes)
    # subQ_nodes.append(subQ2_nodes)
    # haveSSN = True if del_substr00_instances[0][0] == del_substr01_instances[0][0] else False
    return subQ_nodes, haveSSN


def getsubstring_nodes(amr_info, subQ_nodes_tmp, haveSSN):
    substring_nodes = []
    if haveSSN:
        for sn in subQ_nodes_tmp:
            substring_nodes_tmp = []
            com_n = sn[1:][0]
            amr_graph_top_com_n = penman.encode(amr_info['amr_graph'], top=com_n)
            parse_amr_graph_top_com_n = penman.parse(amr_graph_top_com_n)
            var_com_n, branches_com_n = parse_amr_graph_top_com_n.node

            for brh in branches_com_n[1:]:
                role_com_n, target_com_n = brh
                if target_com_n[0] != sn[0] and type(target_com_n) != str:
                    tree_target_com_n = penman.format(target_com_n)
                    amr_graph_tree_target_com_n = penman.decode(tree_target_com_n)

                    for ins in amr_graph_tree_target_com_n.instances():
                        substring_nodes_tmp.append(ins.source)

            substring_nodes.append(substring_nodes_tmp)
    else:
        for sn in subQ_nodes_tmp:
            substring_nodes_tmp = []
            substring_nodes_tmp1 = []
            com_n = sn[:][0]
            amr_graph_top_com_n = penman.encode(amr_info['amr_graph'], top=com_n)
            parse_amr_graph_top_com_n = penman.parse(amr_graph_top_com_n)
            var_com_n, branches_com_n = parse_amr_graph_top_com_n.node

            for brh in branches_com_n[1:]:
                role_com_n, target_com_n = brh
                if type(target_com_n) != str:
                    tree_target_com_n = penman.format(target_com_n)
                    amr_graph_tree_target_com_n = penman.decode(tree_target_com_n)

                    substring_nodes_tmp1 = [ins.source for ins in amr_graph_tree_target_com_n.instances()]
                    if amr_info['node_list'][0] not in substring_nodes_tmp1:
                        substring_nodes_tmp.extend(substring_nodes_tmp1)

            substring_nodes.append(substring_nodes_tmp)

    return substring_nodes


def find_other_n_e_d(node_edge_dN, amr_info, nodes_edge_with_degreeNode):
    node_edge_dN_flag = True
    for ed in amr_info['edge_list_nodigit1']:
        el_no = ed.split('-')
        if el_no[1] == node_edge_dN and el_no[0] != amr_info['degreeNode']:
            node_edge_dN_flag = False
    if node_edge_dN_flag and node_edge_dN != amr_info['degreeNode']:
        nodes_edge_with_degreeNode.append(node_edge_dN)

    return nodes_edge_with_degreeNode


def default_sub_Q(amr_info, type):
    subQ_tmp = {"subQ1": amr_info['sent'], "sec_unknown": 'comparision', "subQ2": amr_info['sent'],
                "graph_subQ1": amr_info['graph'], "graph_subQ2": amr_info['graph'],
                "question": amr_info['sent'], "type": type, "rule_type": rule_type}

    return subQ_tmp


class GenerateSubQuestion:
    def comp_type01_subQ(self, amr_info, type, com_substr_pos, rule_type, key):
        type_rsv = ["taller", "longer", "larger"]
        type_rsv2 = ["closer", "closest", "longer", "longest"]
        amr_edges = amr_info['edge_list']
        subQ_dict = dict()

        # # Type 4 and Type 5
        # if rule_type in ["type4", "type5"]:
        #     tp_nodes = [key for key, value in amr_info['node_dict'].items() if type in value]
        #     tp_node = tp_nodes[0] if tp_nodes else []
        #
        subQ_nodes_tmp, haveSSN = haveSameStartNode(amr_edges, com_substr_pos)

        substring_nodes = getsubstring_nodes(amr_info, subQ_nodes_tmp, haveSSN)

        assert len(com_substr_pos) == 2
        subQ1_edges = [(amr_edges[node][0], amr_edges[node][1], amr_edges[node][2]) for node in com_substr_pos[1]]
        subQ2_edges = [(amr_edges[node][0], amr_edges[node][1], amr_edges[node][2]) for node in com_substr_pos[0]]

        if haveSSN and subQ_nodes_tmp[0][0] == subQ_nodes_tmp[1][0] == amr_info['degreeNode']:
            return default_sub_Q(amr_info, type)

        nodes_after_degree = []
        if amr_info['degreeNode']:
            amr_graph_top_dg = penman.encode(amr_info['amr_graph'], top=amr_info['degreeNode'])
            parse_amr_graph_top_dg = penman.parse(amr_graph_top_dg)
            var_dg, branches_dg = parse_amr_graph_top_dg.node

            for bch in branches_dg[1:]:
                role_dg, target_dg = bch
                if amr_info['node_list'].index(target_dg[0]) > amr_info['node_list'].index(var_dg) and target_dg[0] != \
                        amr_info['unknownNode'] and target_dg[0] not in type_rsv2:
                    graph_dg = penman.format(target_dg)
                    amr_graph_dg = penman.decode(graph_dg)
                    nodes_in_graph_dg = [ins[0] for ins in amr_graph_dg.instances()]
                    nodes_after_degree.extend(nodes_in_graph_dg)

        for i, com_substr in enumerate(com_substr_pos):
            del_subQ_edges = []
            del_subQ_nodes = []
            del_subQ_instances = []
            nodes_edge_with_degreeNode = []

            amr_graph_copy_for_subQ = copy.deepcopy(amr_info['amr_graph'])
            subQQ_edges = subQ1_edges if i == 0 else subQ2_edges
            subQQ_nodes_in_edges = [[e[0], e[2]] for e in subQQ_edges]
            subQQ_nodes_in_edges = [l for ll in subQQ_nodes_in_edges for l in ll]

            #
            # Type 4 and Type 5
            if rule_type in ["type4", "type5"]:
                tp_nodes = [key for key, value in amr_info['node_dict'].items() if type in value]
                subQ_nodes_tmp0 = subQ_nodes_tmp[1] if i == 0 else subQ_nodes_tmp[0]
                tp_node = tp_nodes[0] if tp_nodes and tp_nodes[0] not in subQ_nodes_tmp0 else []

            # Find the edge of unknown node
            amr_graph_tmp = penman.encode(amr_graph_copy_for_subQ, top=amr_info['unknownNode'])
            parse_amr_graph_tmp = penman.parse(amr_graph_tmp)
            var, branches = parse_amr_graph_tmp.node

            unknown_edge_tmp = []
            for branch in branches[1:]:
                role, target = branch
                if '/' in str(target):
                    t_graph = role + ' ' + penman.format(target)
                    re_t_graph = re.split('\(', "(".join(t_graph.split()))
                else:
                    t_graph = ""
                    re_t_graph = []
                if subQQ_edges[0][1] in t_graph and subQQ_edges[0][2] in re_t_graph:
                    unknown_edge_tmp.append((amr_info['unknownNode'], role.split('-')[0], target[0].split('-')[0]))
                else:
                    unknown_edge_tmp.append(tuple())

            # extract nodes and edges
            for node in com_substr:
                s, r, t = amr_edges[node][0], amr_edges[node][1], amr_edges[node][2]
                del_subQ_edges.append((s, r, t))

                if (s, ":instance", amr_info['node_dict'][s]) not in del_subQ_instances:
                    del_subQ_nodes.append(s)
                    del_subQ_instances.append((s, ":instance", amr_info['node_dict'][s]))
                if (t, ":instance", amr_info['node_dict'][t]) not in del_subQ_instances:
                    del_subQ_nodes.append(t)
                    del_subQ_instances.append((t, ":instance", amr_info['node_dict'][t]))

            unknown_edges = [edge for edge in amr_info['amr_graph'].edges() if amr_info['unknownNode'] in edge]
            adverb_node = amr_info['adverb_nodes'][0] if amr_info['adverb_nodes'] else False
            adverb_node_in_type_rsv2 = True if set(amr_info['adverb_dict'].values()).intersection(type_rsv2) else False
            more_node = amr_info['node_dict_rev']['more'] if 'more' in amr_info['node_dict_rev'].keys() else False
            nodeAftercommonnode = []
            for trip in amr_info['amr_graph'].triples:
                # del edges
                if trip in del_subQ_edges:
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)
                    continue
                # del nodes and edges in the substring
                if trip[0] in substring_nodes[i] or trip[2] in substring_nodes[i]:
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)
                    continue

                if haveSSN:
                    # del instance
                    if trip in del_subQ_instances[1:]:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        continue
                    # del op
                    # if trip[0] in del_subQ_nodes[1:] or trip[2] in del_subQ_nodes[1:]:
                    #     amr_graph_copy_for_subQ.triples.remove(trip)
                    #     amr_graph_copy_for_subQ.epidata.pop(trip)
                    #     continue
                    if trip[0] in del_subQ_nodes[1:] and trip[2] != del_subQ_nodes[0]:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        nodeAftercommonnode.append(trip[2])
                        continue
                    if trip[2] in del_subQ_nodes[1:] and trip[0] != del_subQ_nodes[0]:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        nodeAftercommonnode.append(trip[0])
                        continue
                else:
                    # del instance
                    if trip in del_subQ_instances:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        continue
                    # del op
                    # if trip[0] in del_subQ_nodes or trip[2] in del_subQ_nodes:
                    #     amr_graph_copy_for_subQ.triples.remove(trip)
                    #     amr_graph_copy_for_subQ.epidata.pop(trip)
                    #     continue
                    if trip[0] in del_subQ_nodes:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        nodeAftercommonnode.append(trip[2])
                        continue
                    if trip[2] in del_subQ_nodes:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        nodeAftercommonnode.append(trip[0])
                        continue

                # del type node
                if rule_type in ["type4", "type5"] and tp_node in trip and (del_subQ_instances[0][0] != tp_node):
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)
                    continue

                # del edges which are related to degreeNode
                if amr_info['degreeNode'] in trip:
                    if amr_info['degreeNode'] != amr_info['node_list'][0]:
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)
                        node_edge_dN = trip[2] if trip[0] == amr_info['degreeNode'] and "instance" not in trip[1] else trip[0]
                        # nodes_edge_with_degreeNode = find_other_n_e_d(node_edge_dN, amr_info, nodes_edge_with_degreeNode)
                        node_edge_dN_flag = True
                        for ed in amr_info['edge_list_nodigit1']:
                            el_no = ed.split('-')
                            if el_no[1] == node_edge_dN and el_no[0] != amr_info['degreeNode']:
                                node_edge_dN_flag = False
                        if node_edge_dN_flag and node_edge_dN != amr_info['degreeNode'] and amr_info['node_list'].index(
                                node_edge_dN) > amr_info['node_list'].index(amr_info['degreeNode']):    # '5ac3d4b6554299204fd21e93'
                            nodes_edge_with_degreeNode.append(node_edge_dN)
                        continue
                    else:
                        if (adverb_node in trip or more_node in trip) and not adverb_node_in_type_rsv2:
                            amr_graph_copy_for_subQ.triples.remove(trip)
                            amr_graph_copy_for_subQ.epidata.pop(trip)
                            node_edge_dN = trip[2] if trip[0] == amr_info['degreeNode'] and "instance" not in trip[1] else trip[0]
                            node_edge_dN_flag = True
                            for ed in amr_info['edge_list_nodigit1']:
                                el_no = ed.split('-')
                                if el_no[1] == node_edge_dN and el_no[0] != amr_info['degreeNode']:
                                    node_edge_dN_flag = False
                            if node_edge_dN_flag and node_edge_dN != amr_info['degreeNode'] and node_edge_dN in \
                                    amr_info['node_list'] and amr_info['node_list'].index(node_edge_dN) > amr_info[
                                'node_list'].index(amr_info['degreeNode']):
                                nodes_edge_with_degreeNode.append(node_edge_dN)
                            continue

                # # del nodes_edge with adverb_node
                # if adverb_node in trip:
                #     amr_graph_copy_for_subQ.triples.remove(trip)
                #     amr_graph_copy_for_subQ.epidata.pop(trip)
                #
                #     node_edge_adN = trip[2] if trip[0] == adverb_node and "instance" not in trip[1] else trip[0]
                #     nodes_edge_with_degreeNode.append(node_edge_adN)
                #     continue

                # del edges which are related to more_node
                if more_node and more_node in trip:
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)
                    continue

                # del edges of adverb
                if amr_info['adverb_nodes'] and not adverb_node_in_type_rsv2 and (amr_info['adverb_nodes'][0] in trip and trip[1] != ':instance'):
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)
                    node_edge_adN = trip[2] if trip[0] == adverb_node else trip[0]
                    if node_edge_adN in amr_info['node_list'] and amr_info['node_list'].index(node_edge_adN) > amr_info['node_list'].index(adverb_node):
                        nodes_edge_with_degreeNode.append(node_edge_adN)
                    continue

                # replace the relation node
                if amr_info['unknownNode'] in trip and trip[1] != ":instance":
                    if len(unknown_edges) == 1:
                        amr_graph_copy_for_subQ, rpl_rel = delete_rel_node(amr_graph_copy_for_subQ, trip, type, amr_info['degreeNode'])
                    else:
                        # len(unknown_edges) > 1:
                        if trip in unknown_edge_tmp or tuple(reversed(trip)) in unknown_edge_tmp:     # if trip in unknown_edges
                            amr_graph_copy_for_subQ, rpl_rel = delete_rel_node(amr_graph_copy_for_subQ, trip, type, amr_info['degreeNode'])
                        else:
                            rpl_rel = trip[1]
                        # if trip == unknown_edge_tmp or trip == tuple(reversed(unknown_edge_tmp)):     # if trip in unknown_edges
                        #     amr_graph_copy_for_subQ, rpl_rel = delete_rel_node(amr_graph_copy_for_subQ, trip, type, amr_info['degreeNode'])


            # add comparision node
            # -----
            if rule_type in ["type1", "type2"]:
                subQ_unknown_edge = []
                for trip in amr_graph_copy_for_subQ.triples:
                    if amr_info['unknownNode'] in trip and trip[1] != ":instance":
                        subQ_unknown_edge.append(trip)

                flag = False
                if amr_info['degreeNode'] and type in type_rsv:
                    if len(subQ_unknown_edge) == 1:
                        amr_graph_copy_for_subQ = finetuning_edge(amr_graph_copy_for_subQ, amr_info, subQ_unknown_edge[0],
                                                                  adverb_node, rpl_rel)
                    else:
                        for edge in subQ_unknown_edge:
                            if edge[0] in amr_info['verb_list'] or edge[2] in amr_info['verb_list']:
                                amr_graph_copy_for_subQ = finetuning_edge(amr_graph_copy_for_subQ, amr_info, edge, adverb_node, rpl_rel)
                                flag = True
                        if not flag and subQ_unknown_edge:
                            amr_graph_copy_for_subQ = finetuning_edge(amr_graph_copy_for_subQ, amr_info, subQ_unknown_edge[0], adverb_node, rpl_rel)
                else:
                    for edge in subQ_unknown_edge:
                        if edge[0] in amr_info['verb_list'] or edge[2] in amr_info['verb_list']:
                            comp_node_trip = (edge[0], rpl_rel, edge[2])
                            amr_graph_copy_for_subQ.triples.remove(edge)
                            amr_graph_copy_for_subQ.epidata.pop(edge)
                            amr_graph_copy_for_subQ.triples.append(comp_node_trip)
                            amr_graph_copy_for_subQ.epidata[comp_node_trip] = []
                            flag = True
                    if not flag and subQ_unknown_edge:
                        trip_del = subQ_unknown_edge[0]
                        comp_node_trip = (trip_del[0], rpl_rel, trip_del[2])
                        amr_graph_copy_for_subQ.triples.remove(trip_del)
                        amr_graph_copy_for_subQ.epidata.pop(trip_del)
                        amr_graph_copy_for_subQ.triples.append(comp_node_trip)
                        amr_graph_copy_for_subQ.epidata[comp_node_trip] = []

            # -----

            # delete 'or' and 'and' node  'betwwen
            or_and_list = []
            for ins in list(amr_graph_copy_for_subQ.instances()):
                #  tranform amr-* to amr-unknown
                if 'amr' in ins.target and ins.target != "amr-unknown":
                    _trip = (ins.source, ":instance", "amr-unknown")
                    amr_graph_copy_for_subQ.triples.remove(ins)
                    amr_graph_copy_for_subQ.epidata.pop(ins)
                    amr_graph_copy_for_subQ.triples.append(_trip)
                    amr_graph_copy_for_subQ.epidata[_trip] = []

                # find 'or' and 'and' node
                if ('and' in ins or 'or' in ins or 'both' in ins or 'either' in ins or 'between' in ins) and ins.source != amr_info['node_list'][0]\
                        and ins.source in del_subQ_nodes:
                    or_and_list.append(ins.source)
            # assert len(or_and_list) == 1
            if or_and_list:
                or_and_trip = []
                other_or_and_trip = []
                start_of_or_and = []
                amr_graph_copy_for_subQ1 = copy.deepcopy(amr_graph_copy_for_subQ)

                # for trip in amr_info['amr_graph'].triples:
                for trip in amr_graph_copy_for_subQ1.triples:
                    for o_a_node in or_and_list:
                        if trip in amr_graph_copy_for_subQ.triples and o_a_node in trip:
                            if trip[1] != ":instance":
                                j = 1 if i == 0 else 0
                                # if not ("op" in trip[1] and trip[2] not in subQ_nodes_tmp[j]):
                                #     or_and_trip.append(trip)
                                #     amr_graph_copy_for_subQ.triples.remove(trip)
                                #     amr_graph_copy_for_subQ.epidata.pop(trip)
                                # else:
                                #     other_or_and_trip.append(trip)              # '5a7555215542996c70cfaee1'
                                #     amr_graph_copy_for_subQ.triples.remove(trip)
                                #     amr_graph_copy_for_subQ.epidata.pop(trip)

                                if "op" in trip[1] or "mod" in trip[1]:                                 # id:60
                                    if trip[2] not in subQ_nodes_tmp[j]:
                                        other_or_and_trip.append(trip)
                                        amr_graph_copy_for_subQ.triples.remove(trip)
                                        amr_graph_copy_for_subQ.epidata.pop(trip)
                                    else:
                                        or_and_trip.append(trip)
                                        amr_graph_copy_for_subQ.triples.remove(trip)
                                        amr_graph_copy_for_subQ.epidata.pop(trip)
                                else:
                                    start_of_or_and.append(trip)
                                    amr_graph_copy_for_subQ.triples.remove(trip)
                                    amr_graph_copy_for_subQ.epidata.pop(trip)
                            else:
                                amr_graph_copy_for_subQ.triples.remove(trip)
                                amr_graph_copy_for_subQ.epidata.pop(trip)


                if len(or_and_trip) > 1:
                    for trp in or_and_trip:
                        if not (trp[0] in subQQ_nodes_in_edges and trp[2] in subQQ_nodes_in_edges):
                            or_and_trip.remove(trp)
                assert len(or_and_trip) == 1
                all_nodes_in_SubQ_t = [[ins.source, ins.target] for ins in amr_graph_copy_for_subQ.edges()]
                all_nodes_in_SubQ_t = set([n for nl in all_nodes_in_SubQ_t for n in nl])

                before_start_of_or_and = []
                nn_t = or_and_trip[0][2] if or_and_trip[0][2] != or_and_list[0] else or_and_trip[0][0]

                if start_of_or_and:
                    if start_of_or_and[0][0] == or_and_list[0]:
                        nn_s, nn_r = start_of_or_and[0][2], start_of_or_and[0][1]
                    if start_of_or_and[0][2] == or_and_list[0]:
                        nn_s, nn_r = start_of_or_and[0][0], start_of_or_and[0][1]
                else:
                    for eg0 in amr_info['edge_list_nodigit1']:
                        s, t = eg0.split('-')
                        if t == or_and_list[0] and s not in all_nodes_in_SubQ_t and (amr_info['node_list'].index(s) < amr_info['node_list'].index(t)):
                            before_start_of_or_and.append(s)
                        if s == or_and_list[0] and t not in all_nodes_in_SubQ_t and (amr_info['node_list'].index(t) < amr_info['node_list'].index(s)):
                            before_start_of_or_and.append(t)

                    if before_start_of_or_and:
                        amr_graph_top_bs = penman.encode(amr_info['amr_graph'], top=before_start_of_or_and[0])
                        parse_amr_graph_top_bs = penman.parse(amr_graph_top_bs)
                        var_bs, branches_bs = parse_amr_graph_top_bs.node

                        for bch in branches_bs[1:]:
                            role_bs, target_bs = bch
                            if amr_info['node_list'].index(target_bs[0]) < amr_info['node_list'].index(var_bs):
                                nn_s, nn_r = target_bs[0], role_bs
                    else:
                        nn_s, nn_r = amr_info['node_list'][0], "ARG"

                amr_graph_copy_for_subQ.triples.append((nn_s, nn_r, nn_t))
                amr_graph_copy_for_subQ.epidata[(nn_s, nn_r, nn_t)] = []

                if start_of_or_and:
                    other_or_and_trip.extend(start_of_or_and[1:])

                for other_trip in other_or_and_trip:
                    other_nn_t = other_trip[0] if other_trip[0] != or_and_trip[0][0] else other_trip[2]     # id:60
                    amr_graph_copy_for_subQ.triples.append((nn_s, nn_r, other_nn_t))
                    amr_graph_copy_for_subQ.epidata[(nn_s, nn_r, other_nn_t)] = []

            fl = []
            # delete nodes_edge_with_degreeNode
            for trip in list(amr_graph_copy_for_subQ.triples):
                if amr_info['degreeNode'] != amr_info['node_list'][0] and (
                        trip[0] in nodes_edge_with_degreeNode or trip[2] in nodes_edge_with_degreeNode):
                    amr_graph_copy_for_subQ.triples.remove(trip)
                    amr_graph_copy_for_subQ.epidata.pop(trip)

            # About nodeAftercommonnode
            for na in nodeAftercommonnode:
                for trip in list(amr_graph_copy_for_subQ.triples):
                    trip_l = [trip[0], trip[2]]
                    substring_nodes_tmp = del_subQ_nodes + substring_nodes[i]
                    if na in trip and (trip_l[0] in substring_nodes_tmp or trip_l[1] in substring_nodes_tmp):
                        amr_graph_copy_for_subQ.triples.remove(trip)
                        amr_graph_copy_for_subQ.epidata.pop(trip)

            # delete node without edges
            all_nodes_in_SubQ = [[ins.source, ins.target] for ins in amr_graph_copy_for_subQ.edges()]
            all_nodes_in_SubQ = set([n for nl in all_nodes_in_SubQ for n in nl])
            for ins in list(amr_graph_copy_for_subQ.instances()):
                if ins.source not in all_nodes_in_SubQ:
                    amr_graph_copy_for_subQ.triples.remove(ins)
                    amr_graph_copy_for_subQ.epidata.pop(ins)

            # delete edge whose node is not in the edge
            for edge in list(amr_graph_copy_for_subQ.epidata):
                if edge[0] not in all_nodes_in_SubQ and edge[2] not in all_nodes_in_SubQ:
                    amr_graph_copy_for_subQ.triples.remove(edge)
                    amr_graph_copy_for_subQ.epidata.pop(edge)
            # -----

            subQ_nodes = [ins.source for ins in amr_graph_copy_for_subQ.instances()]
            if subQ_nodes:
                if amr_info['unknownNode'] in subQ_nodes:
                    try:
                        graph_subQ = penman.encode(amr_graph_copy_for_subQ, top=amr_info['unknownNode'])
                    except:
                        return default_sub_Q(amr_info, type)
                else:
                    # if amr_info['unknownNode']:
                    #     graph_subQ = penman.encode(amr_graph_copy_for_subQ, top=subQ_nodes[0])
                    # else:
                    unknown_node = 'a' if 'a' not in subQ_nodes else 'a0'
                    _trip = (unknown_node, ":instance", "amr-unknown")
                    amr_graph_copy_for_subQ.triples.append(_trip)
                    amr_graph_copy_for_subQ.epidata[_trip] = []

                    if subQ_nodes[0] in amr_info['verb_list']:
                        _trip1 = (unknown_node, "ARG", subQ_nodes[0])
                    else:
                        _trip1 = (unknown_node, "polarity", subQ_nodes[0])

                    amr_graph_copy_for_subQ.triples.append(_trip1)
                    amr_graph_copy_for_subQ.epidata[_trip1] = []
                    try:
                        graph_subQ = penman.encode(amr_graph_copy_for_subQ, top=unknown_node)
                    except:
                        return default_sub_Q(amr_info, type)
            else:
                graph_subQ = penman.encode(amr_info['amr_graph'], top=amr_info['unknownNode'])

            subQ, _ = gtos.generate([graph_subQ], disable_progress=True)

            subQ_dict["subQ{}".format(str(i+1))] = subQ
            subQ_dict["graph_subQ{}".format(str(i + 1))] = graph_subQ

        subQ_dict["sec_unknown"] = "comparison"
        subQ_dict["question"] = amr_info['sent']
        subQ_dict["type"] = type
        subQ_dict["rule_type"] = rule_type
        subQ_dict["key"] = key
        return subQ_dict


def extract_rel_dict(graph, amr_graph, sub_graphs):
    source_list = []
    target_list = []
    leaf_list_nodes = []
    # extract edge node
    for edge in amr_graph.edges():
        source_list.append(edge.source)
        target_list.append(edge.target)

    leaf_list = [trg for trg in target_list if trg not in source_list]

    for trip in amr_graph.triples:
        if trip[0] in leaf_list and trip[2] != "name":
            if "op" in trip[1] and trip[2][0] == '"':
                leaf_list_nodes.append(eval(trip[2]))
            else:
                leaf_list_nodes.append(trip[2])

    graph_string = graph.split('\n')
    rep_list = []
    import re
    for z in graph_string[1:]:
        if "/" not in z and z not in rep_list:
            if not set(z.split('\"')).intersection(set(leaf_list_nodes)):
                rel_targ = re.sub(u"\)", "", z).strip()
                rep_list.append((str(rel_targ)))

    rel_dict = dict()
    for rep in rep_list:
        rel, tar = rep.split()
        for sub_g in sub_graphs:
            if sub_g[1] == rel and sub_g[2][0] == tar:
                if len(sub_g[2]) == 1:
                    for sub_g_tmp in sub_graphs:
                        if sub_g[2] == sub_g_tmp[2][0] and len(sub_g_tmp[2]) != 1:
                            rel_dict[rep] = sub_g_tmp[2]
                else:
                    rel_dict[rep] = sub_g[2]

    return rel_dict


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
    node_list, edge_list, node_dict, node_dict_rev = [], [], dict(), dict()
    verb_list, noun_list = [], []
    edge_list_nodigit = []
    edge_list_nodigit1 = []
    edgesOfDegree = []
    amr_info = dict()
    unknownNode, degreeNode = None, None
    before_after_list = []
    sent = graph.split("\n")[0][8:]     # sent
    amr_graph = penman.decode(graph)  # generate AMR_graph

    if len(amr_graph.epidata) != len(amr_graph.triples):
        amr_graph = alignment_amr_graph(amr_graph)

    for ins in amr_graph.instances():
        node_list.append(ins.source)
        node_dict[ins.source] = ins.target
        node_dict_rev[ins.target] = ins.source
        #
        if 'amr' in ins.target: unknownNode = ins.source
        if 'have-degree-91' in ins.target or "have-quant-91" in ins.target:  degreeNode = ins.source

        if 'before' in ins.target or 'after' in ins.target:
            before_after_list.append(ins.source)

    other_degreeNode = []
    for ba in before_after_list:        # ?
        for edge in amr_graph.edges():
            if (edge.role == ":time" and edge.target == ba) or (edge.role == ":time" and edge.source == ba):
                ot_degreeNode = 'earlier' if 'b' in ba else 'later'
                other_degreeNode.append(ot_degreeNode)

    # extract verb node and noun node
    verbInsent, nounInsent, AndNode = VerbNounInSent(amr_graph, sent)
    verb_list.extend(verbInsent)
    noun_list.extend(nounInsent)
    AndNode = AndNode[:1]

    # -----
    sub_graphs = []

    def extract_graph(graph):
        parse_graph = penman.parse(graph)
        var, branches = parse_graph.node
        for branch in branches[1:]:
            role, target = branch
            headNodeOfSubTree = [sg[2][0] for sg in sub_graphs]

            if type(target) != str:
                extract_graph(penman.format(target))

            if type(target) == str and target in headNodeOfSubTree:
                index = headNodeOfSubTree.index(target)
                sub_graphs.append((var, role, sub_graphs[index][2]))
            else:
                sub_graphs.append((var, role, target))

        return

    extract_graph(graph)
    rel_dict = extract_rel_dict(graph, amr_graph, sub_graphs)     # similar to bridge code
    # -----

    for edge in amr_graph.edges():
        edge_list.append((edge.source, edge.role, edge.target))
        edge_list_nodigit.append((edge.source[0] + '-' + edge.target[0]))
        edge_list_nodigit1.append((edge.source + '-' + edge.target))
        target_tmp = copy.deepcopy(edge.target)

        def generate_chain(edge_list, target_tmp):            # 1115
            for i, (s, r, t) in enumerate(edge_list[:-1]):
                if target_tmp == s:
                    edge_list.append((s, r, t))
                    edge_list_nodigit.append(edge_list_nodigit[i])
                    edge_list_nodigit1.append(edge_list_nodigit1[i])
                    target_tmp = t
                    generate_chain(edge_list, target_tmp)
            return

        if target_tmp in rel_dict.values():         # new part
            generate_chain(edge_list, target_tmp)

        if degreeNode and degreeNode in edge:
            if degreeNode == edge.source:
                edgesOfDegree.append(node_dict[edge.target].split('-')[0])
            else:
                edgesOfDegree.append(node_dict[edge.source].split('-')[0])
    #
    sent_pos = nlp.annotate(sent, props)
    parsed_sent = json.loads(sent_pos)
    adverb_list = [tok['word'] for tok in parsed_sent['sentences'][0]['tokens'] if
                   (tok['pos'] in ['RBR', 'RBS', 'JJR', 'JJS'] or tok['word'] in ['first', 'last', 'later'])]

    adverb_dict = dict()
    for advb in adverb_list:    # 1115
        advb_key = spacy_nlp(advb)[0].lemma_
        if degreeNode:
            if advb_key in edgesOfDegree:
                adverb_dict[advb_key] = advb
        else:
            adverb_dict[advb_key] = advb



    # adverb_list = [spacy_nlp(advb)[0].lemma_ for advb in adverb_list]
    # extract adverb nodes and edges
    adverb_tags = [':ord', ":quant"]       # adverb_tags = [':ord', ":quant", ":time"]      # Q2
    adverb_nodes = []
    for trip in amr_graph.edges():
        if trip[1] in adverb_tags or node_dict[trip[2]].split('-')[0] in adverb_dict.keys():          # !!!!!flag-> the latter expression
            adverb_nodes.append(trip[2])
    #

    amr_info['graph'] = graph
    amr_info['sent'] = sent
    amr_info['amr_graph'] = amr_graph
    amr_info['node_list'] = node_list
    amr_info['edge_list'] = edge_list
    amr_info['verb_list'] = list(set(verb_list))
    amr_info['noun_list'] = list(set(noun_list))
    amr_info['AndNode'] = AndNode
    #
    amr_info['adverb_dict'] = adverb_dict
    amr_info['adverb_nodes'] = adverb_nodes
    amr_info['unknownNode'] = unknownNode
    amr_info['degreeNode'] = degreeNode
    amr_info['edgesOfDegree'] = edgesOfDegree
    #
    amr_info['edge_list_nodigit'] = edge_list_nodigit
    amr_info['edge_list_nodigit1'] = edge_list_nodigit1
    amr_info['node_dict'] = node_dict
    amr_info['node_dict_rev'] = node_dict_rev
    # adj_graph_array = construct_adj_graph(amr_info)
    # amr_info['adj_graph'] = adj_graph_array

    return amr_info


def construct_adj_graph(amr_info):
    node = amr_info['node_list']
    edge = amr_info['edge_list']
    node_map = [[0 for val in range(len(node))] for val in range(len(node))]

    for x, y, val in edge:
        node_map[node.index(x)][node.index(y)] = node_map[node.index(y)][node.index(x)] = val

    return node_map


def isSamePath(common_substring):
    # '5a879ba55542993e715abfc3'
    right_path = []
    com_subs_list = common_substring.split()
    for i in range(len(com_subs_list) - 1):
        com_str_i = com_subs_list[i].split('-')
        com_str_i_1 = com_subs_list[i + 1].split('-')
        if com_str_i[-1] != com_str_i_1[0]:
            return False
        # if com_str_i[-1] == com_str_i_1[0]:
        #     right_path.append(com_subs_list[i])

    return True     # True if len(right_path) == (len(com_subs_list) - 1) else False


def find_common_substr(amr_info):
    amr_edge_nodigit = amr_info['edge_list_nodigit']
    string = ' '.join(amr_edge_nodigit)

    c = CommonSubStr()
    _, common_substring_list = c.getMaxCommonStr(string)

    common_substring_pos = []
    common_substring_list = list(set(common_substring_list))
    common_substring_list1 = copy.deepcopy(common_substring_list)
    for com_string in common_substring_list1:
        cstring_tmp = ''
        for cstring in com_string.split():
            if len(cstring) != 3 and com_string in common_substring_list:
                common_substring_list.remove(com_string)
            else:
                if len(cstring) == 3:
                    cstring_tmp = cstring_tmp + ' ' + cstring

        if cstring_tmp:
            common_substring_list.append(cstring_tmp.strip())
    # for com_string in common_substring_list1:
    #     for cstring in com_string.split():
    #         if len(cstring) != 3 and com_string in common_substring_list:
    #             common_substring_list.remove(com_string)
    #         else:
    #             common_substring_list.append(cstring)

    common_substring_list = list(set([com.strip() for com in common_substring_list]))
    string_len = [len(s) for s in common_substring_list]
    sorted_string = sorted(list(set(string_len)), reverse=True)

    longest_common_substring = ""
    flag = True
    # for ss0 in sorted_string:
    #     for i, ss1 in enumerate(string_len):
    #         if ss0 == ss1 and flag and ('o' in common_substring_list[i] or 'a' in common_substring_list[i]):
    #             longest_common_substring = common_substring_list[i]
    #             flag = False
    #             break
    for ss0 in sorted_string:
        for i, ss1 in enumerate(string_len):
            if ss0 == ss1 and flag and len(common_substring_list[i]) >= 3 and ('o' in common_substring_list[i] or 'a' in common_substring_list[i]):
                common_substring = common_substring_list[i]
                is_s_p = isSamePath(common_substring)       # to judge if the path is correct
                if is_s_p:
                    for str in common_substring.split():
                        if str in amr_info['edge_list_nodigit']:
                            ind = amr_info['edge_list_nodigit'].index(str)
                            t_str = amr_info['edge_list_nodigit1'][ind]
                            for s in t_str.split('-'):
                                if 'a' in s and 'and' in amr_info['node_dict'][s]:
                                    longest_common_substring = common_substring_list[i]
                                    flag = False
                                    break
                                if 'o' in s and 'or' in amr_info['node_dict'][s]:
                                    longest_common_substring = common_substring_list[i]
                                    flag = False
                                    break

    if not longest_common_substring and sorted_string:
        for ss in sorted_string:
            if ss >= 3:
                longest_subs_ind = ss
                list_ind = string_len.index(longest_subs_ind)
                is_s_p0 = isSamePath(common_substring_list[list_ind])
                if is_s_p0:
                    longest_common_substring = common_substring_list[list_ind]
                    break

        # if sorted_string[0] >= 3:
        #     longest_subs_ind = sorted_string[0]
        #     list_ind = string_len.index(longest_subs_ind)
        #     longest_common_substring = common_substring_list[list_ind]

    # longest_common_substring = common_substring_list[string_len.index(max(string_len))]
    print("longest common substring: %s" % longest_common_substring)
    common_substring = longest_common_substring.split()

    for i in range(len(amr_edge_nodigit) - len(common_substring) + 1):
        if amr_edge_nodigit[i:i + len(common_substring)] == common_substring:
            common_substring_pos.append(tuple(range(i, i + len(common_substring))))

    return common_substring, common_substring_pos[:2]


def add_question_mark(question):
    if '?' not in question:
        question = (question[:-1] + '?') if question[-1] in punc else (question + '?')

    return question


def save_subQ_to_excel(subQ_dict):
    pd_list = []

    for samp in subQ_dict:
        pd_list.append([samp['key'], samp['subQ1'], samp['sec_unknown'], samp['subQ2'], samp['question']])

    df = pd.DataFrame(pd_list, columns=['key','subQ1', 'sec_unknown', 'subQ2', 'ques'])
    df.to_excel("sub_question_comp_final_1117.xlsx", encoding='utf-8', index=False)

    with open(r"subQ_dict_comp_final_list_1105.pkl", 'wb') as f:
        pickle.dump(subQ_dict, f)

    # with open(r"subQ_dict_comp_final_list.pkl", 'rb') as f1:
    #     subQ_dict = pickle.load(f)

    print("hello")


def judge_ques_type(value):
    questionType1 = ["less", "earlier", "earliest", "first", "shorter", "smaller", "older", "oldest", "closer"]
    questionType2 = ["more", "most", "later", "last", "latest", "longer", "longest", "larger", "younger", "newer", "taller", "higher", "highest"]
    questionType3 = ["are", "were", 'is', 'was', 'did', 'does', 'do', 'which', 'what', 'who', 'where']
    questionType4 = ['both', 'either']
    questionType5 = ['same', 'difference', 'different']

    questionType6 = ['or', 'and']

    questionType1_value = [kw for kw in questionType1 if kw in value]
    questionType2_value = [kw for kw in questionType2 if kw in value]
    questionType3_value = [kw for kw in questionType3 if kw in value.lower()]
    questionType4_value = [kw for kw in questionType4 if kw in value.lower()]
    questionType5_value = [kw for kw in questionType5 if kw in value.lower()]
    questionType6_value = [kw for kw in questionType3 if
                           kw in value.lower() and ('or' in value.lower() or 'and' in value.lower())]

    if not questionType1_value and not questionType2_value:
        judge_ques_tmp.append(value)

    if not questionType1_value and not questionType2_value and not questionType3_value:
        judge_ques_tmp1.append(value)

    if not questionType1_value and not questionType2_value and not questionType4_value and not questionType5_value and not questionType6_value:
        judge_ques_tmp2.append(key + '     ' + value)

    if not questionType1_value and not questionType2_value and not questionType3_value and not questionType4_value and not questionType5_value:
        judge_ques_tmp3.append(key + '     ' + value)

    print("i=%d" % i)


if __name__ == '__main__':
    full_file = 'hotpotqa_data/hotpot_train_v1.1.json'
    test_file = 'hotpotqa_data/hotpot_dev_distractor_v1.json'

    bridge_file = dict()
    comparison_file = dict()

    with open(test_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)  # [:100]

    for file in full_data:
        # if file['type'] == 'bridge':
        #     question = add_question_mark(file['question'])
        #     bridge_file[file['_id']] = question

        if file['type'] == 'comparison':
            question = add_question_mark(file['question'])
            comparison_file[file['_id']] = question

    stog = amrlib.load_stog_model()
    gtos = amrlib.load_gtos_model()
    gen_subQ = GenerateSubQuestion()
    subQ_dict = []
    error_ques = []

    print("The length of comparison questions is: %d" % len(comparison_file))
    judge_ques_tmp, judge_ques_tmp1, judge_ques_tmp2, judge_ques_tmp3 = [], [], [], []
    for i, (key, value) in enumerate(comparison_file.items()):
        if i >= 0:
            print("\n *****i=%d*****" % i)

            graph = stog.parse_sents([value])           #

            # data preprocess
            amr_info = data_process(graph[0])

            # find longest common substring
            common_substring, common_substring_pos = find_common_substr(amr_info)

            # ---------------types of question ---------------
            questionType1 = ["less", "earlier", "earliest", "first", "shorter", "smaller", "older", "oldest", "closer", 'closest']
            questionType2 = ["more", "most", "later", "last", "latest", "longer", "longest", "larger", "younger", "newer",
                             "taller", "higher", 'tallest', "highest"]
            questionType3 = ["are", "were", 'is', 'was', 'did', 'does', 'do', 'which', 'what', 'who', 'where']
            questionType4 = ['both', 'either']
            questionType5 = ['same', 'difference', 'different']

            questionType1_value = [kw for kw in questionType1 if kw in amr_info['sent'] and kw in amr_info['adverb_dict'].values()]
            questionType2_value = [kw for kw in questionType2 if kw in amr_info['sent'] and kw in amr_info['adverb_dict'].values()]
            questionType3_value = [kw for kw in questionType3 if kw in value.lower() and ('or' in value.lower() or 'and' in value.lower())]
            questionType4_value = [kw for kw in questionType4 if kw in value.lower()]
            questionType5_value = [kw for kw in questionType5 if kw in value.lower()]

            # --------------------------------------------------
            if common_substring:
                # Type1
                if set(questionType1_value).intersection(questionType1):
                    print("t1...")
                    Q_type = list(set(questionType1_value).intersection(questionType1))[0]
                    subQ_dict.append(gen_subQ.comp_type01_subQ(amr_info, Q_type, common_substring_pos, "type1", key))
                    # print("t1...")
                    continue

                # Type2
                if set(questionType2_value).intersection(questionType2):
                    print("t2...")
                    Q_type = list(set(questionType2_value).intersection(questionType2))[0]
                    subQ_dict.append(gen_subQ.comp_type01_subQ(amr_info, Q_type, common_substring_pos, "type2", key))
                    # print("t2...")
                    continue

                # Type 4
                if set(questionType4_value).intersection(questionType4):
                    print("t4...")
                    Q_type = list(set(questionType4_value).intersection(questionType4))[0]
                    subQ_dict.append(gen_subQ.comp_type01_subQ(amr_info, Q_type, common_substring_pos, "type4", key))
                    continue

                # Type 5
                if set(questionType5_value).intersection(questionType5):
                    print("t5...")
                    Q_type = list(set(questionType5_value).intersection(questionType5))[0]
                    subQ_dict.append(gen_subQ.comp_type01_subQ(amr_info, Q_type, common_substring_pos, "type5", key))
                    continue

                # Type 3    last one
                if set(questionType3_value).intersection(questionType3):
                    print("t3...")
                    Q_type = list(set(questionType3_value).intersection(questionType3))[0]
                    subQ_dict.append(gen_subQ.comp_type01_subQ(amr_info, Q_type, common_substring_pos, "type3", key))
                    continue

                print("The question doesn't belong to any type: %s" % key)
                error_ques.append(key)
            else:
                print("Can't find common substring!")
                Q_type = "onehop"
                rule_type = "type6"
                subQ_tmp = {"subQ1": [amr_info['sent']], "sec_unknown": 'comparision', "subQ2": [amr_info['sent']],
                            "graph_subQ1": amr_info['graph'], "graph_subQ2": amr_info['graph'],
                            "question": amr_info['sent'], "type": Q_type, "rule_type": rule_type, "key": key}

                subQ_dict.append(subQ_tmp)
                print("The question doesn't belong to any type: %s" % key)
                error_ques.append(key)

    # save_subQ_to_excel(subQ_dict)


