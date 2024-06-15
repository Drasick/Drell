import json
import pandas as pd
import numpy as np

"""
主要调用三个函数
 * return_doc_logits 读取 logits 从而得到不同的模型对于每个文档的每个三元组的97种关系的预测结果
 * return_doc_logits_2024 因为后来自己实验的格式不太一样，所以修改了 return_doc_logits 功能一样
 * return_eider_logits 因为 Eider 的实验结果是同时处理 document 和 伪文档的，因此得到 logits 的逻辑不太一样
"""

# 返回关系预测分数
def return_doc_logits(test_data_path = "dataset/docred/dev.json", # 测试数据集的路径
                  rel2id_path = "../ass3/ATLOP/meta/rel2id.json", # 关系和数字对应的字典路径
                  logits_path = "./dataNEW/res_slm_llm_dreeam/logits.json" # 小模型预测关系的结果(每种关系的预测分数) 
                 ):
    
    """
    INPUT - 载入测试集和关系预测结果
    OUTPUT - 一个 list list中元素的数量和文档数量一致, list的每个元素是一个 dict, dict中的key为(h_idx,t_idx),value为 97 种关系的预测分数
    """
    test_data = open(test_data_path, 'r', encoding='utf-8')
    json_info = test_data.read()
    df = pd.read_json(json_info)
    
    # 关系名称 和 ID 对应
    rel2id = json.load(open(rel2id_path, 'r'))
    
    # ID 和 关系名称 对应
    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        
    df_rel = pd.read_json(logits_path)
    
    
    doc_index = 0
    j = 0
    doc_relations = []
    doc_relation = {}
    while True:
        if j == len(df_rel['doc_index']):
            doc_relations.append(doc_relation)
            doc_relation = {}
            break

        if df_rel['doc_index'][j] != doc_index + 1:
            doc_index += 1
            doc_relations.append(doc_relation)
            doc_relation = {}
            continue


        # 使用enumerate获取元素索引和值，并根据值进行排序
        sorted_numbers = sorted(enumerate(df_rel['relations'][j]), key=lambda x: x[1], reverse=True)

        sorted_names = [id2rel[index] for index, _ in sorted_numbers]
        sorted_values = [value for _, value in sorted_numbers]

        relations = {}
        for i in range(len(sorted_names)):
            relations[sorted_names[i]] = sorted_values[i]

        doc_relation[(df_rel['h_index'][j], df_rel['t_index'][j])] = relations

        j += 1
        
    return doc_relations


# 返回关系预测分数
def return_doc_logits_2024(test_data_path = "dataset/docred/dev.json", # 测试数据集的路径
                  rel2id_path = "../ass3/ATLOP/meta/rel2id.json", # 关系和数字对应的字典路径
                  logits_path = "./dataNEW/res_slm_llm_dreeam/logits.json" # 小模型预测关系的结果(每种关系的预测分数) 
                 ):
    
    """
    INPUT - 载入测试集和关系预测结果
    OUTPUT - 一个 list list中元素的数量和文档数量一致, list的每个元素是一个 dict, dict中的key为(h_idx,t_idx),value为 97 种关系的预测分数
    """
    test_data = open(test_data_path, 'r', encoding='utf-8')
    json_info = test_data.read()
    df = pd.read_json(json_info)
    
    # 关系名称 和 ID 对应
    rel2id = json.load(open(rel2id_path, 'r'))
    
    # ID 和 关系名称 对应
    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        
    df_rel = pd.read_json(logits_path)
    
    
    doc_index = 0
    j = 0
    doc_relations = []
    doc_relation = {}
    while True:
        if j == len(df_rel['doc_idxs']):
            doc_relations.append(doc_relation)
            doc_relation = {}
            break

        if df_rel['doc_idxs'][j] != doc_index:
            doc_index += 1
            doc_relations.append(doc_relation)
            doc_relation = {}
            continue


#         # 使用enumerate获取元素索引和值，并根据值进行排序
#         sorted_numbers = sorted(enumerate(df_rel['logprobs_r'][j]), key=lambda x: x[1], reverse=True)

#         sorted_names = [id2rel[index] for index, _ in sorted_numbers]
#         sorted_values = [value for _, value in sorted_numbers]

#         relations = {}
#         for i in range(len(sorted_names)):
#             relations[sorted_names[i]] = sorted_values[i]

        doc_relation[(df_rel['h_idx'][j], df_rel['t_idx'][j])] = dict(sorted(df_rel['logprobs_r'][j].items(), key=lambda item: item[1], reverse=True))

        j += 1
        
    return doc_relations



def process_data(data):
    sum_dict = {}
    count_dict = {}

    for sublist in data:
        for tup in sublist:
            second_num = tup[1]
            first_num = tup[0]

            if second_num in sum_dict:
                sum_dict[second_num] += first_num
                count_dict[second_num] += 1
            else:
                sum_dict[second_num] = first_num
                count_dict[second_num] = 1

    # 计算平均值
    average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    result_list = [(average_dict[key], key) for key in average_dict]
    return result_list

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 避免数值溢出
    return list(e_x / e_x.sum(axis=0))

def return_eider_logits(rel2id_path = '../ass3/ATLOP/meta/rel2id.json',
                        doclogits_path = './dataNEW/res_slm_llm_edier/title2score_eider_EIDER_bert_eider_test_best.pkl',
                        rulelogits_path = './dataNEW/res_slm_llm_edier/title2score_evi_rule_EIDER_bert_eider_test_best.pkl'):
#     fr = open('./dataset/docred/dev.json', 'r', encoding='utf-8')
#     json_info = fr.read()
#     df = pd.read_json(json_info)

    rel2id = json.load(open(rel2id_path, 'r'))

    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        

    # 由文章直接得到的top-
    all_res = pd.read_pickle(doclogits_path) # 直接文档的结果
    all_res2 = pd.read_pickle(rulelogits_path) # 伪文档的结果


    doc_relations = []

    for key in all_res.keys():
    #     print(key)
        doc_relation = {}
        for pair in all_res[key].keys():
            doc_topk = process_data(all_res[key][pair])
            relation_list = [-100] * 97
            NA_in = False
            min_score = 1000
            # 先查看是否有NA在里面
    #         print(doc_topk)
            for topk in doc_topk:
                if topk[1] == 0:
                    NA_in = True
                    NA_score = topk[0]
                min_score = min(min_score, topk[0])
            # NA 不在里面，那么就是NA当作是最小值还要小一点
            if not NA_in:
                NA_score = min_score - 1
            # 得到每个位置的相对分数
    #         print(NA_score)
            for topk in doc_topk:    
                relation_list[topk[1]] = topk[0] - NA_score
    #         print(pair,relation_list)    
            # 如果存在伪文档    
            if key in all_res2 and pair in all_res2[key]:
                doc_topk = process_data(all_res2[key][pair])
                # 叠加
                NA_in = False
                min_score = 1000
                # 先查看是否有NA在里面
                for topk in doc_topk:
                    if topk[1] == 0:
                        NA_in = True
                        NA_score = topk[0]
                    min_score = min(min_score, topk[0])
                # NA 不在里面，那么就是NA当作是最小值还要小一点
                if not NA_in:
                    NA_score = min_score - 1
                # 得到每个位置的相对分数
                for topk in doc_topk:  
                    if relation_list[topk[1]] == -100:
                        # 第一次出现就叠加
                        relation_list[topk[1]] = topk[0] - NA_score
                    else:
                        # 否则就更新
                        relation_list[topk[1]] += topk[0] - NA_score
                        
            for i in range(len(relation_list)):
                relation_list[i] = relation_list[i] - min(relation_list)
    #         relation_list = softmax(relation_list)
                        
            # 使用enumerate获取元素索引和值，并根据值进行排序
            sorted_numbers = sorted(enumerate(relation_list), key=lambda x: x[1], reverse=True)

            # 获取前五个元素的索引和值
        #     top5_names = [id2rel[index] for index, _ in sorted_numbers[:5]]
        #     top5_values = [value for _, value in sorted_numbers[:5]]
            top5_names = [id2rel[index] for index, _ in sorted_numbers]
            top5_values = [value for _, value in sorted_numbers]

            relations = {}
            for i in range(len(top5_names)):
                relations[top5_names[i]] = top5_values[i]

            doc_relation[pair] = relations
        doc_relations.append(doc_relation)

    return doc_relations