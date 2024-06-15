import os
import json
import pandas as pd
import re
# import openai
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file

# openai.api_key  = os.environ['OPENAI_API_KEY']

# # openai 接口

# def get_completion_from_messages(messages, 
#                                  model="gpt-3.5-turbo", 
#                                  temperature=0, max_tokens = 500):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=temperature, 
# #         max_tokens=max_tokens, 
#     )
#     return response.choices[0].message["content"]

# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0, # this is the degree of randomness of the model's output
#     )
# #     return response.usage.completion_tokens
#     return response.choices[0].message["content"]


def remove_space_before_punctuation(text):
    # 使用正则表达式将符号前的空格替换为空字符串
    cleaned_text = re.sub(r'\s+([.,\'\"])', r'\1', text)
    # 去除左括号右边的空格
    cleaned_text = re.sub(r'\(\s+', '(', cleaned_text)
    # 去除右括号左边的空格
    cleaned_text = re.sub(r'\s+\)', ')', cleaned_text)
    # 去除连字符前后空格
    cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)

    return cleaned_text

# docred数据集的关系和id对应的载入
def return_rel2dict(file_path = './dataset/docred/rel_info.json'):
    fr = open(file_path, 'r', encoding='utf-8')
    rel_info = fr.read()
    rel_info = eval(rel_info)

    # print('the number of relation: ',len(rel_info))

    p_to_num = {}
    num_to_p = {}
    for i,key in enumerate(rel_info.keys()):
        p_to_num[key] = i
        num_to_p[i] = key
    num_to_p[len(rel_info.keys())] = 'NA'
    p_to_num['NA'] = len(rel_info.keys())

    p_to_name = {}
    name_to_p = {}
    for key in rel_info.keys():
        p_to_name[key] = rel_info[key]
        name_to_p[rel_info[key]] = key
    p_to_name['NA'] = 'NA'
    name_to_p['NA'] = 'NA'
    return p_to_num, num_to_p, p_to_name, name_to_p

# 读取关系模板
def return_templates(file_path = './dataNEW/rel_templates.xlsx'):
    df_templates = pd.read_excel(file_path)

    p2templates = {}
    ps = df_templates['关系编号'].values
    templates = df_templates['关系模板'].values
    for i,p in enumerate(ps):
        p2templates[p.strip()] = templates[i]
    
    return p2templates

# 读取数据集的所有数据
def return_docred(file_path = './dataset/docred/dev.json',test_data=False): 
    fr = open(file_path, 'r', encoding='utf-8')
    json_info = fr.read()
    df = pd.read_json(json_info)

    titles = []
    for i in range(len(df['vertexSet'])):
        titles.append(df['title'][i])

    # 实体
    entities = []
    for i in range(len(df['vertexSet'])):
        enames = []
        for entity_class in df['vertexSet'][i]:
            ename = set()
            for entity_name in entity_class:
                ename.add(entity_name['name'])
            enames.append(list(ename))
        entities.append(enames)
        
    # 实体类型引入
    # 实体
    entity_types = []
    for i in range(len(df['vertexSet'])):
        etypes = []
        for entity_class in df['vertexSet'][i]:
            entity_type = set()
            for entity_name in entity_class:
                entity_type.add(entity_name['type'])
            etypes.append(list(entity_type)[0])
        entity_types.append(etypes)
        
    # 实体所在的句子序列导入
    # 实体
    entity_indexs = []
    for i in range(len(df['vertexSet'])):
        eindexs = []
        for entity_class in df['vertexSet'][i]:
            eindex = set()
            for entity_name in entity_class:
                eindex.add(entity_name['sent_id'])
            eindexs.append(list(eindex))
        entity_indexs.append(eindexs)
        
    # 文档
    documents_raw = []
    for i in range(len(df['sents'])):
        document_raw = []
        for j,sentence in enumerate(df['sents'][i]):
            sentence_str = ""
            for word in sentence[:-1]:
                sentence_str += word
                sentence_str += " "
            sentence_str += sentence[-1]
            document_raw.append(remove_space_before_punctuation(sentence_str))
        documents_raw.append(document_raw) 
        
    relations = []
    # 测试集没有 labels 这一列
    if test_data == False:
        for i in range(len(df['sents'])):
            relation = df['labels'][i]
            relations.append(relation)
    else:
        relations = []    
    return titles, entities, entity_types, entity_indexs, documents_raw, relations 


# # 载入预处理文本
# def return_gptprocess(doc_path = './dataNEW/dev_documents.json', entity_tuple_path = './dataNEW/dev_entity_tuples.json'):
#     fr = open(doc_path, 'r', encoding='utf-8')
#     json_info = fr.read()
#     df_docs = pd.read_json(json_info)


#     # 载入抽取的三元组与实体
#     fr = open(entity_tuple_path, 'r', encoding='utf-8')
#     json_info = fr.read()
#     df_tuples = pd.read_json(json_info)
#     doc_entities = list(df_tuples['entities'].values)
#     doc_triples = list(df_tuples['triples'].values)

#     return df_docs, doc_entities, doc_triples

# def find_related_triples(head_names, tail_names, h_indexs, t_indexs, db, hb, triples, sentences):
#     # 返回匹配的三元组字符串
#     """
#     (Udawalawe National Park, lies on the boundary of, Sabaragamuwa Province)
#     (Udawalawe National Park, located in, Sri Lanka)
#     (Udawalawe National Park, important habitat for, Sri Lankan elephants)
#     """
#     # 选取一个最相近得实体放入模糊匹配
#     query_head = db.similarity_search(head_names[0], k = 1)
#     # 返回相似度分数，数值越小，说明越相似
# #     query_head = db.similarity_search_with_score(query)
#     head_sets = set(head_names)
#     head_sets.add(query_head[0].page_content)
#     head_names = list(head_sets)

#     query_tail = db.similarity_search(tail_names[0], k = 1)
#     tail_sets = set(tail_names)
#     tail_sets.add(query_tail[0].page_content)
#     tail_names = list(tail_sets)
        
#     related_head_triples = set()
#     related_tail_triples = set()
    
#     for name in head_names:
#         for triple in triples:
#             if name in triple:
#                 related_head_triples.add(triple)
#     for name in tail_names:
#         for triple in triples:
#             if name in triple:
#                 related_tail_triples.add(triple)
   
#     # 求交集
#     golden_triples = related_head_triples & related_tail_triples
    
#     if len(golden_triples) > 0:
#         return list(golden_triples)
    
#     # 如果存在某一种三元组没有找到，就只能通过文本来进行匹配
#     if len(related_head_triples) == 0 or len(related_tail_triples) == 0:
#         return []
    
    
#     # 返回全部三元组
#     return_triples = []
#     for triple in list(related_head_triples):
# #         if len(return_triples) == 2:
# #             break
#         return_triples.append(triple)
        
#     for triple in list(related_tail_triples):
# #         if len(return_triples) == 4:
# #             break
#         return_triples.append(triple)
#     return return_triples


# def return_topK(file_path = './dataNEW/res_0924_topK/topK.json'):
#     fr = open(file_path, 'r', encoding='utf-8')
#     json_info = fr.read()
#     df_topk = pd.read_json(json_info)
    
#     offline_top5 = []
#     i = 0 
#     j = 0
#     offline_top5_r = {}
#     while True:
#         if i == len(df_topk['doc_idxs']):
#             break
#         if df_topk['doc_idxs'][i] != j:
#             j += 1
#             offline_top5.append(offline_top5_r)
#             offline_top5_r = {}
#             continue
            
#         predict_labels = [key for key in df_topk['logprobs_r'][i].keys()]
#         predict_scores = [df_topk['logprobs_r'][i][key] for key in df_topk['logprobs_r'][i].keys()]
#         top5_rs = {}

#         for _ in range(5):
#             if len(predict_scores) == 0 :
#                 break
#             max_index = predict_scores.index(max(predict_scores))
#             max_relation = predict_labels[max_index]
#             top5_rs[max_relation] = max(predict_scores)
#             predict_scores.remove(max(predict_scores))
#             predict_labels.remove(max_relation)
#         offline_top5_r[(df_topk['h_idx'][i], df_topk['t_idx'][i])] = top5_rs
#         i += 1

#     return offline_top5

# def find_answer(related_triples, top5_relations, head_name, tail_name, head_type, tail_type, p2name={}, p2templates ={}, ht_pair="", verbose = False):
#     final_answer = [{
#         'h_idx': ht_pair[0],
#         't_idx': ht_pair[1],
#         'r': "NA"
#     }]
#     choices = []
#     relations = []
#     relation_scores = []
#     prompt = "**According to the tuples as follows:**\n" + related_triples + "\n\n"
#     prompt += "**True or False:**\n" 
#     i = 0
#     for relation in top5_relations.keys():
#         if relation != "NA":
# #             if relation.split(" ")[-1] == "of" or relation.split(" ")[-1] == "by":
# #                 now_choice = f"{chr(ord('A') + i)}. {head_name}({head_type})'s {relation} {tail_name}({tail_type})."
# #             else:
# #                 now_choice = f"{chr(ord('A') + i)}. {head_name}({head_type})'s {relation} is {tail_name}({tail_type})."
#             # 读取模板
#             now_choice = f"{chr(ord('A') + i)}. " + p2templates[relation]
#             now_choice = now_choice.replace('<head>',f"{head_name}({head_type})" )
#             now_choice = now_choice.replace('<tail>',f"{tail_name}({tail_type})")

#             choices.append(now_choice)
#             relations.append(relation)
#             relation_scores.append(top5_relations[relation])
            
#             i += 1
            
#     now_choice = f"{chr(ord('A') + i)}. None of the above options is correct."
#     choices.append(now_choice)
    
#     # 2种可能关系的选择题
#     prompt += "\n".join(choices)
#     # COT 思维链
#     prompt += "\n\nLet's think step by step:\n"
#     prompt += "1. Summary the provided tuples' information.\n"
#     # 需要注意到头尾实体之间的类型是否符合逻辑
#     prompt += "2. Carefully evaluate each statement, considering whether this relation possible between the entity type of the subject and object.\n"
#     prompt += "3. Return all True letters.\n"
#     prompt += "4. Return a best True letter.\n"
#     # 输出格式
#     prompt += "\nUse the following format:\nSummary: <tuples' information>\nEvaluate: <each statement>\nAll True letters: <the best statement>\nBest letter: <a letter>"
    
#     # 得到结果
#     predict = get_completion(prompt)
#     # 输出中间状态
#     if verbose:
#         print(ht_pair)
#         print(prompt)
#         print("-"*100)
#         print(predict)
#         print("="*100)
        
#     # 处理输出结果
#     predict = predict.strip().split("\n")
#     # 找到所有的输出结果
#     result = "None"
#     for sentence in predict:
#         # 找到多选关系的那一行
#         if "All True letters:" in sentence :
#             result = sentence
#             break
    
#     # 说明不存在关系
#     if "None" in result:
#         print(f"{ht_pair}-({head_name}, {tail_name})-> NA")
#         return None
#     else:
#         # 得到所有的选项的下标
#         all_letters = [letter.strip() for letter in result[len("All True letters:"):].replace("and",",").split(",")]
#         for letter in all_letters:
#             if len(letter) > 1:
#                 if ord(letter[0]) >=  ord('A') and ord(letter[0]) <=  (ord('A') + i - 1):
#                     all_letters.append(letter[0])
#                 all_letters.remove(letter)
#                 continue
#             if len(letter) == 0:
#                 continue
#             if ord(letter) <  ord('A') or ord(letter) >  (ord('A') + i - 1):
#                 all_letters.remove(letter)
            
#         all_indexs = [ord(letter) - ord('A')  for letter in all_letters if len(letter) != 0]
        
#         choosen_relations = []
#         choosen_scores = []
#         # 查看一下所有选项
#         for i in all_indexs:
#             choosen_relations.append(relations[i])
#             choosen_scores.append(relation_scores[i])
            
#         # 只选择一个关系
#         if len(choosen_relations) == 1 :
#             print(f"{ht_pair}-({head_name}, {tail_name})-> {choosen_relations[0]} {p2name[choosen_relations[0]]}")
#             final_answer[0]['r'] = choosen_relations[0]
#             return final_answer
#             # if choosen_scores[0] > top5_relations["NA"]:
#             #     print(f"{ht_pair}-({head_name}, {tail_name})-> {choosen_relations[0]} {p2name[choosen_relations[0]]}")
#             #     final_answer[0]['r'] = choosen_relations[0]
#             #     return final_answer
#             # else:
#             #     print(f"{ht_pair}-({head_name}, {tail_name})-> NA")
#             #     return None
#         # 否则返回多个关系
#         elif len(choosen_relations) > 1:
#             i = 0
#             for relation in choosen_relations:
#                 if "NA" not in top5_relations or top5_relations[relation] > top5_relations["NA"]:
#                     print(f"{ht_pair}-({head_name}, {tail_name})-> {relation} {p2name[relation]}")
#                     if len(final_answer) == i:
#                         final_answer.append(final_answer[0])
#                     final_answer[i]['r'] = relation
#                     i += 1
#             return final_answer
#     print(f"{ht_pair}-({head_name}, {tail_name})-> NA")
#     return None





