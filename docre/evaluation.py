#!/usr/bin/env python
import sys
import os
import os.path
import json

def gen_train_facts(data_file_name, truth_dir):
    # 获取文件名 
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    # 拼接成真实路径
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))
    
    # 如果当前文件存在，读取其中的所有元组
    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train
    
    # 否则就根据数据集来构建
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    # 每个data 对应一个 document
    for data in ori_data:
        # 获取当前文档的所有头尾实体对
        vertexSet = data['vertexSet']
        # 获得所有的关系
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    # 所有的同义词都构建这种关系
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train

def evaluate(data_path = "./docred", 
             test_data = "dev.json", 
             result_data="./result.json", 
             output_path="./", 
             train_annotated_path = "/train_annotated.json", 
             compare_distant = True):
    input_dir = data_path
    truth_dir = os.path.join(input_dir, 'ref')
    
    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    if os.path.isdir(truth_dir):
        # 读取所有三元组 fact
        fact_in_train_annotated = gen_train_facts(data_path + train_annotated_path, truth_dir)
        if compare_distant:
            fact_in_train_distant = gen_train_facts(data_path + "/train_distant.json", truth_dir)
        else:
            fact_in_train_distant = set([])

        # 输出文件名为 scores.txt
        output_filename = os.path.join(output_path, 'socres.txt')
        # 打开写入
        output_file = open(output_filename, 'w')

        # dev.json 载入 dev
        truth_file = os.path.join(data_path, test_data)
        truth = json.load(open(truth_file))

        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        # 遍历每一篇文档
        for x in truth:
            # 文档标题
            title = x['title']
            # print(title)
            titleset.add(title)

            # 获取所有实体
            vertexSet = x['vertexSet']
            # 把标题和实体集绑定
            title2vectexSet[title] = vertexSet

            # 读取所有的标注关系
            for label in x['labels']:
                r = label['r']

                h_idx = label['h']
                t_idx = label['t']
                # 把一个标准输出和 证据句子的 list 转 set 绑定在一起
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                # 统计证据长度
                tot_evidences += len(label['evidence'])

        # 总共有的关系数量
        tot_relations = len(std)

        # 提交 result.json 文件
#         submission_answer_file = os.path.join(result_path, "result.json")
        submission_answer_file = result_data
        tmp = json.load(open(submission_answer_file))
        # result.json 里面的 dict 按照 title, h_idx, t_idx, r 进行排序
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        # 将第一个 dict 先放入列表
        submission_answer = [tmp[0]]
        # 遍历后面的所有元素
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i-1]
            # 和前面不重复就存入
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])
    #         else:
    #             print("remove", x['title'], x['h_idx'], x['t_idx'], x['r'])

        correct_re = 0
        correct_evidence = 0 # 正确的证据
        pred_evi = 0 # 每个关系 的['evidence'] 中的所有数量

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])

        # 遍历所有结果
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            # 读取 如果不在 title -> 实体集就跳过
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            # 否则读取当前文档名称的所有实体集
            vertexSet = title2vectexSet[title]

            # 是否有预测证据句子list
            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            # 预测句子长度增加
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                # 预测对的关系 + 1
                correct_re += 1
                # 标准的证据句子 set
                stdevi = std[(title, r, h_idx, t_idx)]
                # 重合的数量
                correct_evidence += len(stdevi & evi)
                # 
                in_train_annotated = in_train_distant = False
                # 当前的三元组是否在 训练集出现过
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                # 在训练集正确的数量 +1
                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

        # 正确的关系三元组 除以整体的三元组的数量 -> precision
        re_p = 1.0 * correct_re / len(submission_answer)

        # 正确三元组占标准答案的数据的比例 -> recall 
        re_r = 1.0 * correct_re / tot_relations
        if re_p+re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        # 证据句子的 precision 和 recall
        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi>0 else 0
        evi_r = 1.0 * correct_evidence / (tot_evidences+0.000000000000000001)
        if evi_p+evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        # 忽略训练集中已经出现的三元组得到的 precision
        re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
        re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) / (len(submission_answer)-correct_in_train_distant)

        # f1 计算
        if re_p_ignore_train_annotated+re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train+re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)


        print("Precision:",re_p)
        print("Recall:",re_r)
        print ('RE_F1:', re_f1)
        print ('Evi_F1:', evi_f1)
        print ('RE_ign_F1:', re_f1_ignore_train_annotated)
        print ('RE_ignore_distant_F1:', re_f1_ignore_train)
        
        print("内容已经保存到:", output_filename)

        output_file.write("Precision: %f\n" % re_p)
        output_file.write("Recall: %f\n" % re_r)

        output_file.write("RE_F1: %f\n" % re_f1)
        output_file.write("Evi_F1: %f\n" % evi_f1)

        output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
        output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)


        output_file.close()