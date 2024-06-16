from c2net.context import prepare
import warnings
from docre.processData import return_rel2dict,return_templates,return_docred
from docre.processLogits import return_doc_logits_2024
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import matplotlib.pyplot as plt
from collections import Counter
import sklearn.model_selection
from itertools import permutations 
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
random.seed(527)
warnings.filterwarnings("ignore") 


c2net_context = prepare()

#获取数据集路径
dataset_path = c2net_context.dataset_path+"/"+"dataset"
rel_templates_path = c2net_context.dataset_path+"/"+"rel_templates"
docred_logits_path = c2net_context.dataset_path+"/"+"docred-logits"

#获取预训练模型路径
meta_llama_3_8b_path = c2net_context.pretrain_model_path+"/"+"Meta-Llama-3-8B"
meta_llama_3_8b_instruct_path = c2net_context.pretrain_model_path+"/"+"Meta-Llama-3-8B-Instruct"

#输出结果必须保存在该目录
you_should_save_here = c2net_context.output_path


# In[4]:


# 关系和关系训练编号，以及关系和关系名称的字典
p_to_num, num_to_p, p_to_name, name_to_p = return_rel2dict(dataset_path + "/docred/rel_info.json")
print("P159 CLSNUM: ", p_to_num['P159'],"\nP159 NAME: ",p_to_name['P159'])

# 读取数据集
titles, entities, entity_types, entity_indexs, documents_raw, relations = return_docred(dataset_path + "/docred/dev.json")

# 读取关系模板 dict 
p2templates = return_templates(rel_templates_path + "/rel_templates.xlsx")
print(p2templates['P159'])

# 读取 DREEAM 的预测结果
atlop_relations = return_doc_logits_2024(test_data_path = dataset_path + "/docred/dev.json",
                                    rel2id_path = dataset_path + "/meta/rel2id.json",
                                    logits_path = docred_logits_path + "/atlop/dev/dev_logits.json")


# In[101]:


import torch
import torch.nn.functional as F

inputs = [] # 记录输入的prompt
completions = [] # 记录正确答案
statments = [] # 记录所有选项
TOP_K = 4
for i in range(len(documents_raw)):
    entity_pairs = {}
    for entity_a in range(len(entities[i])):
        for entity_b in range(len(entities[i])):
            if entity_a != entity_b:
                entity_pairs[(entity_a,entity_b)] = ['Na']
    
    for relation in relations[i]:
        if 'Na' in entity_pairs[(relation['h'],relation['t'])]:
            entity_pairs[(relation['h'],relation['t'])] = [relation['r']]
        else:
            entity_pairs[(relation['h'],relation['t'])].append(relation['r'])
    
    questions = []
    prompts = []
    answers = []
    
    # 构建所有的关系模板
    for pair in entity_pairs.keys():
        question = []
        answer = []
        # atlop 的 97 种关系的预测结果
        logits = atlop_relations[i][pair]
        # 选取 top_k 的关系作为关系模板构建
        
        # 选择前 TOP_K 个键
        keys = list(logits.keys())[:TOP_K]
        # 提取对应的 logits 值
        logits_values = torch.tensor([logits[key] for key in keys])
        # 计算 Softmax
        softmax_values = F.softmax(logits_values, dim=0)
        # 将 Softmax 结果映射回原来的键
        softmax_logits = {keys[i]: softmax_values[i].item() for i in range(len(keys))}
        # 只处理 Na 最大的
        if list(logits.keys())[0] == 'Na' and softmax_values[1] >= softmax_values[0] * 0.3:
            print(f"doc {i}", [(key, softmax_logits[key]) for key in softmax_logits.keys()], " >>>", entity_pairs[pair])
            j = 0
            for logit in list(logits.keys())[:TOP_K]:
                if logit in entity_pairs[pair] and logit != 'Na':
                    answer.append(f"{chr(ord('A') + j)}")
                if logit == 'Na':
                    continue
                head_name = entities[i][pair[0]][random.randint(0, len(entities[i][pair[0]]) - 1)] # 随机选一个别名,增加鲁棒性
                tail_name = entities[i][pair[1]][random.randint(0, len(entities[i][pair[1]]) - 1)] # 随机选一个别名,增加鲁棒性
                head_type = entity_types[i][pair[0]]
                tail_type = entity_types[i][pair[1]]
                now_question = f"{chr(ord('A') + j)}. " + p2templates[logit]
                now_question = now_question.replace('<head>',f"{head_name}({head_type})" )
                now_question = now_question.replace('<tail>',f"{tail_name}({tail_type})" )
                return_dict = {
                    'title':titles[i],
                    'h':pair[0],
                    't':pair[1],
                    'r':logit,
                    'score': softmax_logits[logit]
                }
                question.append((return_dict, now_question))
                j += 1
            return_dict = {
                    'title':titles[i],
                    'h':pair[0],
                    't':pair[1],
                    'r':'Na',
                    'score': softmax_logits['Na']
                }
            question.append((return_dict, f"{chr(ord('A') + j)}. None of the above options is correct."))     
            questions.append(question)
            if len(answer) == 0:
                answers.append(f"{chr(ord('A') + j)}")
            else:
                answers.append(answer)
    # 构建所有的 prompt
    for k in range(len(questions)):
        now_question = ""
        for question in questions[k]:
            now_question += question[1] + '\n'
        prompt = f"""##INSTRUCTION: Read the ##DOCUMENT and answer the ##QUESTION. Write the answers in ##ANSWER.
        
##DOCUMENT: {" ".join(documents_raw[i])}

##QUESTION: Which of the following is right?
{now_question}
##ANSWER: """
        prompts.append(prompt)
     
    inputs.append(prompts)
    completions.append(answers)
    statments.append(questions)



# In[7]:


model_id = meta_llama_3_8b_instruct_path

 
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
#     quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2= False,
    device_map="auto", 
)

model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id


# In[8]:


from tqdm import tqdm


# In[103]:


import pandas as pd
slm_res = pd.read_json('./refine_atlop_dev/dev_results.json')

final_titles = []
final_h_idxs = []
final_t_idxs = []
final_rs = [] 

# 保存小模型的所有结果
for i in range(len(slm_res['title'])):
    final_titles.append(slm_res['title'][i])
    final_h_idxs.append(slm_res['h_idx'][i])
    final_t_idxs.append(slm_res['t_idx'][i])
    final_rs.append(slm_res['r'][i])


# In[104]:


correct = 0
mis_correct = 0
for prompts, answers, questions in tqdm(zip(inputs,completions,statments), total=len(inputs), desc = f"Refine docred-dev document with ATLOP in llama3-8B-instruct..."):

    for prompt, answer, question in zip(prompts, answers, questions):
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=True, top_p=0.9, temperature=1.8, output_scores=True, return_dict_in_generate=True)
        generated_token_id = output.sequences[:, -1].item() # 获取生成的token id
        scores = output.scores[0] # 获取得分分布
        probabilities = torch.nn.functional.softmax(scores, dim=-1) # 计算softmax以获得每个token的概率
        all_probabilities = probabilities[0].tolist()
        # 将 LLM 和 SLM 的分数进行累加
        for i, prob in enumerate(all_probabilities):
            if prob > 0:
                predict = tokenizer.decode([i]).split()
                if len(predict) > 0 and predict[0] in ['A','B','C','D']:
                    question[ord(predict[0]) - ord('A')][0]['score'] += prob # 分数加权
        all_relations = [choice[0] for choice in question]
        threshold = all_relations[-1]['score']
        for i,now_relation in enumerate(all_relations):
            if now_relation['score'] > threshold and now_relation['r'] != 'Na':
                final_titles.append(now_relation['title'])
                final_h_idxs.append(now_relation['h'])
                final_t_idxs.append(now_relation['t'])
                final_rs.append(now_relation['r'])
                if chr(ord('A') + i) in answer:
                    correct += 1
                else:
                    mis_correct += 1
#         max_relation = max(all_relations, key=lambda x: x['score'])       
#         if max_relation['r'] != 'Na':
#             final_titles.append(max_relation['title'])
#             final_h_idxs.append(max_relation['h'])
#             final_t_idxs.append(max_relation['t'])
#             final_rs.append(max_relation['r']):
print(f"correct: {correct}  mis_correct: {mis_correct}")


# In[105]:


df_result = pd.DataFrame(zip(final_titles, final_h_idxs, final_t_idxs, final_rs), columns = ['title','h_idx', 't_idx', 'r'])
df_result.to_json("./data_analyze/refine_results.json", orient='records')


# ### baseline

# In[7]:


from docre.evaluation import evaluate

evaluate(data_path = dataset_path + "/docred",  # 测试的数据集
         test_data = "dev.json",          # dev 测试
         result_data = "./refine_atlop_dev/dev_results.json", # 模型的预测结果
         output_path = "./refine_atlop_dev"
        )


# ### refine result

# In[16]:


from docre.evaluation import evaluate

evaluate(data_path = dataset_path + "/docred",  
         test_data = "dev.json",         
         result_data = "./data_analyze/refine_results.json", 
         output_path = "./data_analyze"
        )