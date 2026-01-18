import json
import random
from tqdm import tqdm

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)

    return data


def write_json(file,write_list):

    with open(file, 'w') as file:
        json.dump(write_list, file,ensure_ascii=False, indent=4)


def conbine_corpus(lis1,lis2):

    all_lis = []
    question_list = []
    repeat_num = 0

    for sample in lis1:
        all_lis.append(sample)
        question_list.append(sample["question"])

    for sample in tqdm(lis2):
        if sample["question"] in question_list:
            repeat_num+=1
            continue    
        else:
            all_lis.append(sample)
    print(repeat_num)

    return all_lis


intent_aware = read_json("./input/intent_aware_cls_ce_valid.json")
time_aware = read_json("./input/time_aware_cls_ce_valid.json")
knowledge_aware = read_json("./input/knowledge_aware_cls_ce_valid.json")
self_aware = read_json("./input/self_aware_cls_ce_llama2_7b_chat_train.json")


all_data = conbine_corpus(time_aware,self_aware)


write_json("./corpus/corpus.json",all_data)

