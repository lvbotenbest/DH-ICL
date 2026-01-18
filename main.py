from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import torch
import os
import json
from tqdm import tqdm
import random
import argparse
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments.
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="PreSelect: Self-aware language model for question answering")
    
    # Model configuration
    parser.add_argument('--model_dir', 
                    default="./pretrain-model/Llama-2-7b-chat-hf", 
                    type=str,
                    help='Path to the pre-trained model directory (e.g., ./pretrain-model/Llama-2-7b-chat-hf)')
    
    # File paths
    parser.add_argument('--test_file', 
                    default="./model_outputs.json", 
                    type=str,
                    help='Path to save the evaluation results and outputs (e.g., ./model_outputs.json)')
    parser.add_argument('--data_file', 
                    type=str, 
                    required=True,
                    help='Path to the input dataset file for evaluation (required)')
    parser.add_argument('--demon_file', 
                    type=str, 
                    required=True,
                    help='Path to the demonstration examples file for in-context learning (required)')
    parser.add_argument('--time_demon_file', 
                    type=str, 
                    default="./input/time_aware_demon3.json",
                    help='Path to time-aware demonstration examples file')
    
    # Evaluation configuration
    parser.add_argument('--dynamic', 
                    action='store_true', 
                    default=True,
                    help='Enable dynamic instruction generation using in-context examples')
    parser.add_argument('--self_aware', 
                    action='store_true', 
                    default=True,
                    help='Enable self-aware evaluation mode')
    parser.add_argument('--only_instruction', 
                    action='store_true', 
                    default=False,
                    help='Use instruction-only mode without demonstrations')
    
    # Dataset selection
    parser.add_argument('--dataset', 
                    type=str, 
                    choices=['triviaqa', 'taqa', 'wq', 'freshqa'], 
                    default='triviaqa',
                    help='Dataset to use for evaluation: triviaqa, taqa, wq, or freshqa')
    
    args = parser.parse_args()  
    return args


def read_json(file):
    """Read and parse JSON file.
    
    Args:
        file (str): Path to the JSON file
        
    Returns:
        dict/list: Parsed JSON data
    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def find_special_token_positions(input_ids, special_token_ids):
    """Find positions of special tokens in input sequence.
    
    Args:
        input_ids: Input token IDs tensor
        special_token_ids: Special token IDs tensor to search for
        
    Returns:
        list: List of starting positions where special tokens are found
    """
    input_ids = input_ids.squeeze().tolist()
    special_token_ids = special_token_ids.squeeze().tolist()
    
    positions = []
    special_len = len(special_token_ids)
    
    # Slide window to find special token sequences
    for i in range(len(input_ids) - special_len + 1):
        if input_ids[i:i + special_len] == special_token_ids:
            positions.append(i)

    return positions


def find_discriminate_logits(logits, special_token_ids, input_ids):
    pos_start = find_special_token_positions(input_ids, special_token_ids)[:-1]
    
    label_list = []
    input_ids = input_ids.squeeze()
    bias = []
    true_bias = 0
    false_bias = 0

    num_false = 0
    num_true = 0
    num_bias = 0
    true_logits_sum = 0
    false_logits_sum = 0

    all_true = 0
    all_false = 0

    special_token_length = special_token_ids.shape[-1]
    pos_start = pos_start[1:]
    
    for pos in pos_start:
        pos = pos + special_token_length
        label = input_ids[pos].item()
        label_list.append(label)
        label_score = logits[pos-1, label].item()
        
        if label == 3009:
            true_logits_sum += label_score
            all_true += 1
            false_ = 4541
        else:
            false_ = 3009
            false_logits_sum += label_score
            all_false += 1

        bias_ = label_score - logits[pos-1, false_].item()

        if bias_ < 0:
            num_bias += 1
            if label == 3009:
                true_bias -= bias_
                num_true += 1
            else:
                false_bias -= bias_
                num_false += 1

    if num_bias > 0:
        all_true = max(all_true, 1)
        all_false = max(all_false, 1)
        return -true_bias/all_true, -false_bias/all_false
    else:
        return 0, 0



def generate_inference(input_text, model, tokenizer, max_token):
    messages = [{"role": "user", "content": input_text}]
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        input_ids=input_ids,
        do_sample=True,
        temperature=0.6, 
        top_p=0.9, 
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_token,
    )
    outputs = model.generate(**generate_kwargs)
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)





def infer_true_false(input_text, model, tokenizer, max_token, add_bias):
    if not add_bias:
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        generate_kwargs = dict(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_token,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        
        outputs = model.generate(**generate_kwargs)

        if outputs["logits"][0][0, 3009].item() > outputs["logits"][0][0, 4541].item():
            return "true"
        else:
            return "false"
    else:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad(): 
            outputs = model(input_ids.to(model.device), return_dict=True)
            end_token = torch.tensor([[13, 22550, 29901]])
    
            priori_true, priori_false = find_discriminate_logits(outputs["logits"][0], end_token, input_ids)

            true_logits = outputs["logits"][0][-1, 3009].item() - priori_true
            false_logits = outputs["logits"][0][-1, 4541].item() - priori_false

        if true_logits > false_logits:
            return "true"
        else:
            return "false"






def infer_true_false_time_aware(input_text, model, tokenizer, max_token, add_bias):
    if not add_bias:
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        generate_kwargs = dict(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_token,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        
        outputs = model.generate(**generate_kwargs)

        if outputs["logits"][0][0, 3009].item() > outputs["logits"][0][0, 4541].item():
            return "false"
        else:
            return "true"
    else:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad(): 
            outputs = model(input_ids.to(model.device), return_dict=True)
            end_token = torch.tensor([[13, 22550, 29901]])

            priori_true, priori_false = find_discriminate_logits(outputs["logits"][0], end_token, input_ids)

            true_logits = outputs["logits"][0][-1, 3009].item() - priori_true
            false_logits = outputs["logits"][0][-1, 4541].item() - priori_false

        if true_logits > false_logits:
            return "false"
        else:
            return "true"



  



def infer_true_false_cad(wo_context, input_with_context, model, tokenizer):
    wo_context_input_ids = tokenizer.encode(wo_context, return_tensors="pt") 
    input_ids = tokenizer.encode(input_with_context, return_tensors="pt")

    with torch.no_grad(): 
        wo_context_outputs = model(wo_context_input_ids.to(model.device), return_dict=True)
        outputs = model(input_ids.to(model.device), return_dict=True)

        end_token = torch.tensor([[13, 22550, 29901]])

        wo_context_true_logits = wo_context_outputs["logits"][0][-1, 3009].item()
        wo_context_false_logits = wo_context_outputs["logits"][0][-1, 4541].item()

        with_context_true_logits = outputs["logits"][0][-1, 3009].item()
        with_context_false_logits = outputs["logits"][0][-1, 4541].item()

        true_logits = with_context_true_logits - wo_context_true_logits
        false_logits = with_context_false_logits - wo_context_false_logits

    if true_logits > false_logits:
        return "true"
    else:
        return "false"






def generate_demonstration(output_file, sample_num, input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    demonstrate_list = []
    right_num = 0
    wrong_num = 0

    for item in data:
        if random.randint(1, 100) == 34:
            if item["not_ret_answer_result"] == "true":
                if right_num < sample_num:
                    demonstrate_list.append(item)
                    right_num += 1
            else:
                if wrong_num < sample_num:
                    demonstrate_list.append(item)
                    wrong_num += 1
    
    random.shuffle(demonstrate_list)

    with open(output_file, "w") as file:
        json.dump(demonstrate_list, file, ensure_ascii=False, indent=4)







def construct_instruction(demonstrate_file, time_demon=None):
    instruction = "[INST] You are a student being tested.  For each question provided, first go through a thinking phase (no need to output specific content). Then, assess whether you can answer it correctly based on your knowledge.\
 If you believe you can answer it correctly, output 'true'. If you feel your answer might be incorrect, output 'false'. Additionally, if the question is asking about a recent event, for example, if words like recently, latest, or currently appear, also output 'false'. \
\nHere are some examples and the output format.[/INST]\n\n"
   
    with open(demonstrate_file, 'r') as f:
        demonstrate_data = json.load(f)

    if time_demon:
        demonstrate_data = demonstrate_data + time_demon

    demonstrate_instruct = ""
    num = 0
    think_phase = "<0x00>"
    add_think = False
    
    for line in demonstrate_data:
        question = line["question"]
        
        if "wo_ret_answer_result" in line:
            key = "wo_ret_answer_result"
        else:
            key = "not_ret_answer_result"
        answer = line[key]
        
        if add_think:
            one_demon_data = f"Question: {question}\nThink: {think_phase}\nAnswer:{answer}\n"
        else:
            one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonstrate_instruct += one_demon_data

    return instruction + demonstrate_instruct



def construct_instruction_time_aware(demonstrate_file):
    instruction = "[INST]It is now the year 2024. Please determine whether the following question is asking about a recent event. For example, if words like 'recently,' 'latest,' or 'currently' appear, they indicate a recent event. Any other questions unrelated to time are also not considered as asking about a recent event.\
 If the question is asking about a recent event, output 'false'; if the question is not asking about a recent event, output 'true'.\
 \nHere are some examples and the output format.[/INST]\n\n"
   
    with open(demonstrate_file, 'r') as f:
        demonstrate_data = json.load(f)

    demonstrate_instruct = ""
   
    num_true = 0
    num_false = 0
    
    for line in demonstrate_data:
        question = line["question"]
        answer = line["not_ret_answer_result"]
        
        if answer == "false":
            num_true += 1
        else:
            num_false += 1
            
        one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonstrate_instruct += one_demon_data

    return instruction + demonstrate_instruct


def construct_dynamic_time_aware_instruction(demon_list):
    instruction = "[INST]Please determine whether the following question is asking about a recent event. For example, if words like 'recently,' 'latest,' or 'currently' appear, they indicate a recent event. However, certain other words, such as specific years like '1990,' 'first time', do not indicate a question about a recent event.\
If the question is asking about a recent event, output 'true'; if the question is not asking about a recent event, output 'false'.[/INST]\n\n"           
    demonstrate_instruct = ""
   
    num_true = 0
    num_false = 0
    
    for line in reversed(demon_list):
        question = line["question"]
        answer = "false" if line["not_ret_answer_result"] == "true" else "true"

        if answer == "false":
            num_true += 1
        else:
            num_false += 1
            
        one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonstrate_instruct += one_demon_data

    return instruction + demonstrate_instruct, num_true, num_false



def construct_dynamic_instruction(demon_list):
    instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
 If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly, output 'false'. Additionally, if the question is asking about a recent event, for example, if words like recently, latest, or currently appear, also output 'false'.\
\nHere are some examples and the output format.[/INST]\n\n"

    demonstrate_instruct = ""
   
    num_true = 0
    num_false = 0
    
    for line in reversed(demon_list):
        question = line["question"]
        answer = line["not_ret_answer_result"]

        if answer == "true":
            num_true += 1
        else:
            num_false += 1
            
        one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonstrate_instruct += one_demon_data

    return instruction + demonstrate_instruct, num_true, num_false




def test_instruction(icl, new_question):
    add_think = False
    
    if add_think:
        think_phase = "<0x00>"
        test_question = f"Question: {new_question}\nThink: {think_phase}\nAnswer:"
    else:
        test_question = f"Question: {new_question}\nAnswer:"

    return icl + test_question


def get_instruction_only():
    instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
 If you believe you can answer it correctly, Answer:true. If you are unsure whether you can answer it correctly, Answer:false.[/INST]\n\n"

    return instruction



def self_judge(icl, data, model, tokenizer):
    pred_true = 0 
    pred_false = 0
    real_true = 0
    real_false = 0
    true_acc = 0
    false_acc = 0
    
    for item in tqdm(data):
        input_ = test_instruction(icl, item["question"])
        out = infer_true_false(input_, model, tokenizer, 1, False)
        out = out.replace("\n", " ")

        if "true" in out:
            pred_true += 1
        elif "false" in out:
            pred_false += 1
        else:
            logger.error("WRONG OUTPUT FORMAT")

        if "true" in item["not_ret_answer_result"]:
            real_true += 1
            if "true" in out:
                true_acc += 1
        elif "false" in item["not_ret_answer_result"]:
            real_false += 1
            if "false" in out:
                false_acc += 1
        
    logger.info(f"real_true: {real_true}    real_false: {real_false}    pred_true: {pred_true}    pred_false: {pred_false}")
    logger.info(f"false_acc: {false_acc}    true_acc: {true_acc}")
    logger.info(f"all right: {false_acc + true_acc}    acc: {(true_acc + false_acc)/(real_true + real_false)}")




    



def evaluate_model(label_data, model, tokenizer, args, time_demon):
    """Evaluate model performance on given data."""
    if not args.dynamic and not args.only_instruction:
        icl = construct_instruction(args.demon_file, time_demon)

    logger.info("Starting evaluation with 20 demonstrations")
    
    pred_true = 0 
    pred_false = 0
    real_true = 0
    real_false = 0
    true_acc = 0
    false_acc = 0
    response_acc = 0
    not_ret_acc = 0
    ret_acc = 0

    for item in tqdm(label_data):
        self_label = item

        if args.self_aware:
            if args.dynamic:
                icl, true_num, false_num = construct_dynamic_instruction(self_label["demon_data"][:20])
                
            elif args.only_instruction:
                icl = get_instruction_only()

            input_ = test_instruction(icl, self_label["question"])
            self_aware_out = infer_true_false(input_, model, tokenizer, 1, True)
            
        out = self_aware_out

        if "true" in out:
            pred_true += 1
            if item["not_ret_result"] == "true":
                response_acc += 1
        elif "false" in out:
            pred_false += 1
            if item["ret_result"] == "true":
                response_acc += 1
        else:
            logger.error("WRONG OUTPUT FORMAT")

        if "true" in item["not_ret_result"]:
            real_true += 1
            if "true" in out:
                true_acc += 1
        elif "false" in item["not_ret_result"]:
            real_false += 1
            if "false" in out:
                false_acc += 1
        else:
            raise KeyError
        
        if "true" in item["not_ret_result"]:
            not_ret_acc += 1
        
        if "true" in item["ret_result"]:
            ret_acc += 1

    logger.info(f"real_true: {real_true}    real_false: {real_false}    pred_true: {pred_true}    pred_false: {pred_false}")
    logger.info(f"false_acc: {false_acc}    false_acc %: {false_acc/pred_false if pred_false > 0 else 0}    true_acc: {true_acc}")
    logger.info(f"all right: {false_acc + true_acc}    acc: {(true_acc + false_acc)/(real_true + real_false)}")
    logger.info(f"result_em_acc: {response_acc/(real_true + real_false)}")
    logger.info(f"not_ret_acc: {not_ret_acc/(real_true + real_false)}")
    logger.info(f"ret_acc: {ret_acc/(real_true + real_false)}")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_dir
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Dynamic: {args.dynamic}")
    logger.info(f"Self-aware: {args.self_aware}")
    logger.info(f"Only instruction: {args.only_instruction}")

    model_name = args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    time_demon = read_json(args.time_demon_file)

    if args.dataset == 'triviaqa':
        label_data = read_json(args.data_file)
    
    elif args.dataset == 'wq':
        label_data = read_json(args.data_file)
           
    elif args.dataset == 'taqa':
        label_data = read_json(args.data_file)

    elif args.dataset == 'freshqa':
        label_data = read_json(args.data_file)
        logger.info("Load freshqa data")
        time_label_data = read_json(args.data_file.replace("self-aware", "freshqa_time"))

    evaluate_model(label_data, model, tokenizer, args, time_demon)