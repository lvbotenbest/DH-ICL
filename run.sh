#!/bin/bash

# Simple PreSelect Evaluation Runner

echo "PreSelect Evaluation Runner"
echo "========================="

# Create necessary directories
mkdir -p data outputs input

echo "Running evaluation examples..."
echo ""

# Example 1: TriviaQA
echo "1. Running TriviaQA evaluation..."
python main.py \
    --model_dir ./pretrain-model/Llama-2-7b-chat-hf \
    --data_file ./data/triviaqa_data.json \
    --demon_file ./data/demonstrations.json \
    --dataset triviaqa \
    --test_file ./outputs/triviaqa_results.json

echo ""

# Example 2: FreshQA
echo "2. Running FreshQA evaluation..."
python main.py \
    --model_dir ./pretrain-model/Llama-2-7b-chat-hf \
    --data_file ./data/freshqa_data.json \
    --demon_file ./data/demonstrations.json \
    --dataset freshqa \
    --test_file ./outputs/freshqa_results.json

echo ""

# Example 3: WebQuestions (instruction only)
echo "3. Running WebQuestions evaluation (instruction only)..."
python main.py \
    --model_dir ./pretrain-model/Llama-2-7b-chat-hf \
    --data_file ./data/wq_data.json \
    --demon_file ./data/demonstrations.json \
    --dataset wq \
    --only_instruction \
    --test_file ./outputs/wq_results.json

echo ""
echo "All evaluations completed!"
echo ""
echo "Usage examples:"
echo "python main.py --model_dir <path> --data_file <path> --demon_file <path> --dataset <choice> --test_file <path>"
echo ""
echo "Datasets: triviaqa, taqa, wq, freshqa"
echo "Options: --dynamic, --self_aware, --only_instruction"
