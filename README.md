# DH-ICL: Dynamic Hypertext In-Context Learning

This repository implements the DH-ICL method from the paper [DH-ICL.pdf](docs/DH-ICL.pdf).

## Overview

DH-ICL is a dynamic in-context learning approach that combines retrieval-based corpus selection with adaptive demonstration generation for improved question answering performance.

## Dataset

This project uses the corpus dataset from the [UAR](https://github.com/xiami2019/UAR) repository.


## Usage

### Step 1: Prepare Corpus
```bash
python get_corpus.py
```
This script processes and combines different awareness datasets to create the training corpus.

### Step 2: Run Retrieval and Generation
```bash
bash rag.sh
```
This script performs retrieval-based demonstration selection and generates the final dataset.

### Step 3: Run Evaluation
```bash
bash run.sh
```
This script runs the main evaluation using the prepared dataset.

## File Structure

- `docs/` - Documentation and papers
- `get_corpus.py` - Corpus preparation and combination
- `retrieve_icl.py` - Retrieval-based in-context learning implementation
- `main.py` - Main evaluation script
- `rag.sh` - Retrieval and generation pipeline
- `run.sh` - Evaluation runner

## Parameters

### Main Evaluation Parameters
- `--model_dir` - Path to pre-trained model directory
- `--data_file` - Path to input dataset file (required)
- `--demon_file` - Path to demonstration examples file (required)
- `--dataset` - Dataset choice: triviaqa, taqa, wq, freshqa
- `--dynamic` - Enable dynamic instruction generation
- `--self_aware` - Enable self-aware evaluation

### Retrieval Parameters
- `--encoder` - BGE model name or path for embeddings
- `--max_query_length` - Maximum query token length (default: 256)
- `--max_passage_length` - Maximum passage token length (default: 256)
- `--fp16` - Use FP16 for inference (default: False)

## Example

```bash
# Basic evaluation
python main.py \
    --model_dir ./pretrain-model/Llama-2-7b-chat-hf \
    --data_file ./data/triviaqa_data.json \
    --demon_file ./data/demonstrations.json \
    --dataset triviaqa \
    --test_file ./outputs/results.json
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{lv-etal-2025-whether,
    title = "Whether {LLM}s Know If They Know: Identifying Knowledge Boundaries via Debiased Historical In-Context Learning",
    author = "Lv, Bo  and
      Liu, Nayu  and
      Shen, Yang  and
      Liu, Xin  and
      Luo, Ping  and
      Yu, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.999/",
    doi = "10.18653/v1/2025.findings-acl.999",
    pages = "19516--19528",
    ISBN = "979-8-89176-256-5"
}
```

## License

This project is licensed under the MIT License.
