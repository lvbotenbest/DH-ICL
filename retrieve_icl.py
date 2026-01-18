import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from FlagEmbedding import FlagModel
from typing import cast, List, Union, Tuple
import csv
import json
from rouge import Rouge

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="bge model",
        metadata={'help': 'BGE model name or path for generating text embeddings (e.g., BAAI/bge-large-en-v1.5)'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use FP16 precision for inference to reduce memory usage and speed up computation'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction prompts to improve retrieval performance'}
    )
    
    max_query_length: int = field(
        default=256,
        metadata={'help': 'Maximum number of tokens for query sequences (truncated if longer)'}
    )
    max_passage_length: int = field(
        default=256,
        metadata={'help': 'Maximum number of tokens for passage sequences (truncated if longer)'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Batch size for embedding computation (higher values may use more memory but process faster)'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory string for building the retrieval index (e.g., Flat, IVF, PQ)'}
    )
    k: int = field(
        default=50,
        metadata={'help': 'Number of top-k similar passages to retrieve for each query'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save computed embeddings to disk for future reuse to avoid recomputation'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load pre-computed embeddings from disk instead of computing them again'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'File path where embeddings will be saved or loaded from'}
    )
    retrieve_file: str = field(
        default="",
        metadata={'help': 'Input file path containing queries to retrieve passages for'}
    )
    output_file: str = field(
        default="",
        metadata={'help': 'Output file path to save retrieval results and retrieved passages'}
    )
    corpus_file: str = field(
        default="",
        metadata={'help': 'Corpus file path containing the passage database for retrieval'}
    )
    search_batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size for search operations during retrieval (affects speed vs memory tradeoff)'}
    )



def index(model: FlagModel, corpus: Union[List[str], str], batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        corpus_embeddings = model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(model: FlagModel, queries, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries, batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices






def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    
    output_file=args.output_file


    # corpus_file = "/home/zhangyanan/zyn/lb/PreSelect/model_without_ret_acc/llama2_7b_wo_ret_6w.json"
    corpus_file = args.corpus_file

    with open(corpus_file,"r", encoding="utf-8") as f:

        corpus_data = json.load(f)

       


    corpus = [item["question"] for item in corpus_data]
    # question_ids = [item["question_id"] for item in corpus_data]
    labels = ["true" if  str(item["label"])=="1" else "false" for item in corpus_data]
    
    # labels = [item["wo_ret_answer_result"] for item in corpus_data]
    print("corpus length: ",len(corpus))

    print(len(corpus))


    eval_question = []
    not_ret_result = []
    ret_result = []


    retrieve_file=args.retrieve_file

    with open(retrieve_file, 'r') as f:
        # 读取 JSON 数据
       test_data = json.load(f)

    for item in test_data:
        
        eval_question.append(item["question"])
        not_ret_result.append(item["not_ret_result"])
        ret_result.append(item["ret_result"])

 
    print("evalution data length: ",len(eval_question))



    model = FlagModel(
        args.encoder, 
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
        use_fp16=args.fp16
    )
    
    faiss_index = index(
        model=model, 
        corpus=corpus, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_question, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.search_batch_size, 
        max_length=args.max_query_length
    )
    
  
    indice_list = []

    
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        indice_list.append(indice)
     



    print(len(indice_list))
    output_list = []

    for item in zip(indice_list,eval_question,not_ret_result,ret_result):

        
        demonstrate_datas=[]

        

        for indice in item[0]:
           
            demon_dict = dict(
                question=corpus[indice],
                not_ret_answer_result=labels[indice],
            )
            demonstrate_datas.append(demon_dict)
        

        output_list.append(
        dict(
            question=item[1],
            not_ret_result=item[2],
            ret_result=item[3],
            demon_data=demonstrate_datas,
        )
    )


    with open(output_file, 'w') as file:
        json.dump(output_list, file,ensure_ascii=False, indent=4)


       



if __name__ == "__main__":
    main()

