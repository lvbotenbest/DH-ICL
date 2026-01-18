CUDA_VISIBLE_DEVICES=1 python retrieve_icl.py \
                       --k 20 \
                       --batch_size 1024 \
                       --retrieve_file ./input/input_data.json \
                       --output_file ./output/output_data.json \
                       --corpus_file ./corpus/corpus.json \

