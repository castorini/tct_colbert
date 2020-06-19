# Embeeding Search for IR
Here, we conduct the experiments using Siamese BERT-base model. The maximum query and passage lengths are set to 32 and 150 (not including special tokens) respectively. For each query (document), we also put \[CLS\] and \[Q\](\[D\]) in the beginning. For the queries no longer than length of 32, we pad them with \[MASK\] tokens.

Requirement
---
pip install mkl
conda install mkl-service --no-update-dependencies
pip install tensorflow-gpu
pip install numpy
pip install faiss
pip install faiss-gpu
pip install progressbar

Passage Re-ranking with max min pooling embeeding
---
Here, we use max-min pooling embedding with dimension 768 to represent each query and passage.
```shell=bash
mkdir prediction
candidate_file=msmarco-passage/top1000.dev.tsv
answer_file=msmarco-passage/qrels.dev.small.tsv
topk=1000
query_word_num=1
doc_word_num=1
emb_dim=768
data_type=16 # 16 or 32 bits for embedding storage
emb_path=./msmarco-passage/max_min_pooling
first_stage_path=first_stage_result
result_file=./prediction/rerank_result
###############################################
python rerank.py --candidate_file $candidate_file --emb_path max_min_pooling \
                 --topk $topk --emb_path $emb_path --data_type $data_type \
                 --query_word_num $query_word_num --doc_word_num $doc_word_num --emb_dim $emb_dim \
                 --result_file $result_file.tsv
# Evaluation
python ./convert_msmarco_to_trec_run.py --input_run $result_file.tsv --output_run $result_file.trec
python ./msmarco_eval.py \
 msmarco-passage/qrels.dev.small.tsv $result_file.tsv
./trec_eval.9.0.4/trec_eval -c -mrecall.1000 -mmap -mndcg_cut.1,3 \
 $answer_file $result_file.trec
```
We do not consider the time for query embedding generation into latency computation currently.
Results  | Dev
------------| :------:
MRR10            | 0.3098
mAP              | 0.3140
Recall@1000      | 0.8140
Latency (s/query)| 0.0070

End to End Passage Retrieval
---
```shell=bash
mkdir prediction index
answer_file=msmarco-passage/qrels.dev.small.tsv
topk=1000
query_word_num=1
doc_word_num=1
emb_dim=768
data_type=16 # 16 or 32 bits for embedding storage
index_file=./index/msmarco_max_min_pool
emb_path=./msmarco-passage/max_min_pooling
first_stage_path=first_stage_result
result_file=./prediction/rank_result.tsv
###############################################
# indexing (Turn on "quantize" option with faster search and space saving)
python index.py --num_files 10 --index_file $index_file \
     --topk $topk --emb_path $emb_path --data_type 32 \ #Faiss only accept 32 bits for indexing
     --query_word_num $query_word_num --doc_word_num $doc_word_num --emb_dim $emb_dim
     #--quantize
# First-stage search with faiss
python search.py --num_files 10 --index_file $index_file --pickle_file $first_stage_path/result.pickle\
    --topk $topk --emb_path $emb_path --data_type 32 \ #Faiss only accept 32 bits for indexing
    --query_word_num $query_word_num --doc_word_num $doc_word_num --emb_dim $emb_dim
# Output final result (Turn on "rerank" option if you want to rerank over the first-stage search result. For pooling embedding, we do not have to do that.)
python output_result.py --topk $topk --emb_path $emb_path --data_type $data_type --first_stage_path $first_stage_path\
                        --query_word_num $query_word_num --doc_word_num $doc_word_num \
                        --emb_dim $emb_dim --result_file $result_file \
                        #--rerank
# Evaluation
python ./convert_msmarco_to_trec_run.py --input_run $result_file.tsv --output_run $result_file.trec
python ./msmarco_eval.py \
 $answer_file $result_file.tsv
./trec_eval.9.0.4/trec_eval -c -mrecall.1000 -mmap -mndcg_cut.1,3 \
 $answer_file $result_file.trec
```
Results  | Dev
------------| :------:
MRR10            | 0.3013
mAP              | 0.3082
Recall@1000      | 0.9401
Latency (s/query)| 0.1000


ColBERT Passage Re-ranking reproduce
---
As for re-ranking with contextual word embedding matching , we currently only conduct re-ranking on colab TPU since it consumes much storage space (Each query and document are represented with embedding size 128\*32 and 128\*150 respectively).
Results  | Dev
------------| :------:
MRR10            | 0.3504
mAP              | 0.3531
Recall@1000      | 0.8140