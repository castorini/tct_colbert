# Dense Retrieval with TCT-ColBERT
The repo is the code for our paper:
*[Distilling Dense Representations for Ranking
using Tightly-Coupled Teachers](https://arxiv.org/pdf/2010.11386.pdf)* Sheng-Chieh Lin, Jheng-Hong Yang and Jimmy Lin

**\*\*\*\*\* Most of the code in this repository was revised from [Passage Re-ranking with BERT repository](https://github.com/nyu-dl/dl4marco-bert).**\*\*\*\*\* 

## MS Marco Dataset
```shell=bash
DATA_DIR=./msmarco-passage
mkdir ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv -P ${DATA_DIR}
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
tar -xvf ${DATA_DIR}/collection.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.dev.tar.gz -C ${DATA_DIR}
unzip uncased_L-24_H-1024_A-16.zip
```

## Convert Msmarco Train and dev set to Tfrecord
Here we set max passage length 150 plus 4 tokens '[CLS]', '[', 'D', ']', and max query length 32 plus 4 tokens '[CLS]', '[', 'Q', ']'.
```shell=bash
DATA_DIR=./msmarco-passage
MODEL_DIR=./uncased_L-12_H-768_A-12
mkdir ${DATA_DIR}/tfrecord
# Generate training data
python ./tfrecord_generation/convert_msmarco_to_tfrecord_tower.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_query_length=36\
  --max_seq_length=154 \
  --num_eval_docs=1000 \
  --train_dataset_path=msmarco-passage/triples.train.small.tsv \
# Generate dev set for re-ranking
python ./tfrecord_generation/convert_msmarco_to_tfrecord_tower.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_query_length=36\
  --max_seq_length=154 \
  --num_eval_docs=1000 \
  --dev_qrels_path=${DATA_DIR}/qrels.dev.small.tsv \
  --dev_dataset_path=${DATA_DIR}/top1000.dev.tsv \
```

## TCT-ColBERT Training
To train TCT-ColBERT, first store dataset_train_tower.tf and dataset_dev_tower.tf in $Your_GS_Folder/msmarco-passage. Then, we can start to train our teacher model ColBERT! If do_eval set True, we also use the trained bi-encoder to rerank the Msmarco dev set. If using GPU, set use_tpu=False and remove tpu_address option.
```shell=bash
python train/main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_train=True \
               --do_eval=True \
               --train_model=teacher \
               --eval_model=teacher \
               --num_train_steps=160000 \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --init_checkpoint=$Your_GS_Folder/uncased_L-12_H-768_A-12/bert_model.ckpt \
               --data_dir=$Your_GS_Folder/msmarco-passage/tfrecord \
               --train_file=dataset_train_tower.tf \
               --eval_file=dataset_dev_tower.tf \
               --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/colbert \
```
The ColBERT re-ranking result:
Reranking  | Dev
------------| :------:
MRR10            | 0.350

After training ColBERT, we then set $colbert_checkpoint to the ColBERT checkpoint and start training TCT-ColBERT.
```shell=bash
python train/main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_train=True \
               --do_eval=True \
               --train_model=student \
               --eval_model=student \
               --num_train_steps=160000 \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --init_checkpoint=$colbert_checkpoint \
               --data_dir=$Your_GS_Folder/msmarco-passage/tfrecord \
               --train_file=dataset_train_tower.tf \
               --eval_file=dataset_dev_tower.tf \
               --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/tct-colbert \
```
The TCT-ColBERT re-ranking result:
Reranking  | Dev
------------| :------:
MRR10            | 0.332

With the model, you can either convert the model to pytorch model conduct dense retrieval using [Pyserini](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert.md) or directly use our provided reference code below.

## TCT-ColBERT Embedding Output

### Msmarco Collection and Dev Queries Tfrecord Conversion
We first transform Msmarco collection and dev queries to terecord, and upload the msmarco-passage/tfrecord to $Your_GS_Folder for TPU inference.
```shell=bash
DATA_DIR=./msmarco-passage
MODEL_DIR=./uncased_L-12_H-768_A-12
mkdir ${DATA_DIR}/tfrecord
# Convert passages in the collection
python ./tfrecord_generation/convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=154 \
  --corpus_path=${DATA_DIR}/collection.tsv \
  --doc_type=passage \
  --corpus=msmarco
# Convert queries in dev set
python ./tfrecord_generation/convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=36 \
  --corpus_path=${DATA_DIR}/queries.dev.small.tsv \
  --doc_type=query \
  --corpus=queries.dev.small
```

### TCT-ColBERT Corpus and Query Embedding Output
To output corpus embedding (with 16 bit), we set default eval_batch_size to 40 and num_tpu_cores to 8. Since the MARCO passage corpus contains 8841823 passages so we split the passages into two tf recrod files, msmarco0.tf (8841800 passages) and msmarco1.tf (23 passages) and then output them saperatley. It takes within one hour.
```shell=bash
#Output Corpus embeddings
python train/main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_output=True \
               --eval_model=student \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --eval_checkpoint=$tct-colbert_checkpoint \
               --output_dir=$output_folder \
               --data_dir=$Your_GS_Folder/msmarco-passage/tfrecord \
               --embedding_file=msmarco0 \

python train/main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_output=True \
               --eval_model=student \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --eval_checkpoint=$tct-colbert_checkpoint \
               --output_dir=$output_folder \
               --data_dir=$Your_GS_Folder/msmarco-passage/tfrecord \
               --embedding_file=msmarco1 \
               --num_tpu_cores=1 \
               --eval_batch_size=1 \ 

# Output Query embeddings
python train/main.py --use_tpu=True \
          --tpu_address=$tpu_address \
          --do_output=True \
          --eval_model=student \
          --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
          --eval_checkpoint=$tct-colbert_checkpoint \
          --output_dir=$output_folder \
          --data_dir=$Your_GS_Folder/msmarco-passage \
          --embedding_file=queries.dev.small \
          --num_tpu_cores=1 \
          --eval_batch_size=20 \
          --doc_type=0 \ # default doc_type 1: Passage; doc_type 0: Query
```


## Faiss Brute-force search with TCT-ColBERT Embeddings

### Requirement
tensorflow-gpu, faiss-gpu, progressbar

Indexing all MSMARCO passages in a file (Exhuasive search) requires 26 GB. For example, if only 4GB GPU is available for search, you can set max_passage_each_index to 1000,000 and 8 indexing files will be generated. Then, we search each index for topk passages, and merge and sort them to get the final ranking result. Here, we use average pooling embedding with dimension 768 (with 32 bits) to represent each query and passage. Similar to re-ranking, first store your query and passage embeddings tf record in the folders query_emb and corpus_emb respectively, and put qrel and id_to_query files in the current folder.
```shell=bash

qerl_file=msmarco-passage/qrels.dev.small.tsv
topk=1000
num_files=2 # We currently save passage embeddings into 2 files: msmarco0.tf, msmarco1.tf.
max_passage_each_index=10,000,000
num_index=0 # It depends on max_passage_each_index: num_index=8.8M/max_passage_each_index
data_type=32 # 16 or 32 bits for embedding storage
corpus_type=passage 
index_file=indexes/msmarco_$corpus_type
gpu_device=0
query_emb=./query_emb/queries.dev.small.tf
corpus_emb=./corpus_emb/msmarco
id_to_query=./msmarco-passage/tfrecord/queries.dev.small.id # the lookup id map from query tfrecord generation 
first_stage_path=first_stage_result
result_file=./prediction/rank_result
mkdir prediction indexes $first_stage_path
###############################################
# indexing
python3 ./dr/index.py --num_files $num_files --index_file $index_file \
     --topk $topk --corpus_emb_path $corpus_emb --data_type $data_type \
     --corpus_type $corpus_type --max_passage_each_index $max_passage_each_index \
# First-stage search with faiss
for i in {0..$num_index}
do
    python ./dr/search.py --offset $i --index_file $index_file\_$i --pickle_file $first_stage_path/result\_$i.pickle \
        --topk $topk --query_emb_path $query_emb --data_type 32 \
        --query_word_num $query_word_num --emb_dim $emb_dim --batch_size 1 --use_gpu \
        --gpu_device $gpu_device --passage_per_index $max_passage_each_index
done
# Output final result
python3 ./dr/output_result.py --topk $topk --emb_path $emb_path --data_type $data_type --first_stage_path $first_stage_path\
                         --result_file $result_file.tsv \
                         --corpus_type $corpus_type \
                         --id_to_query_path $id_to_query \

# Evaluation
python3 ./eval/convert_msmarco_to_trec_run.py --input_run $result_file.tsv --output_run $result_file.trec
python3 ./eval/msmarco_eval.py \
 $qerl_file $result_file.tsv
./trec_eval.9.0.4/trec_eval -c -mrecall.1000 \
 $qerl_file $result_file.trec
```
The TCT-ColBERT retrieval result:
Retrieval  | Dev
------------| :------:
MRR10            | 0.335




