# Dense Retrieval with TCT-ColBERT
The repo is the code for our paper:
*[Distilling Dense Representations for Ranking
using Tightly-Coupled Teachers](https://arxiv.org/pdf/2010.11386.pdf)* Sheng-Chieh Lin, Jheng-Hong Yang and Jimmy Lin

**\*\*\*\*\* Most of the code in this repository was revised from [Passage Re-ranking with BERT repository](https://github.com/nyu-dl/dl4marco-bert).**\*\*\*\*\*

## MS Marco Dataset
```shell=bash
export DATA_DIR=./msmarco-passage
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
exprot DATA_DIR=./msmarco-passage
export MODEL_DIR=./uncased_L-12_H-768_A-12
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
To train TCT-ColBERT, first upload msmarco-passage/tfrecord to $Your_GS_Folder. Also upload BERT-based model ,[uncased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip), as well. Then, we can start to train our teacher model ColBERT! If do_eval set True, we also use the trained bi-encoder to rerank the Msmarco dev set. If using GPU, set use_tpu=False and remove tpu_address option.
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
               --output_dir=$Your_GS_Folder/colbert_checkpoint \
```
The ColBERT re-ranking result:
Reranking  | Dev
------------| :------:
MRR10            | 0.350

After training ColBERT, we then set $colbert_checkpoint to the ColBERT checkpoint and start training TCT-ColBERT. Note that the training step setting 160K is the one used in our Arxiv paper, [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers](https://arxiv.org/pdf/2010.11386.pdf). In this paper, [In-Batch Negatives for Knowledge Distillation with Tightly-Coupled
Teachers for Dense Retrieval](https://aclanthology.org/2021.repl4nlp-1.17/), we train TCT-ColBERT for 500K steps and got even better results.
```shell=bash
python train/main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_train=True \
               --do_eval=True \
               --train_model=student \
               --eval_model=student \
               --num_train_steps=160000 \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --output_dir=$Your_GS_Folder/colbert_checkpoint
               --data_dir=$Your_GS_Folder/msmarco-passage/tfrecord \
               --train_file=dataset_train_tower.tf \
               --eval_file=dataset_dev_tower.tf \
               --output_dir=$Your_GS_Folder/tct-colbert_checkpoint \
```
The TCT-ColBERT re-ranking result:
Reranking  | Dev
------------| :------:
MRR10            | 0.332

With the model, you can either convert the model to pytorch model conduct dense retrieval using [Pyserini](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert.md) or directly use our provided reference code below.

## TCT-ColBERT Embedding Output
We demonstrate how to encode the embeddings and conduct brute force search using Faiss.
### Msmarco Collection and Dev Queries Tfrecord Conversion
We first transform Msmarco collection and dev queries to tfrecord.
```shell=bash
export DATA_DIR=./msmarco-passage
export MODEL_DIR=./uncased_L-12_H-768_A-12
export QUERY_NAME=queries.dev.small
# We first split the collection into 10 parts
split -d -l 1000000 ${DATA_DIR}/collection.tsv ${DATA_DIR}/collection.part
# Convert passages in the collection
python ./tfrecord_generation/convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/corpus_tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=154 \
  --corpus_path=${DATA_DIR} \
  --corpus_prefix=collection.part \
  --doc_type=passage \
# Convert queries in dev set
python ./tfrecord_generation/convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/query_tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=36 \
  --corpus_path=${DATA_DIR}/${QUERY_NAME}.tsv \
  --doc_type=query \
  --output_filename=q${QUERY_NAME}
```

### TCT-ColBERT Corpus and Query Embedding Output
```shell=bash
#Output Corpus embeddings on CC
export DATA_DIR=./msmarco-passage
export MODEL_DIR=./uncased_L-12_H-768_A-12
for i in $(seq -f "%01g" 0 9)
do
  srun --gres=gpu:p100:1 --mem=16G --cpus-per-task=2 --time=2:00:00 \
  python train/main.py --use_tpu=False \
                 --tpu_address=$tpu_address \
                 --do_output=True \
                 --eval_model=student \
                 --bert_pretrained_dir=${MODEL_DIR} \
                 --eval_checkpoint=${CHECKPOINT} \
                 --max_doc_length=154 \
                 --doc_type=1 \
                 --eval_batch_size=100
                 --output_dir=${DATA_DIR}/doc_emb \
                 --data_dir=${DATA_DIR}/corpus_tfrecord \
                 --embedding_file=collection.part-${i} \
done


# Output Query embeddings
python train/main.py --use_tpu=False \
          --tpu_address=$tpu_address \
          --do_output=True \
          --eval_model=student \
          --bert_pretrained_dir=${MODEL_DIR} \
          --eval_checkpoint=${CHECKPOINT} \
          --output_dir=${DATA_DIR}/query_emb \
          --data_dir=${DATA_DIR}/query_tfrecord \ \
          --embedding_file=queries.dev.small \
          --doc_type=0 \
          --eval_batch_size=1 \
          # default doc_type 1: Passage; doc_type 0: Query
```


## Faiss Brute-force search with TCT-ColBERT Embeddings

### Requirement
tensorflow-gpu, faiss-gpu, progressbar

Indexing all MSMARCO passages in a file (Exhuasive search) requires 26 GB. For example, if only 4GB GPU is available for search, you can set max_passage_each_index to 1000,000 and 8 indexing files will be generated. Then, we search each index for topk passages, and merge and sort them to get the final ranking result. Here, we use average pooling embedding with dimension 768 (with 32 bits) to represent each query and passage. Similar to re-ranking, first store your query and passage embeddings tf record in the folders query_emb and corpus_emb respectively, and put qrel and id_to_query files in the current folder.
```shell=bash
export CORPUS_EMB=./msmarco-passage/doc_emb
export QUERY_EMB=./msmarco-passage/query_emb
export QUERY_NAME=queries.dev.small
export INDEX_PATH=./msmarco-passage/indexes
exprot DATA_DIR=./msmarco-passage
export INTERMEDIATE_PATH=./msmarco-passage/intermediate
###############################################
# indexing using faiss
python ./dr/index.py --index_path ${INDEX_PATH} \
     --corpus_emb_path ${CORPUS_EMB} --passages_per_file 1000000 \

# First-stage search with Faiss

for index in ${INDEX_PATH}/*
do
    python ./dr/search.py --index_file $index --intermediate_path ${INTERMEDIATE_PATH} \
          --topk 1000 --query_emb_path ${QUERY_EMB}/embeddings-${QUERY_NAME}.tf \
          --batch_size 144 --threads 36
done

# Merge and output final result
python ./dr/output_result.py --topk 1000 --intermediate_path ${INTERMEDIATE_PATH} \
                         --result_file result.tsv \
                         --id_to_doc_path ${DATA_DIR}/corpus_tfrecord \
                         --id_to_query_path ${DATA_DIR}/query_tfrecord

# Evaluation
python3 ./eval/msmarco_eval.py \
 qrels.dev.small.tsv result.tsv

```
The TCT-ColBERT retrieval result:
Retrieval  | Dev |
| ------------ | ----------- |
MRR10      | 0.335 |




