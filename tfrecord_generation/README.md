# Generate Tfrecord for Query and Document
Here, we conduct the experiments using Siamese BERT-base model. The maximum query and passage lengths are set to 32 and 150 (not including special tokens) respectively. For each query (document), we also put \[CLS\] and \[Q\](\[D\]) in the beginning. For the queries no longer than length of 32, we pad them with \[MASK\] tokens.

Msmarco Collection and Dev Queries Tfrecord Conversion
---
```shell=bash
DATA_DIR=./msmarco-passage
MODEL_DIR=./uncased_L-12_H-768_A-12
mkdir ${DATA_DIR}/tfrecord
# Convert passages in the collection
python ./convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=154 \ #150 plus 4 tokens '[CLS]', '[', 'D', ']'
  --corpus_path ${MODEL_DIR}/collection.tsv\
  --doc_type doc\
  --corpus msmarco
# Convert queries in dev set
python ./convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_seq_length=36 \ #32 plus 4 tokens '[CLS]', '[', 'Q', ']'
  --corpus_path ${MODEL_DIR}/queries.dev.small.tsv\
  --doc_type query\
  --corpus queries.dev
```

Convert First-stage retrieval candidate for TPU re-ranking
---
```shell=bash
DATA_DIR=./msmarco-passage
MODEL_DIR=./uncased_L-12_H-768_A-12
mkdir ${DATA_DIR}/tfrecord
python ./convert_msmarco_to_tfrecord_tower.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --max_query_length=36\
  --max_seq_length=400 \
  --num_eval_docs=1000 \
  --dev_qrels_path=${DATA_DIR}/qrels.dev.small.tsv \
  --dev_dataset_path=${DATA_DIR}/top1000.dev.tsv \
```
