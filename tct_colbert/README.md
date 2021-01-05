# TCT-ColBERT training and embedding output using TPU

TCT-ColBERT Training
---
To train TCT-ColBERT, we first convert triplet.small.train.tsv to dataset_train_tower.tf. If do_eval set True, we also use the trained bi-encoder to rerank the Msmarco dev set (we also convert top1000.dev.tsv to dataset_dev_tower.tf in advance).
```shell=bash
python main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_train=True \
               --do_eval=True \
               --init_checkpoint=model.ckpt-0 \
               --num_train_steps=160000 \
               --output_dir=$Your_GS_Folder \
               --kd_source=colbert \
               --loss=kl \
               --train_file=dataset_train_tower.tf

```
TCT-ColBERT Corpus Embedding Output
---
To output corpus embedding, we set default eval_batch_size to 40 and num_tpu_cores to 8. Since the MARCO passage corpus contains 8841823 passages so we split the passages into two tf recrod files, msmarco0.tf (8841800 passages) and msmarco1.tf (23 passages) and then output them saperatley.
```shell=bash
#Output Corpus embeddings
python main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_output=True \
               --init_checkpoint=model.ckpt-160000 \
               --output_dir=gs://jackir/output \
               --embedding_file=msmarco0 \

python main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_output=True \
               --init_checkpoint=model.ckpt-160000 \
               --output_dir=$Your_GS_Folder \
               --embedding_file=msmarco1 \
               --num_tpu_cores=1 \
               --eval_batch_size=1 \

```
TCT-ColBERT Query Embedding Output
---
```shell=bash
# Output Query embeddings
for file in queries.dev.small0 dl2019.queries.eval0
do
    python main.py --use_tpu=True \
                   --tpu_address=$tpu_adress \
                   --do_output=True \
                   --init_checkpoint=model.ckpt-160000 \
                   --embedding_file=$file \
                   --num_tpu_cores=1 \
                   --eval_batch_size=20 \
                   --doc_type=0 \ # default doc_type 1: Passage; doc_type 0: Query
done

```
