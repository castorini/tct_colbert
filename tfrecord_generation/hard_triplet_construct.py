import random
from collections import defaultdict
import tensorflow as tf
import tokenization
import time
import numpy as np
from random import choices
flags = tf.flags
FLAGS = flags.FLAGS



flags.DEFINE_string(
    "pos_candidate_file", None,
    "passage ranking file: qid \t docid \t rank \t score")
flags.DEFINE_string(
    "candidate_file", None,
    "passage ranking file: qid \t docid \t rank \t score")
flags.DEFINE_string(
    "qrel", None,
    "qrel file path")
flags.DEFINE_string(
    "qid_to_query", None,
    "query file path: qid \t query")
flags.DEFINE_string(
    "collection", None,
    "collection file path: docid \t document")
flags.DEFINE_string(
    "output_folder", None,
    "output tfrecord file path")
flags.DEFINE_integer(
    "max_seq_length", 400,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_integer(
    "max_query_length", 36,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")
flags.DEFINE_string(
    "vocab_file",
    "./data/bert/uncased_L-24_H-1024_A-16/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer(
    "neg_samples", 1,
    "Number of negative samples for each query.")
flags.DEFINE_integer(
    "repetition", 3,
    "Number of negative samples for each query.")
flags.DEFINE_integer(
    "random_seed", 42,
    "Random stat for drawing negative samples.")
flags.DEFINE_integer(
    "topk", 1000,
    "Randomly sample from topk candidates")
flags.DEFINE_integer(
    "topk_pos", 5,
    "Randomly sample from topk candidates")

def write_to_tf_record(writer, tokenizer, query, docs, labels,
                       ids_file=None, query_id=None, doc_ids=None, is_train=True):
  query = tokenization.convert_to_unicode(query)
  query_token_ids, _ = tokenization.convert_to_colbert_input(
      text='[Q] '+query, max_seq_length=FLAGS.max_query_length, tokenizer=tokenizer,
      add_cls=True, filtering=False, padding_mask=True)

  query_token_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_token_ids))



  feature = {}
  feature['query_ids']=query_token_ids_tf
  for i, (doc_text, label) in enumerate(zip(docs, labels)):

    doc_token_ids, filter_mask = tokenization.convert_to_colbert_input(
          text='[D] '+doc_text,
          max_seq_length=FLAGS.max_seq_length,
          tokenizer=tokenizer,
          add_cls=True, filtering=False, padding_mask=False)



    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_ids))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]))


    if is_train:
      feature['doc_ids'+str(label)]=doc_ids_tf
    else:
      feature['doc_ids']=doc_ids_tf



    feature['label']=labels_tf
    if ids_file:
      ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')

    if not is_train:
      features = tf.train.Features(feature=feature)
      example = tf.train.Example(features=features)
      writer.write(example.SerializeToString())

  if is_train:
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())

def convert_train_dataset(tokenizer):
  random.seed(FLAGS.random_seed)

  print('Read qrels...')
  qrels = defaultdict(list)
  with open(FLAGS.qrel, 'r') as f:
    for line in f:
      qid, _, docid, _ = line.split('\t')
      qrels[qid].append(docid)
  print('Read positive candidate file...')
  pos_qid_docid = defaultdict(list)
  with open(FLAGS.pos_candidate_file, 'r') as f:
    for i, line in enumerate(f):
      qid, docid, rank, _ = line.split('\t')
      rank = int(rank)
      if rank <= 200:
        pos_qid_docid[qid].append(docid)
      # if i >1000*50:
      #   break

  # print('Read query sim file...')
  # qid_sim_qid = defaultdict(list)
  # with open(FLAGS.query_sim_file, 'r') as f:
  #   for i, line in enumerate(f):
  #     qid, sim_qid, rank, _ = line.split('\t')
  #     rank = int(rank)
  #     if rank!=1:
  #       qid_sim_qid[qid].append(sim_qid)



  print('Read qrels...')
  qrels = defaultdict(list)
  with open(FLAGS.qrel, 'r') as f:
    for line in f:
      qid, _, docid, _ = line.split('\t')
      qrels[qid].append(docid)

  print('Read queries...')
  qid2query = {}
  with open(FLAGS.qid_to_query, 'r') as f:
    for line in f:
      qid, query = line.split('\t')
      qid2query[qid] = query.strip()

  print('Read candidate file...')
  qrank = defaultdict(lambda:1000)
  qid_docid = defaultdict(list)
  qid_doc_score = defaultdict(list)
  hard_neg_qid_docid = defaultdict(list)
  with open(FLAGS.candidate_file, 'r') as f:
    for i, line in enumerate(f):
      qid, docid, rank, score = line.split('\t')
      rank = int(rank)
      if rank<=FLAGS.topk:
        qid_docid[qid].append(docid)
        qid_doc_score[qid].append(float(score))
        # if (docid not in pos_qid_docid[qid]) and (rank <= 100):
        #   hard_neg_qid_docid[qid].append(docid)
      if docid in qrels[qid]:
        qrank[qid] = rank
      # if i >1000*50:
      #   break

  print('Collection...')
  docid2doc = {}
  with open(FLAGS.collection, 'r') as f:
    for line in f:
      docid, doc = line.split('\t')
      docid2doc[docid] = doc.strip()


  num_lines = len(qid_docid.keys())
  print('{} examples found.'.format(num_lines))
  writer = tf.python_io.TFRecordWriter(
       FLAGS.output_folder + '/dataset_hard_train_tower.tf')
  print('Converting Train to tfrecord...')
  start_time = time.time()

  for j in range(FLAGS.repetition):
    for i, qid in enumerate(qid_docid.keys()):


      if i % 1000 == 0:
        time_passed = int(time.time() - start_time)
        print('Processed training set, round {} line {} of {} in {} sec'.format(
            j, i, num_lines, time_passed))
        hours_remaining = (num_lines*FLAGS.repetition - i - j*num_lines) * time_passed / (max(1.0, i + j*num_lines) * 3600)
        print('Estimated hours remaining to write the training set: {}'.format(
            hours_remaining))


      query = qid2query[qid]
      neg_samples = FLAGS.neg_samples

      negative_docids = random.sample(qid_docid[qid][:FLAGS.topk] ,neg_samples)
      # negative_docids = choices(hard_neg_qid_docid[qid]+qid_docid[qid][:FLAGS.topk], qid_doc_score[qid][:FLAGS.topk])


      try:
        positive_docids = qrels[qid]
        positive_docid = random.sample(positive_docids,1)[0]
        positive_doc = docid2doc[positive_docid]
      except:
        positive_docid = pos_qid_docid[qid][0]
        positive_doc = docid2doc[positive_docid]
        # continue


      for k, negative_docid in enumerate(negative_docids):

        negative_doc = docid2doc[negative_docid]

        write_to_tf_record(writer=writer,
                       tokenizer=tokenizer,
                       query=query,
                       docs=[positive_doc, negative_doc],
                       labels=[1, 0])

  writer.close()



def main():
    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)
    convert_train_dataset(tokenizer)


if __name__ == '__main__':
  main()