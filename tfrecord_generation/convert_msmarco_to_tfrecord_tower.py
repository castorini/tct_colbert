"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import re
import tensorflow as tf
import time
# local module
import tokenization
# import spacy; from spacy.lang.en import English; nlp = English()
# nlp.add_pipe(nlp.create_pipe('sentencizer'))


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "output_folder", None,
    "Folder where the tfrecord files will be written.")

flags.DEFINE_string(
    "vocab_file",
    "./data/bert/uncased_L-24_H-1024_A-16/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "train_dataset_path",
    None,
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "dev_dataset_path",
    None,
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "eval_dataset_path",
    None,
    "Path to the MSMARCO eval dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "dev_qrels_path",
    "./data/qrels.dev.tsv",
    "Path to the query_id relevant doc ids mapping.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_eval_docs", 1000,
    "The maximum number of docs per query for dev and eval sets.")


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
    # docs=[]
    # for doc in nlp(doc_text).sents:
    #   docs.append(doc)
    doc_token_ids, filter_mask = tokenization.convert_to_colbert_input(
          text='[D] '+doc_text,
          max_seq_length=FLAGS.max_seq_length,
          tokenizer=tokenizer,
          add_cls=True, filtering=False, padding_mask=False)



    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_ids))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]))

    # doc_filter_tf = tf.train.Feature(
    #     int64_list=tf.train.Int64List(value=filter_mask))


    if is_train:
      feature['doc_ids'+str(label)]=doc_ids_tf
      # feature['doc_filter'+str(label)]=doc_filter_tf
    else:
      feature['doc_ids']=doc_ids_tf
      # feature['doc_filter']=doc_filter_tf



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


def convert_eval_dataset(set_name, tokenizer):
  print('Converting {} set to tfrecord...'.format(set_name))
  start_time = time.time()

  if set_name == 'dev':
    dataset_path = FLAGS.dev_dataset_path
    relevant_pairs = set()
    with open(FLAGS.dev_qrels_path) as f:
      for line in f:
        query_id, _, doc_id, _ = line.strip().split('\t')
        relevant_pairs.add('\t'.join([query_id, doc_id]))
  else:
    dataset_path = FLAGS.eval_dataset_path

  queries_docs = collections.defaultdict(list)
  query_ids = {}
  with open(dataset_path, 'r') as f:
    for i, line in enumerate(f):
      #query_id, doc_id, query, doc, _, _ = line.strip().split('\t')
      query_id, doc_id, query, doc = line.strip().split('\t')
      label = 0
      if set_name == 'dev':
        if '\t'.join([query_id, doc_id]) in relevant_pairs:
          label = 1
      queries_docs[query].append((doc_id, doc, label))
      query_ids[query] = query_id

  # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
  queries = list(queries_docs.keys())  # Need to copy keys before iterating.
  for query in queries:
    docs = queries_docs[query]
    docs += max(
        0, FLAGS.num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
    queries_docs[query] = docs

  assert len(
      set(len(docs) == FLAGS.num_eval_docs for docs in queries_docs.values())) == 1, (
          'Not all queries have {} docs'.format(FLAGS.num_eval_docs))

  writer = tf.python_io.TFRecordWriter(
      FLAGS.output_folder + '/dataset_' + set_name + '_tower.tf')

  query_doc_ids_path = (
      FLAGS.output_folder + '/query_doc_ids_' + set_name + '.txt')

  with open(query_doc_ids_path, 'w') as ids_file:
    for i, (query, doc_ids_docs) in enumerate(queries_docs.items()):
      doc_ids, docs, labels = zip(*doc_ids_docs)
      query_id = query_ids[query]

      write_to_tf_record(writer=writer,
                         tokenizer=tokenizer,
                         query=query,
                         docs=docs,
                         labels=labels,
                         ids_file=ids_file,
                         query_id=query_id,
                         doc_ids=doc_ids,
                         is_train=False)

      if i % 100 == 0:
        print('Writing {} set, query {} of {}'.format(
            set_name, i, len(queries_docs)))
        time_passed = time.time() - start_time
        hours_remaining = (
            len(queries_docs) - i) * time_passed / (max(1.0, i) * 3600)
        print('Estimated hours remaining to write the {} set: {}'.format(
            set_name, hours_remaining))
  writer.close()


def convert_train_dataset(tokenizer):
  print('Converting Train to tfrecord...')

  start_time = time.time()

  print('Counting number of examples...')
  num_lines = sum(1 for line in open(FLAGS.train_dataset_path, 'r'))
  print('{} examples found.'.format(num_lines))
  writer = tf.python_io.TFRecordWriter(
      FLAGS.output_folder + '/dataset_train_tower.tf')

  with open(FLAGS.train_dataset_path, 'r') as f:
    for i, line in enumerate(f):
      if i % 1000 == 0:
        time_passed = int(time.time() - start_time)
        print('Processed training set, line {} of {} in {} sec'.format(
            i, num_lines, time_passed))
        hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
        print('Estimated hours remaining to write the training set: {}'.format(
            hours_remaining))

      try:
        query, positive_doc, negative_doc = line.rstrip().split('\t')
      except:
        print("debug")

      write_to_tf_record(writer=writer,
                         tokenizer=tokenizer,
                         query=query,
                         docs=[positive_doc, negative_doc],
                         labels=[1, 0])

  writer.close()


  # start_time = time.time()

  # print('Counting number of examples...')
  # num_lines = sum(1 for line in open(FLAGS.train_dataset_path, 'r'))
  # print('{} examples found.'.format(num_lines))
  # writer = tf.python_io.TFRecordWriter(
  #     FLAGS.output_folder + '/dataset_train.tf')

  # with open(FLAGS.train_dataset_path, 'r') as f:
  #   for i, line in enumerate(f):
  #     if i % 1000 == 0:
  #       time_passed = int(time.time() - start_time)
  #       print('Processed training set, line {} of {} in {} sec'.format(
  #           i, num_lines, time_passed))
  #       hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
  #       print('Estimated hours remaining to write the training set: {}'.format(
  #           hours_remaining))
  #     try:
  #       sample_id, rel, query, doc= line.rstrip().split('\t')
  #     except:
  #       continue


  #     if rel=='pos':
  #       positive_doc = doc
  #     elif rel=='neg':
  #       negative_doc = doc
  #       write_to_tf_record(writer=writer,
  #                          tokenizer=tokenizer,
  #                          query=query,
  #                          docs=[positive_doc, negative_doc],
  #                          labels=[1, 0])
  #     # if i>=1000:
  #     #   break

  # writer.close()

def main():

  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)


  if FLAGS.dev_dataset_path!=None:
    convert_eval_dataset(set_name='dev', tokenizer=tokenizer)
  if FLAGS.train_dataset_path!=None:
    convert_train_dataset(tokenizer=tokenizer)

  # if FLAGS.eval_dataset_path:
  #   convert_eval_dataset(set_name='eval', tokenizer=tokenizer)
  print('Done!')

if __name__ == '__main__':
  main()
