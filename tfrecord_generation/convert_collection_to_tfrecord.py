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
import math
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
    "corpus",
    "msmarco",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_bool(
    "dry_run",
    False,
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "corpus_path",
    "./data/top1000.dev.tsv",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "doc_type", "doc",
    "Denote the type of document, either doc or query")


flags.DEFINE_integer(
    "num_eval_docs", 1000,
    "The maximum number of docs per query for dev and eval sets.")

flags.DEFINE_integer(
    "max_seg", 4,
    "The maximum number of docs per query for dev and eval sets.")


def write_to_tf_record(writer, tokenizer, doc_text,
                      docid, doc_type):
  if doc_type=='doc':
    doc_ids = tokenization.convert_to_colbert_input(
        text=['[','d',']']+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=False, tokenize=False)
  elif doc_type=="passage":
    doc_ids = tokenization.convert_to_colbert_input(
        text='[D] '+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=False)
  elif doc_type=="query":
    doc_ids = tokenization.convert_to_colbert_input(
        text='[Q] '+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=True)

  docid=int(docid)
  doc_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=doc_ids))


  docid_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[docid]))


  features = tf.train.Features(feature={
      'doc_ids': doc_ids_tf,
      'docid': docid_tf,
  })
  example = tf.train.Example(features=features)
  writer.write(example.SerializeToString())


def convert_passage_corpus(corpus, tokenizer, doc_type):
  print('Counting number of documents...')
  num_lines = sum(1 for line in open(FLAGS.corpus_path, 'r'))
  print('{} passages found.'.format(num_lines))

  remain = num_lines%40
  if doc_type=='query':
    remain = 0
  print('Converting {} to tfrecord...'.format(FLAGS.corpus_path))
  start_time = time.time()
  docids=[]
  docs=[]
  with open(FLAGS.corpus_path) as f:
      for line in f:
        try:
          docid, doc = line.strip().split('\t')
        except:
          doc=[]
          for i, content in enumerate(line.strip().split('\t')):
            if i==0:
              docid=content
            else:
              doc.append(content)

          doc=' '.join(doc)
        docids.append(docid)
        docs.append(doc)

  counter=0
  id_writer = open(FLAGS.output_folder + '/'+ corpus + '.id', 'w')
  writer = tf.python_io.TFRecordWriter(
      FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')

  for i, doc in enumerate(docs):

    if (i % (num_lines-remain) == 0) and (i!=0): #8841800
      writer.close()
      counter+=1
      writer = tf.python_io.TFRecordWriter(
        FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')


    write_to_tf_record(writer=writer,
                       tokenizer=tokenizer,
                       doc_text=doc,
                       docid=i,
                       doc_type=doc_type)

    id_writer.write('%d\t%s\n'%(i,docids[i]))
    if (i+1) % 1000000 == 0:
      print('Writing {} corpus, doc {} of {}'.format(
          corpus, i, len(docs)))
      time_passed = time.time() - start_time
      hours_remaining = (
          len(docs) - i) * time_passed / (max(1.0, i) * 3600)
      print('Estimated hours remaining to write the {} corpus: {}'.format(
          corpus, hours_remaining))

  writer.close()
  id_writer.close()


def convert_doc_corpus(corpus, tokenizer, doc_type, seg_length, max_seg):

  def aggregate_passages(doc, seg_length):
    passages = []
    tokens = tokenizer.tokenize(doc)
    token_len = len(tokens)
    seg_num = math.ceil(token_len/seg_length)
    for i in range(seg_num):
      passages.append(tokens[i*seg_length:(i+1)*seg_length])
    return passages

  print('Converting {} to tfrecord...'.format(FLAGS.corpus_path))

  print('Counting number of documents...')
  num_lines = sum(1 for line in open(FLAGS.corpus_path, 'r'))
  print('{} documents found.'.format(num_lines))
  start_time = time.time()

  if not FLAGS.dry_run:
    counter=0
    writer = tf.python_io.TFRecordWriter(
        FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')
    id_writer = open(FLAGS.output_folder + '/'+ corpus + '.id', 'w')
    num_id=0
    with open(FLAGS.corpus_path) as f:
      for doc_num, line in enumerate(f):

        content = line.strip().split('\t') #docid, url, title, doc

        docid = content[0]
        doc = ' '.join(content[1:])


        passages = aggregate_passages(doc, seg_length-4)

        # if len(passages)>1:
        #   print('check')

        for passage in passages[:max_seg]:
          write_to_tf_record(writer=writer,
                             tokenizer=tokenizer,
                             doc_text=passage,
                             docid=num_id,
                             doc_type=doc_type)

          id_writer.write('%d\t%s\n'%(num_id, docid))
          if (num_id+1) % 1000000 == 0:
            print('Writing {} corpus, doc {}, passage {} of {}'.format(
                corpus, doc_num, num_id, num_lines))
            time_passed = time.time() - start_time
            hours_remaining = (
                num_lines - doc_num) * time_passed / (max(1.0, doc_num) * 3600)
            print('Estimated hours remaining to write the {} corpus: {}'.format(
                corpus, hours_remaining))

          num_id+=1

          if (((num_id+1) % 7651000) == 0): #(chunk_len, max_seg): (154X16) 23654360+19 (256X8)
            print('New')
            writer.close()
            counter+=1
            writer = tf.python_io.TFRecordWriter(
              FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')
      writer.close()
      id_writer.close()
      print('total '+ str(num_id) + ' passages')
      print('finish')
  else:
    counter=0
    num_id=0
    with open(FLAGS.corpus_path) as f:
      for doc_num, line in enumerate(f):

        content = line.strip().split('\t') #docid, url, title, doc
        docid = content[0]
        doc = ' '.join(content[1:])


        passages = aggregate_passages(doc, seg_length-4)


        for passage in passages[:max_seg]:

          if (num_id+1) % 1000000 == 0:
            print('Writing {} corpus, doc {}, passage {} of {}'.format(
                corpus, doc_num, num_id, num_lines))
            time_passed = time.time() - start_time
            hours_remaining = (
                num_lines - doc_num) * time_passed / (max(1.0, doc_num) * 3600)
            print('Estimated hours remaining to write the {} corpus: {}'.format(
                corpus, hours_remaining))

          num_id+=1



      print('total '+ str(num_id) + ' passages')
      print('split '+ str(num_id-num_id%40) + '/' + str(num_id%40) + ' passages')
      print('finish')

def main():

  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)

  if FLAGS.doc_type!='doc': #passage or query
    convert_passage_corpus(corpus=FLAGS.corpus, tokenizer=tokenizer, doc_type=FLAGS.doc_type)
    print('Done!')
  else:
    convert_doc_corpus(corpus=FLAGS.corpus, tokenizer=tokenizer, doc_type=FLAGS.doc_type, seg_length=FLAGS.max_seq_length\
      , max_seg=FLAGS.max_seg)
    print('Done!')

if __name__ == '__main__':
  main()
