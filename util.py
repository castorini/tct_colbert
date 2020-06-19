import os
os.environ['OMP_NUM_THREADS'] = str(16)
import faiss
import mkl
mkl.set_num_threads(16)
import numpy as np
import tensorflow.compat.v1 as tf
from numpy import linalg as LA
from progressbar import *
from collections import defaultdict


def dedupe_index(Index, Distance):
    NewIndex=[]
    max_index_num = 0
    min_index_num = 10000000
    doc_num=0
    sort_index=np.argsort(Distance)[:,::-1]
    for i, index in enumerate(Index):
        sort_id = index[sort_index[i].tolist()]
        uniq_indices=np.sort(np.unique(sort_id,return_index=True)[1]).tolist()#the index for dedupe
        max_index_num = max(max_index_num, len(uniq_indices))
        min_index_num  = min(min_index_num, len(uniq_indices))
        doc_num+=len(uniq_indices)
        NewIndex.append(sort_id[uniq_indices].tolist())
    print("Maximum unique doc id is %d after dedupe"%(max_index_num))
    print("Minimum unique doc id is %d after dedupe"%(min_index_num))
    print("Average unique doc id is %d after dedupe"%(doc_num/Index.shape[0]))
    return NewIndex, max_index_num
def read_pickle(filename):
    with open(filename, 'rb') as f:
        Distance, Index=pickle.load(f)
    return Distance, Index

def read_candidate(file):
    qid_to_qrel = defaultdict(list)
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            qid, qrel, _, _ =line.strip().split('\t')
            qid_to_qrel[int(qid)].append(int(qrel))
    return qid_to_qrel

def write_result(qids, Index, file):
	with open(file, 'w') as fout:
		for i, qid in enumerate(qids):
			pids=Index[i]
			for rank, pid in enumerate(pids):
				fout.write('{}\t{}\t{}\n'.format(qid, pid,rank + 1))


def load_tfrecords(srcfiles, data_num, word_num, dim, data_type='16', batch=1000):
    def _parse_function(example_proto):
        features = {'doc_emb': tf.FixedLenFeature([],tf.string),
                  'docid': tf.FixedLenFeature([],tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        if data_type=='16':
            corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float16)
        elif data_type=='32':
            corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float32)
        docid = tf.cast(parsed_features['docid'], tf.int32)
        return corpus, docid

    with tf.Session() as sess:
      docids=[]
      if data_type=='16':
        corpus_embs = np.zeros((data_num*len(srcfiles) , word_num*dim), dtype=np.float16) #assign memory in advance so that we can save memory without concatenate
      elif data_type=='32':
        corpus_embs = np.zeros((data_num*len(srcfiles) , word_num*dim), dtype=np.float32)
      counter = 0
      for srcfile in srcfiles:
        dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
        dataset = dataset.map(_parse_function) # parse data into tensor
        dataset = dataset.repeat(1)
        dataset = dataset.batch(batch)
        iterator = dataset.make_one_shot_iterator()
        next_data = iterator.get_next()

        while True:
          try:
            corpus_emb, docid = sess.run(next_data)
            sent_num = corpus_emb.shape[0]
            corpus_embs[counter:(counter+sent_num)] = corpus_emb
            docids.append(docid)

            counter+=sent_num
          except tf.errors.OutOfRangeError:

            break
      return corpus_embs[:counter], docids


def normalize(embeddings):
	return (embeddings.T/LA.norm(embeddings,axis=-1)).T