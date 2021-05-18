import os
import pickle
os.environ['OMP_NUM_THREADS'] = str(16)
import faiss
# import mkl
# mkl.set_num_threads(16)
import numpy as np
import tensorflow.compat.v1 as tf
from numpy import linalg as LA
from progressbar import *
from collections import defaultdict


def read_pickle(filename):
	with open(filename, 'rb') as f:
		Distance, Index=pickle.load(f)
	return Distance, Index


def read_id_dict(path):
	if path == None:
		return None, None
	idx_to_id = {}
	id_to_idx = {}
	f = open(path, 'r')
	for i, line in enumerate(f):
		try:
			idx, Id =line.strip().split('\t')
			idx_to_id[int(idx)] = Id
			id_to_idx[Id] = int(idx)
		except:
			print(line+' has no id')
	return idx_to_id, id_to_idx

def write_result(qidxs, Index, Score, file, idx_to_qid, idx_to_docid, topk=None, doc_prefix = ''):
	print('write results...')
	with open(file, 'w') as fout:
		for i, qidx in enumerate(qidxs):
			try:
				qid = idx_to_qid[qidx]
			except:
				qid = qidx
			if topk==None:
				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{}\t{}\t{}\t{}\n'.format(qid, docid, rank + 1, scores[rank]))
			else:
				hit=min(topk, len(Index[i]))
				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs[:hit]):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{}\t{}\t{}\t{}\n'.format(qid, docid, rank + 1, scores[rank]))


def load_tfrecords(srcfiles, data_num, word_num, dim, data_type='16', index=False, batch=1000):
	def _parse_function(example_proto):
		features = {'doc_emb': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
					'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		if data_type=='16':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float16)
		elif data_type=='32':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return corpus, docid
	print('Read embeddings...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		#assign memory in advance so that we can save memory without concatenate
		if (data_type=='16') and not index: # Faiss now only support index array with float32
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float16)
		elif data_type=='32':
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float32)
		else:
			raise Exception('Please assign datatype 16 or 32 bits')
		counter = 0
		i = 0
		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()

			while True:
				try:
					corpus_emb, docid = sess.run(next_data)
					corpus_emb = corpus_emb.reshape(-1, dim)

					sent_num = corpus_emb.shape[0]
					corpus_embs[counter:(counter+sent_num)] = corpus_emb
					docids.append(docid)
					counter+=sent_num
					pbar.update(10 * i + 1)
					i+=sent_num

				except tf.errors.OutOfRangeError:
					break
	return corpus_embs[:counter], docids

def load_tfrecords_doc(srcfiles, data_num, word_num, id_to_doc, new_id_to_doc_path, dim, p_max_num, data_type='16', index=False, batch=1):
	def _parse_function(example_proto):
		features = {'doc_emb': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
		'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		if data_type=='16':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float16)
		elif data_type=='32':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return corpus, docid

	print('Read embeddings...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		if data_type=='16' and not index:
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float16) #assign memory in advance so that we can save memory without concatenate
		elif data_type=='32':
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float32)
		else:
			raise Exception('Please assign datatype 16 or 32 bits')
		fout = open(new_id_to_doc_path, 'w')
		counter = 0
		i = 0
		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()


			corpus_emb, docid = sess.run(next_data)
			doc_emb = corpus_emb.reshape(-1, dim)
			doc_string_id = id_to_doc[docid[0]]
			p_num = 1
			while True:
				try:
					corpus_emb, docid = sess.run(next_data)
					corpus_emb = corpus_emb.reshape(-1, dim)


					if doc_string_id == id_to_doc[docid[0]]:
						# if p_num<p_max_num: #max passage number

						#     doc_emb+=corpus_emb
						#     p_num+=1
						if p_num<=p_max_num:
							corpus_embs[counter:(counter+1)] = doc_emb
							fout.write('{}\t{}\n'.format(counter, doc_string_id))
							counter += 1
							p_num += 1
							doc_emb = corpus_emb
					else:
						if p_num<=p_max_num:
							corpus_embs[counter:(counter+1)] = doc_emb
							fout.write('{}\t{}\n'.format(counter, doc_string_id))
							counter += 1
						doc_emb = corpus_emb
						doc_string_id = id_to_doc[docid[0]]
						p_num = 1

					docids.append(docid)
					pbar.update(10 * i + 1)
					i+=1

				except tf.errors.OutOfRangeError:
					break
		fout.close()
		return corpus_embs[:counter], docids


def normalize(embeddings):
	return (embeddings.T/LA.norm(embeddings,axis=-1)).T