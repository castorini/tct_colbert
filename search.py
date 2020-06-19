import argparse
import os
import faiss
import numpy as np
from util import load_tfrecords, dedupe_index
import pickle
import mkl
import time
from progressbar import *


parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--num_files", type=int, required=True)
parser.add_argument("--query_word_num", type=int, default=1)
parser.add_argument("--doc_word_num", type=int, default=1)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--topk", type=int, default=1000)
parser.add_argument("--docs_per_file", type=int, default=1000000)
parser.add_argument("--pickle_file", type=str, required=True)
parser.add_argument("--index_file", type=str, required=True)
parser.add_argument("--emb_path", type=str)
parser.add_argument("--data_type", type=str)


args = parser.parse_args()

def save_pickle(Distance, Index, filename):
	print('save pickle...')
	with open(filename, 'wb') as f:
		pickle.dump([Distance, Index], f)


def search(query_embs, index_file, batch_size, topk=1000, GPU=True):
	res = faiss.StandardGpuResources()
	res.noTempMemory()
	res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	Distance = []
	Index = []
	batch = batch_size
	print("Read index...")
	if GPU:
		index = faiss.read_index(index_file)
		co = faiss.GpuClonerOptions()
		index = faiss.index_cpu_to_gpu(res, 0, index, co)
	index.nprobe = 10
	print("Search with batch size %d"%(batch))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=query_embs.shape[0]).start()
	start_time = time.time()
	for i in range(query_embs.shape[0]//batch):
		D,I=index.search(query_embs[i*batch:(i+1)*batch], topk)
		Distance.append(D)
		Index.append(I)
		pbar.update(i + 1)
	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))
	Distance = np.concatenate(Distance, axis=0)
	Index = np.concatenate(Index, axis=0)
	index.reset()
	return -Distance, Index

query_embs, qids=load_tfrecords([args.emb_path+'/query_emb/queries.dev00.tf'],\
								data_num=6980, word_num=args.query_word_num, \
								dim=args.emb_dim, data_type=args.data_type)

query_embs=query_embs.reshape((-1, args.emb_dim))
qids=np.concatenate(qids).tolist()
Distance, Index = search(query_embs, index_file=args.index_file, batch_size=args.query_word_num, topk=args.topk)

Index=Index//args.doc_word_num
Index+=args.offset*args.docs_per_file*args.num_files


Index = Index.reshape((-1, args.query_word_num*args.topk))
Distance = Distance.reshape((-1, args.query_word_num*args.topk))
save_pickle(Distance, Index, args.pickle_file)
print('finish')