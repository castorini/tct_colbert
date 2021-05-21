import argparse
import os
import faiss
import numpy as np
from util import load_tfrecords
import pickle
# import mkl
import time
from progressbar import *


def save_pickle(Distance, Index, filename):
	print('save pickle...')
	with open(filename, 'wb') as f:
		pickle.dump([Distance, Index], f)


def search(query_embs, index_file, batch_size, topk, GPU, gpu_device):
	res = faiss.StandardGpuResources()
	res.noTempMemory()
	# res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = gpu_device
	Distance = []
	Index = []
	batch = batch_size
	print("Read index...")
	cpu_index = faiss.read_index(index_file)
	if GPU:
		print("Load index ("+index_file+ ") to GPU...")
		co = faiss.GpuClonerOptions()
		# co.useFloat16 = True
		gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, cpu_index, co)
		# gpu_index.nprobe = 100
	else:
		cpu_index.nprobe = 1000
	print("Search with batch size %d"%(batch))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=query_embs.shape[0]//batch).start()
	start_time = time.time()

	for i in range(query_embs.shape[0]//batch):
		if GPU:
			D,I=gpu_index.search(query_embs[i*batch:(i+1)*batch], topk)
		else:
			D,I=cpu_index.search(query_embs[i*batch:(i+1)*batch], topk)

		Distance.append(D)
		Index.append(I)
		pbar.update(i + 1)
	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))
	Distance = np.concatenate(Distance, axis=0)
	Index = np.concatenate(Index, axis=0)
	return Distance, Index

def query_reformulation_and_search(query_embs, index_file, topk, qid_to_labels, GPU=True):
	res = faiss.StandardGpuResources()
	res.noTempMemory()
	res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	Distance = []
	Index = []

	print("Read index...")
	cpu_index = faiss.read_index(index_file)
	if GPU:
		print("Load index to GPU...")
		co = faiss.GpuClonerOptions()
		# co.useFloat16 = True
		gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
		gpu_index.nprobe = 10
	# else:
	# 	cpu_index.nprobe = 10
	#print("Search with batch size %d"%(batch))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=query_embs.shape[0]).start()
	start_time = time.time()

	for i, (qid, labels) in enumerate(qid_to_labels):
		turn_id = int(qid.split('_')[1])
		query_emb = query_embs[i:(i+1),4:,:] #1, 32, 768
		if turn_id==1:
			concat_query_emb = query_emb
			hist_query_embs = []
		else:
			concat_query_emb = np.concatenate([query_emb]+hist_query_embs, axis=-2) #1, 32*L, 768
			true_id = []
			for idx, label in enumerate(labels):
				if label==1:
					true_id.append(idx)
			concat_query_emb = concat_query_emb[:, true_id, :]

		query_pooling_emb = concat_query_emb.max(axis=-2) + concat_query_emb.min(axis=-2)
		query_pooling_emb = normalize(query_pooling_emb)
		hist_query_embs.append(query_emb)
		if GPU:
			D,I=gpu_index.search(query_pooling_emb, topk)
		else:
			D,I=cpu_index.search(query_pooling_emb, topk)
		Distance.append(D)
		Index.append(I)
		pbar.update(i + 1)

	# for i in range(query_embs.shape[0]//batch):
	# 	if GPU:
	# 		D,I=gpu_index.search(query_embs[i*batch:(i+1)*batch], topk)
	# 	else:
	# 		D,I=cpu_index.search(query_embs[i*batch:(i+1)*batch], topk)
	# 	Distance.append(D)
	# 	Index.append(I)
	# 	pbar.update(i + 1)
	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))
	Distance = np.concatenate(Distance, axis=0)
	Index = np.concatenate(Index, axis=0)
	return Distance, Index


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--offset", type=int, default=0)
	parser.add_argument("--query_word_num", type=int, default=1)
	parser.add_argument("--doc_word_num", type=int, default=1)
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--batch_size", type=int, default=1, help='in order to measure time/query, we have to set batch size to query_word_num')
	parser.add_argument("--passage_per_index", type=int, default=1000000)
	parser.add_argument("--pickle_file", type=str, required=True)
	parser.add_argument("--index_file", type=str, required=True)
	parser.add_argument("--query_emb_path", type=str)
	parser.add_argument("--data_type", type=str)
	parser.add_argument("--use_gpu", action='store_true')
	parser.add_argument("--gpu_device", type=int, default=0)
	args = parser.parse_args()

	query_embs, qids=load_tfrecords([args.query_emb_path],\
									data_num=800000, word_num=args.query_word_num, \
									dim=args.emb_dim, data_type=args.data_type)

	query_embs=query_embs.reshape((-1, args.query_word_num, args.emb_dim))
	query_embs=query_embs.astype(np.float32)
	query_embs = query_embs.mean(axis=-2)

	Distance, Index = search(query_embs, index_file=args.index_file, batch_size=args.query_word_num*args.batch_size, topk=args.topk, GPU=args.use_gpu, gpu_device=args.gpu_device)
	Index=Index//args.doc_word_num
	Index+=args.offset*args.passage_per_index
	Index = Index.reshape((-1, args.query_word_num*args.topk))
	Distance = Distance.reshape((-1, args.query_word_num*args.topk))


	save_pickle(Distance, Index, args.pickle_file)
	print('finish')

if __name__ == "__main__":
	main()