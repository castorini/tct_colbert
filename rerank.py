import argparse
import pickle
import glob
import os
import numpy as np
from util import load_tfrecords, read_candidate, write_result
import time
from progressbar import *


parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str)
parser.add_argument("--candidate_file", type=str, required=True)
parser.add_argument("--docs_per_file", type=int, default=1000000)
parser.add_argument("--doc_word_num", type=int, default=1)
parser.add_argument("--num_files", type=int, default=10)
parser.add_argument("--query_word_num", type=int, default=1)
parser.add_argument("--topk", type=int, default=1000)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--emb_path", type=str)
parser.add_argument("--data_type", type=str)
args = parser.parse_args()

def rank_index(Index, Distance, topk):
	NewIndex=[]
	sort_indexs=np.argsort(Distance)[:,::-1]
	for i, sort_index in enumerate(sort_indexs):
		newindex = []
		K=min(topk, len(Index[i]))
		for idx in sort_index[:K]:
			newindex.append(Index[i][idx])
		NewIndex.append(newindex)

	return NewIndex





query_embs, qids=load_tfrecords([args.emb_path+'/query_emb/queries.dev00.tf'], data_num=6980, word_num=args.query_word_num, \
								data_type=args.data_type, dim=args.emb_dim)
query_embs=query_embs.reshape((-1, args.query_word_num, args.emb_dim))
qids=np.concatenate(qids).tolist()


corpus_files = []
for i in range(args.num_files):
	print('Read path:'+args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')
	corpus_files.append(args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')

print('Load %d tfrecord files...'%(args.num_files))
corpus_embs, docids=load_tfrecords(corpus_files, data_num=args.docs_per_file, word_num=args.doc_word_num, \
								   dim=args.emb_dim, data_type=args.data_type)
corpus_embs = corpus_embs.reshape((-1,args.doc_word_num,args.emb_dim))

qid_to_qrel = read_candidate(args.candidate_file)


widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=10*len(qids)).start()

print("Max sim computation...")
Score=[]
Index = []
start_time = time.time()
for i, qid in enumerate(qids):
	index = qid_to_qrel[qid]
	Index.append(index)
	relevence_corpus_embs=corpus_embs[index]
	query_emb = query_embs[i]
	relevence_corpus_embs = relevence_corpus_embs.reshape((-1,args.emb_dim))
	score = np.dot(query_emb, relevence_corpus_embs.transpose())
	score=score.reshape((args.query_word_num, -1, args.doc_word_num))
	score = score.transpose((1,0,2))
	score=(score.max(axis=-1)).sum(axis=-1)
	score=np.pad(score, (0, args.topk-len(score)), 'constant', constant_values=(-100))
	Score.append(score[np.newaxis,:])
	pbar.update(10 * i + 1)

pbar.finish()
Score = np.concatenate(Score)
print("Ranking...")
Index=rank_index(Index, Score, args.topk)
time_per_query = (time.time() - start_time)/len(qids)
print('Retrieving {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))

write_result(qids, Index, args.result_file)
