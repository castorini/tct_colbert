import argparse
import pickle
import glob
import os
import numpy as np
from util import load_tfrecords, read_pickle, write_result
from progressbar import *


parser = argparse.ArgumentParser()
parser.add_argument("--first_stage_path", type=str)
parser.add_argument("--result_file", type=str)
parser.add_argument("--docs_per_file", type=int, default=1000000)
parser.add_argument("--doc_word_num", type=int, default=1)
parser.add_argument("--num_files", type=int, default=10)
parser.add_argument("--query_word_num", type=int, default=1)
parser.add_argument("--topk", type=int, default=1000)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--emb_path", type=str)
parser.add_argument("--data_type", type=str)
parser.add_argument("--rerank", action='store_true')
args = parser.parse_args()

def rank_index(Index, Distance,topk):
	NewIndex=[]
	sort_indexs=np.argsort(Distance)[:,::-1]
	for i, sort_index in enumerate(sort_indexs):
		newindex = []
		K=min(topk, len(Index[i]))
		for idx in sort_index[:K]:
			newindex.append(Index[i][idx])
		NewIndex.append(newindex)

	return NewIndex

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

query_embs, qids=load_tfrecords([args.emb_path+'/query_emb/queries.dev00.tf'], \
								 data_num=6980, word_num=args.query_word_num, dim=args.emb_dim, \
								 data_type=args.data_type)
query_embs=query_embs.reshape((-1, args.query_word_num, args.emb_dim))
qids=np.concatenate(qids).tolist()
if args.rerank:
	corpus_files = []
	for i in range(args.num_files):
		print('Read path:'+args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')
		corpus_files.append(args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')
	print('Load %d tfrecord files...'%(args.num_files))
	corpus_embs, docids=load_tfrecords(corpus_files, data_num=args.docs_per_file, word_num=args.doc_word_num, dim=args.emb_dim,\
									   data_type=args.data_type)
	corpus_embs = corpus_embs.reshape((-1,args.doc_word_num,args.emb_dim))


Distance=None
Index=None
for filename in glob.glob(os.path.join(args.first_stage_path, '*.pickle')):
	D, I=read_pickle(filename)
	try:
		Distance = np.concatenate([Distance, D], axis=1)
		Index= np.concatenate([Index, I], axis=1)
	except:
		Distance=D
		Index=I

print("Dedupe...")
Index, max_index_num=dedupe_index(Index, Distance)
if args.rerank:
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(Index)).start()
	print("Rerank with Max sim computation...") # To do: Use GPU for rerank
	Score=[]
	for i, index in enumerate(Index):
		relevence_corpus_embs=corpus_embs[index]
		query_emb = query_embs[i]
		relevence_corpus_embs = relevence_corpus_embs.reshape((-1,args.emb_dim))
		score = np.dot(query_emb, relevence_corpus_embs.transpose())
		score=score.reshape((args.query_word_num, -1, args.doc_word_num))
		score = score.transpose((1,0,2))
		score=(score.max(axis=-1)).sum(axis=-1)
		score=np.pad(score, (0, max_index_num-len(score)), 'constant', constant_values=(-100))
		Score.append(score[np.newaxis,:])
		pbar.update(10 * i + 1)
	pbar.finish()
	Score = np.concatenate(Score)
	Index=rank_index(Index, Score, args.topk)

write_result(qids, Index, args.result_file, args.topk)
