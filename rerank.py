import argparse
import pickle
import glob
import os
import numpy as np
from util import load_tfrecords, write_result, read_id_dict
import time
from progressbar import *
from collections import defaultdict



def rank_index(Index, Score, topk):
	RankIndex=[]
	RankScore=[]
	indexes_sort_by_score=np.argsort(Score)[:,::-1]
	for i, sort_index in enumerate(indexes_sort_by_score):
		newindex = []
		newscore = []
		K=min(topk, len(Index[i]))
		for idx in sort_index[:K]:
			newindex.append(Index[i][idx])
			newscore.append(Score[i][idx])
		RankIndex.append(newindex)
		RankScore.append(newscore)

	return RankIndex, RankScore

def read_candidate(file, docid_to_idx):
	qid_to_candidates = defaultdict(list)
	with open(file, 'r') as f:
		if docid_to_idx==None: #if document id is the same as index id, no need for re-mapping
			for i, line in enumerate(f):
				qid, candidate, _, _ =line.strip().split('\t')
				qid_to_candidates[qid].append(int(candidate))
		else:
			for i, line in enumerate(f):
				qid, candidate, _, _ =line.strip().split('\t')
				qid_to_candidates[qid].append(docid_to_idx[candidate])
	return qid_to_candidates

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--result_file", type=str)
	parser.add_argument("--candidate_file", type=str, required=True)
	parser.add_argument("--docs_per_file", type=int, default=1000000)
	parser.add_argument("--doc_word_num", type=int, default=1)
	parser.add_argument("--num_files", type=int, default=10)
	parser.add_argument("--query_word_num", type=int, default=1)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--query_emb_path", type=str)
	parser.add_argument("--corpus_emb_path", type=str)
	parser.add_argument("--data_type", type=str)
	parser.add_argument("--id_to_doc_path", type=str, default=None)
	parser.add_argument("--id_to_query_path", type=str, default=None)
	args = parser.parse_args()

	query_embs, _=load_tfrecords([args.query_emb_path], data_num=800000, word_num=args.query_word_num, \
	                            data_type=args.data_type, dim=args.emb_dim)
	query_embs=query_embs.reshape((-1, args.query_word_num, args.emb_dim))


	corpus_files = []
	for i in range(args.num_files):
		print('Read path:'+args.corpus_emb_path+str(i)+'.tf')
		corpus_files.append(args.corpus_emb_path+str(i)+'.tf')

	print('Load %d tfrecord files...'%(args.num_files))
	corpus_embs, docids=load_tfrecords(corpus_files, data_num=args.docs_per_file, word_num=args.doc_word_num, \
	                                dim=args.emb_dim, data_type=args.data_type)
	corpus_embs = corpus_embs.reshape((-1,args.doc_word_num,args.emb_dim))

	# idx_to_qid={}
	# for idx, qid in enumerate(qid_to_qrel.keys()):
	#     idx_to_qid[idx] = qid
	idx_to_docid, docid_to_idx = read_id_dict(args.id_to_doc_path)
	idx_to_qid, _ = read_id_dict(args.id_to_query_path)
	qid_to_candidates = read_candidate(args.candidate_file, docid_to_idx)
	qidxs = idx_to_qid.keys()
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(qidxs)).start()


	print("Start Reranking...")
	Score=[]
	Index = []
	start_time = time.time()
	for i, qidx in enumerate(qidxs):
		qid = idx_to_qid[qidx]
		index = qid_to_candidates[qid]
		Index.append(index)
		relevence_corpus_embs=corpus_embs[index]
		query_emb = query_embs[i]
		relevence_corpus_embs = relevence_corpus_embs.reshape((-1,args.emb_dim))
		score = np.dot(query_emb, relevence_corpus_embs.transpose())
		score=score.reshape((args.query_word_num, -1, args.doc_word_num))
		score = score.transpose((1,0,2))
		score=(score.max(axis=-1)).sum(axis=-1) #for maxsim calculation
		score=np.pad(score, (0, args.topk-len(score)), 'constant', constant_values=(-1000))
		Score.append(score.tolist())
		pbar.update(10 * i + 1)

	pbar.finish()
	RankIndex, RankScore=rank_index(Index, Score, args.topk)
	time_per_query = (time.time() - start_time)/len(qidxs)
	print('Reranking {} queries ({:0.3f} s/query)'.format(len(qidxs), time_per_query))

	write_result(qidxs, RankIndex, RankScore, args.result_file, idx_to_qid, idx_to_docid, args.topk)

if __name__ == "__main__":
	main()