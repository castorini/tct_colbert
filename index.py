import argparse
import os
os.environ['OMP_NUM_THREADS'] = str(32)
import faiss
import numpy as np
from util import load_tfrecords
import mkl
mkl.set_num_threads(32)
parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--num_files", type=int, required=True)
parser.add_argument("--query_word_num", type=int, default=1)
parser.add_argument("--doc_word_num", type=int, default=1)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--topk", type=int, default=1000)
parser.add_argument("--docs_per_file", type=int, default=1000000)
parser.add_argument("--index_file", type=str, required=True)
parser.add_argument("--quantize", action='store_true')
parser.add_argument("--emb_path", type=str)
parser.add_argument("--data_type", type=str)
args = parser.parse_args()
def index(corpus_embs, save_path, quantize):

	dimension=corpus_embs.shape[1]
	cpu_index = faiss.IndexFlatL2(dimension)
	if quantize:
		ncentroids = 2000
		code_size = dimension//8
		cpu_index = faiss.IndexIVFPQ(cpu_index, dimension, ncentroids, code_size, 8)
		print("Train index...")
		cpu_index.train(corpus_embs)

	print("Indexing...")
	cpu_index.add(corpus_embs)
	faiss.write_index(cpu_index, save_path)


query_embs, qids=load_tfrecords([args.emb_path+'/query_emb/queries.dev00.tf'], \
								 data_num=6980, word_num=args.query_word_num, \
								 dim=args.emb_dim, data_type=args.data_type)
query_embs=query_embs.reshape((-1, args.emb_dim))
qids=np.concatenate(qids).tolist()
corpus_files=[]
for i in range(args.offset*args.num_files, args.offset*args.num_files+args.num_files):
	print('Read path:'+args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')
	corpus_files.append(args.emb_path+'/corpus_emb/'+'msmarco0'+str(i)+'.tf')


print('Load %d tfrecord files...'%(args.num_files))
corpus_embs, docids=load_tfrecords(corpus_files, \
								   data_num=args.docs_per_file, \
								   word_num=args.doc_word_num, \
								   dim=args.emb_dim, data_type=args.data_type)
corpus_embs=corpus_embs.reshape((-1, args.emb_dim))
index(corpus_embs, save_path = args.index_file, quantize=args.quantize)
print('finish')