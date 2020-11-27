import argparse
import os
# os.environ['OMP_NUM_THREADS'] = str(32)
import faiss
import numpy as np
import math
from util import load_tfrecords, load_tfrecords_doc, read_id_dict
# import mkl
# mkl.set_num_threads(32)

def index(corpus_embs, save_path, quantize):

	dimension=corpus_embs.shape[1]
	cpu_index = faiss.IndexFlatIP(dimension)
	if quantize: # still try better way for balanced efficiency and effectiveness
		# ncentroids = 1000
		# code_size = dimension//4
		# cpu_index = faiss.IndexIVFPQ(cpu_index, dimension, ncentroids, code_size, 8)
		# cpu_index = faiss.IndexPQ(dimension, code_size, 8)
		cpu_index = faiss.index_factory(768, "OPQ128,IVF4096,PQ128", faiss.METRIC_INNER_PRODUCT)
		# cpu_index = faiss.GpuIndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_16bit_direct, faiss.METRIC_INNER_PRODUCT)
		print("Train index...")
		cpu_index.train(corpus_embs)


	print("Indexing...")
	cpu_index.add(corpus_embs)
	faiss.write_index(cpu_index, save_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--offset", type=int, default=0, help='in case memory is not enough to load all corpus for index')
	parser.add_argument("--num_files", type=int, required=True, help='set number of tf record files included in index')
	parser.add_argument("--query_word_num", type=int, default=1, help='in case when using token embedding maxsim search instead of pooling embedding')
	parser.add_argument("--doc_word_num", type=int, default=1, help='in case when using token embedding maxsim search instead of pooling embedding')
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--passages_per_file", type=int, default=1000000, help='our default tf record include 1000,000 passages per file')
	parser.add_argument("--index_file", type=str, required=True)
	parser.add_argument("--quantize", action='store_true')
	parser.add_argument("--corpus_emb_path", type=str)
	parser.add_argument("--data_type", type=str, help='16 or 32 bit')
	parser.add_argument("--corpus_type", type=str, default='passage', help='passage or doc')
	parser.add_argument("--id_to_doc_path", type=str, default=None, help='mapping file for id to docid, required when corpus type is doc')
	parser.add_argument("--max_passage_each_doc", type=int, default=1,help='set max chunk for each document as corpus index')
	parser.add_argument("--max_passage_each_index", type=int, default=2000000)
	args = parser.parse_args()

	if args.corpus_type=='doc' and args.id_to_doc_path==None:
		raise Exception('Missing aurguemnt for --id_to_doc_path')
	corpus_files=[]
	for i in range(args.offset*args.num_files, args.offset*args.num_files+args.num_files):
		file = args.corpus_emb_path+str(i)+'.tf'
		print('Read path:'+file)
		corpus_files.append(file)



	print('Load %d tfrecord files...'%(args.num_files))
	if args.corpus_type =='passage':
		corpus_embs, docids=load_tfrecords(corpus_files, \
										   data_num=args.passages_per_file, \
										   word_num=args.doc_word_num, \
										   dim=args.emb_dim, data_type=args.data_type, index=True)
	elif args.corpus_type =='doc':
		id_to_doc, _ = read_id_dict(args.id_to_doc_path)
		corpus_embs, docids=load_tfrecords_doc(corpus_files, \
										   data_num=args.passages_per_file, \
										   word_num=args.doc_word_num, \
										   id_to_doc=id_to_doc, \
										   new_id_to_doc_path='/'.join(args.corpus_emb_path.split('/')[:-1] + ['doc.id']), \
										   dim=args.emb_dim, p_max_num=args.max_passage_each_doc, \
										   data_type=args.data_type, index=True)

	corpus_embs=corpus_embs.reshape((-1, args.emb_dim))
	passage_num = corpus_embs.shape[0]
	index_file_num = int(math.ceil((passage_num)/float(args.max_passage_each_index)))


	for i in range(index_file_num):
		index(corpus_embs[(i*args.max_passage_each_index):((i+1)*args.max_passage_each_index),:], save_path = args.index_file + '_' + str(i), \
			  quantize=args.quantize)
		print('index file:'+str(i))
	print('finish')

if __name__ == "__main__":
	main()