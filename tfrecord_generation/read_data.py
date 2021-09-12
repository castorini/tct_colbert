import os
import glob
import json
class msmarco_corpus():
	def __init__(self, files, meta):
		self.meta = meta.split(',')
		self.files = files
		print('Total {} files'.format(len(self.files)))
		self.lines = []
		for file in self.files:
			with open(file, 'r') as f:
				self.lines+=f.readlines()
		self.num = len(self.lines)
	def output(self):
		for i, line in enumerate(self.lines):
			try: #tsv format id \t text
				docid, text = line.strip().split('\t')
			except: #json format
				info = json.loads(line.strip())
				docid = info['id']
				# if (self.meta[0]=='contents') and (len(self.meta)==1): #msmarcov1 doc json format
				# 	text = info['contents']
				# 	fields = text.split('\n')
				# 	title, text = fields[1], fields[2:]
				# 	if len(text) > 1:
				# 		text = ' '.join(text)
				# 		text = title + ' ' + text
				# else:
				text = []
				for meta in self.meta:
					text.append(info[meta])
				text = ' '.join(text)

			yield docid, text