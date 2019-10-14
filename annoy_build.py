# -*- coding: UTF-8 -*-
from annoy import AnnoyIndex
import pickle as pkl

ann_save_path = "./annoy"
id2word_path = "./id2word"
word2id_path = "./word2id"
id2word = None
word2id = None
annoy_index = None

# 构建索引，只需构建一次
def build_index(tree_num = 100):
	global id2word, word2id

	# 自定义的读取word2vec的函数
	items_vec = load_gensim()
	# 向量维度为200
	a = AnnoyIndex(200)
	i = 0
	id2word, word2id = dict(), dict()
	for word in items_vec.vocab:
		a.add_item(i, items_vec[word])
		id2word[i] = word
		word2id[word] = i
		i += 1
	a.build(tree_num)
	a.save(ann_save_path)
	pkl.dump(id2word, open(id2word_path, "wb"))
	pkl.dump(word2id, open(word2id_path, "wb"))

# 实际运行时加载索引
def annoy_init():
	global id2word, word2id, annoy_index
	id2word = pkl.load(open(id2word_path, "rb"))
	word2id = pkl.load(open(word2id_path, "rb"))
	annoy_index = AnnoyIndex(200)
	annoy_index.load(ann_save_path)

# 近似检索，query为编码后的向量
def annoy_search(query, topn=10):
	global id2word, word2id, annoy_index
	idxes, dists = annoy_index.get_nns_by_vector(query, topn, include_distances=True)
	idxes = [id2word[i] for i in idxes]
	similars = list(zip(idxes, dists))
	return similars

if __name__ == '__main__':
	build_index()