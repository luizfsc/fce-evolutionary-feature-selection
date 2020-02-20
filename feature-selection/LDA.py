# -*- coding: utf-8 -*-

import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction import DictVectorizer

class LDA(object):
     # Constructor
    def __init__(self, textos=[],features_list=[],numTopicos):
        self.corpus = textos
        self.numTopicos = numTopicos
        self.vectorizer = DictVectorizer(dtype=int, sparse=True)
        self.bow = features_list
        
    def executeLDA(self):
        data = self.vectorizer.fit_transform(self.bow)
        corpus_vect_gensim = gensim.matutils.Sparse2Corpus(data, documents_columns=False)
        
        dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                                    id2word=dict((id, word) for word, id in self.vectorizer.vocabulary_.items()))
        #print(dictionary.token2id)
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus_vect_gensim, id2word=dictionary, num_topics=self.numTopicos, update_every=1, chunksize=10000, passes=1) # ,minimum_probability=
               
        vec_gen = lda[corpus_vect_gensim]

        vecs = [vec for vec in vec_gen]
        
        #imprime distribuição de tópicos por documento do conjunto de treinamento
        for i, text in enumerate(self.corpus):
            print("==============================================")
            print(text["TEXT"]+" TÓPICO:" + ', '.join(map(str, vecs[i]))) # get topic probability distribution for a document
            print("==============================================")                    
        