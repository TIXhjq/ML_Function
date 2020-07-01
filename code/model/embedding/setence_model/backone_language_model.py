# _*_ coding:utf-8 _*_
from gensim.models import Word2Vec

class language_model():
    def __init__(self,window_size,dimension_size,work):
        self.crop_size=int(window_size)
        self.unit_size=dimension_size
        self.workers=work

    def word2vec_on_train(self,sentence):
        model=Word2Vec(
            sentences=sentence,
            window=self.crop_size,
            size=self.unit_size,
            sg=1,
            hs=0,
            workers=self.workers,
            iter=3,
            min_count=0
        )

        return model


