# _*_ coding:utf-8 _*_
import numpy as np
from sklearn.manifold import TSNE
from kon.model.embedding.util.util_tool import read_node_label

import matplotlib.pyplot as plt

class evaluate_tools():
    def __init__(self,embeddings,label_path='../wiki/Wiki_labels.txt'):
        self.data=embeddings
        self.X,self.y=read_node_label(label_path)

            # =read_label(label_path)

    def plot_embeddings(self):
        emb_list = []
        for k in self.X:
            emb_list.append(self.data[k])
        emb_list = np.array(emb_list)
        print(emb_list)

        model = TSNE(n_components=2)
        node_pos = model.fit_transform(emb_list)


        color_idx = {}
        for i in range(len(self.X)):
            color_idx.setdefault(self.y[i][0], [])
            color_idx[self.y[i][0]].append(i)

        for c, idx in color_idx.items():
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
        plt.legend()
        plt.show()
