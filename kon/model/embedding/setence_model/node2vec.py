# _*_ coding:utf-8 _*_
from kon.model.embedding.setence_model.walk_core_model import core_model
from kon.model.embedding.util.util_tool import read_graph
from kon.model.embedding.util.evaluate import evaluate_tools
from numpy import random

class node2vec(core_model):

    def __init__(self,Graph,per_vertex,walk_length,window_size,dimension_size,work,p,q):
        super().__init__(Graph,per_vertex,walk_length,window_size,dimension_size,work)
        self.p=p
        self.q=q

    def Learn_Feature(self):
        self.Preprocess_Modified_Weights(self.p,self.q)
        sentence_list=[]
        for num in range(self.walk_epoch):
            random.shuffle(self.all_nodes)
            for node in self.all_nodes:
                sentence=self.random_walk(node,is_edge_sampling=True)
                sentence_list.append(sentence)

        return sentence_list

    def transform(self):
        sentence_list=self.Learn_Feature()
        embeddings=self.embdding_train(sentence_list)

        return embeddings


if __name__=='__main__':
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    node_vec= node2vec(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=1,
        p=0.25,
        q=4
    )

    embeddings=node_vec.transform()
    eval_tool=evaluate_tools(embeddings)
    eval_tool.plot_embeddings()
