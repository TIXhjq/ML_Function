# _*_ coding:utf-8 _*_
from kon.model.embedding.setence_model.walk_core_model import core_model
from kon.model.embedding.util.util_tool import get_node_information
from numpy import random
import numpy as np
import math

class Line(core_model):

    def __init__(self,Graph,per_vertex,walk_length,window_size,dimension_size,work,log_dir,epoch,negative_ratio=0,order='second',batch_size=1024,times=1):
        super().__init__(Graph,per_vertex,walk_length,window_size,dimension_size,work)
        self.times=times
        self.epoch=epoch
        self.log_dir=log_dir
        self.batch_size=batch_size
        self.order=order
        self.negative_ratio=negative_ratio
        self.idx2node,self.node2idx=get_node_information(self.all_nodes)
        self.generate_smapling_table()

    def generate_edge_sampling_table(self):
        #边采样,防止论文中提到的权重差距大导致梯度爆炸
        numEdges = self.numEdges

        edges_sum = 0
        for edge in self.all_edges:
            edges_sum += self.G[edge[0]][edge[1]].get('weight', 1.0)

        #搜索每条边的权重
        all_probability = []
        for edge in self.all_edges:
            probability = self.G[edge[0]][edge[1]].get('weight', 1.0) * numEdges / edges_sum
            all_probability.append(probability)

        self.edge_prab, self.edge_alias = self.optimize_fun.generate_alias_table(all_probability)

    def generate_node_sampling_table(self,power=0.75):
        node_degree = np.zeros(self.numNodes)

        #顶点采样,减少顶点数量,经验值power=0.75,论文将pagerannk的重要性判断定成了顶点的出度,或者说出度代表了每个顶点的权重
        #计算每个顶点出度
        for edge in self.all_edges:
            node_degree[self.node2idx[edge[0]]] += self.G[edge[0]][edge[1]].get('weight', 1.0)

        #对每个出度进行power减少
        weights_sum=0
        for rank in range(self.numNodes):
            weights_sum+=math.pow(node_degree[rank],power)

        #计算每个顶点的概率
        all_probability=[]
        for rank in range(self.numNodes):
            probability=float(math.pow(node_degree[rank],power))/weights_sum
            all_probability.append(probability)

        self.node_prab, self.node_alias = self.optimize_fun.generate_alias_table(all_probability)

    #生成alias需要的table
    def generate_smapling_table(self,power=0.75):
        self.generate_node_sampling_table(power)
        self.generate_edge_sampling_table()

    def generator_postive_data(self,data_index,start_index,end_index,edges_index):
        rank_list=[]
        for rank in range(start_index,end_index):
            rank_list.append(data_index[rank])

        edge_index_list_ = self.optimize_fun.batch_alias_sample(
            prab=self.edge_prab,
            alias=self.edge_alias,
            rank_list=rank_list
        )

        begin_node,end_node=[],[]

        for edge_index_ in edge_index_list_:
            begin_node.append(edges_index[edge_index_][0])
            end_node.append(edges_index[edge_index_][1])

        return begin_node,end_node

    def generator_negative_data(self,begin_node):
        rank_list = []
        for i in range(len(begin_node)):
            rank_list.append(random.choice(list(range(len(self.node_prab)))))

        end_node = self.optimize_fun.batch_alias_sample(
            prab=self.node_prab,
            alias=self.node_alias,
            rank_list=rank_list
        )

        return begin_node,end_node

    def generator_data(self):
        #edges_index:(begin_node,end_node)
        edges_index = []
        for edge in self.all_edges:
            edge_index = (self.node2idx[edge[0]], self.node2idx[edge[1]])
            edges_index.append(edge_index)

        #data_index:index of (edge_index)
        data_size=self.numEdges
        data_index=list(range(data_size))
        random.shuffle(data_index)

        begin_node=[]
        start_index=0
        end_index=min(start_index+self.batch_size,data_size)

        #constrat negative number
        mod=0
        #num(generator negative data)
        mod_size=1+self.negative_ratio

        while True:
            if mod==0:
                begin_node,end_node=self.generator_postive_data(data_index,start_index,end_index,edges_index)
                sign=np.ones(len(begin_node))

            else:
                begin_node,end_node=self.generator_negative_data(begin_node)
                sign=np.ones(len(begin_node))*-1

            if self.order == 'all':
                yield ([np.array(begin_node), np.array(end_node)], [sign, sign])
            else:
                yield ([np.array(begin_node), np.array(end_node)], [sign])

            #控制负样本个数
            mod+=1
            mod%=mod_size

            if mod==0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index>=data_size:
                mod=0
                begin_node=[]
                random.shuffle(data_index)
                start_index=0
                end_index=min(start_index+self.batch_size,data_size)

    def train(self):
        model=self.creat_line_model()
        model.fit_generator(
            self.generator_data(),
            steps_per_epoch=((self.numEdges*(1+self.negative_ratio)-1)//self.batch_size+1)*self.times,
            verbose=1,
            epochs=self.epoch,
            callbacks=self.model_prepare(self.log_dir)
        )

    def get_embedding(self):
        self.embeddings={}
        if self.order=='first':
            embeddings=self.embedding_dict['first'].get_weights()[0]
        elif self.order=='second':
            embeddings=self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                        0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self.embeddings[idx2node[i]] = embedding

        return self.embeddings

    def transform(self):
        self.train()
        self.get_embedding()
        return self.embeddings


if __name__=='__main__':
    from util_tool import read_graph
    Graph=read_graph('model/embedding/wiki/Wiki_edgelist.txt')
    line=Line(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        negative_ratio=1,
        batch_size=128,
        log_dir='model/embedding/setence_model/logs/0/',
        epoch=100,
    )
    embeddings=line.transform()
    from evaluate import evaluate_tools
    tool=evaluate_tools(embeddings)
    tool.plot_embeddings()