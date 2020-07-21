# _*_ coding:utf-8 _*_
from kon.model.embedding.setence_model.walk_core_model import core_model
from kon.model.embedding.util.util_tool import read_graph,get_node_information
import numpy as np

class sdne(core_model):

    def __init__(self, Graph, per_vertex, walk_length, window_size, dimension_size, work,alpha,beta,epochs,batch_size,verbose,hidden_size_list,l1,l2,log_dir):
        super().__init__(Graph, per_vertex, walk_length, window_size, dimension_size, work)
        self.alpha=alpha
        self.beta=beta
        self.batch_size=batch_size
        self.epochs=epochs
        self.verbose=verbose
        self.log_dir=log_dir
        self.pred_all_nodes=self.all_nodes
        self.idx2node, self.node2idx = get_node_information(self.pred_all_nodes)
        self.W,self.W_ = self.generator_adjacency_matrix(self.pred_all_nodes)
        self.L=self.generator_L(self.W_)
        self.model,self.embedding_model=self.creat_model(hidden_size_list=hidden_size_list,l1=l1,l2=l2)

    def generator_adjacency_matrix(self,all_nodes):
        numNodes=len(all_nodes)
        W=np.zeros((numNodes,numNodes))
        W_=np.zeros((numNodes,numNodes))

        for start_vertex in all_nodes:
            start_rank=self.node2idx[start_vertex]
            for end_vertex in list(self.G.neighbors(start_vertex)):
                end_rank=self.node2idx[end_vertex]
                weight=self.G[start_vertex][end_vertex].get('weight',1.0)
                W[start_rank][end_rank]=weight
                W_[start_rank][end_rank]=weight
                W_[end_rank][start_rank]=weight

        return W,W_

    def generator_L(self,W_):
        D = np.zeros_like(W_)

        for i in range(len(W_)):
            D[i][i] = np.sum(W_[i])
        L = D - W_

        return L

    def generator_data(self):
        all_nodes=self.pred_all_nodes
        start_rank=0
        end_rank=min(self.batch_size,self.numNodes)

        while True:
            batch_nodes=all_nodes[start_rank:end_rank]
            node_index_list=[self.node2idx[node] for node in batch_nodes]

            batch_W=self.W[node_index_list,:]
            batch_L=self.L[node_index_list][:,node_index_list]

            input_=[batch_W,batch_L]

            yield (input_,input_)

            start_rank = end_rank
            end_rank += self.batch_size
            end_rank = min(end_rank, self.numNodes)

            if end_rank==self.numNodes:
                start_rank=0
                end_rank=min(self.batch_size,self.numNodes)
                np.random.shuffle(all_nodes)

    def train(self):
        self.model.compile('adam',[self.second_nd(self.beta),self.first_nd(self.alpha)])
        self.model.fit_generator(
            self.generator_data(),
            steps_per_epoch=self.numNodes//self.batch_size,
            epochs=self.epochs,
            callbacks=self.model_prepare(self.log_dir),
            verbose=self.verbose
        )
        return self.model

    def get_embeddings(self):
        embeddings={}
        pred_embeddings=self.embedding_model.predict(self.W,batch_size=self.batch_size)

        rank=0
        for embedding in pred_embeddings:
            embeddings[self.idx2node[rank]]=embedding
            rank+=1
        return embeddings

if __name__=='__main__':
    Graph=read_graph()
    sden_model=sdne(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        beta=5,
        alpha=1e-6,
        verbose=1,
        epochs=1000,
        batch_size=512,
        log_dir='model/embedding/setence_model/logs/0/',
        hidden_size_list=[256, 128],
        l1=1e-5,
        l2=1e-4
    )

    sden_model.train()
    embeddings=sden_model.get_embeddings()

    from kon.model import evaluate_tools
    eval_tool=evaluate_tools(embeddings)
    eval_tool.plot_embeddings()


