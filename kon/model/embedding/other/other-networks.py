# _*_ coding:utf-8 _*_
import networkx as nx

#建立图
G=nx.Graph()

#添加节点
G.add_node(1)
G.add_nodes_from([2,3])

#一个图添加到另一个图中
# H=nx.path_graph(10)
# G.add_nodes_from(H)

#添加边
G.add_edge(1,2)
e=(2,3)
G.add_edge(*e)
# G.add_edges_from(H.edges)

import matplotlib.pyplot as plt
# G=nx.petersen_graph()
nx.draw(G,with_labels=True,font_weight='bold')
plt.show()

#有向图
DG=nx.DiGraph()
DG.add_weighted_edges_from([(1,2,0.5),(3,1,0.75)])
print(DG.out_degree(1,weight='weight'))

