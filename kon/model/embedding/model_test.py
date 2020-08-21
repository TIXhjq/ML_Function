# _*_ coding:utf-8 _*_
from kon.model.embedding.setence_model.deepwalk import DeepWalk
from kon.model.embedding.setence_model.line import Line
from kon.model.embedding.setence_model.node2vec import node2vec
from kon.model.embedding.setence_model.sdne import sdne
from kon.model.embedding.util.util_tool import read_graph
from kon.model.embedding.util.evaluate import evaluate_tools

def deep_walk_run(edgelist_path,is_evluate=False):
    Graph = read_graph(edgelist_path)

    deepwalk = DeepWalk(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=8,
        work=4
    )
    embeddings = deepwalk.transform()
    if is_evluate:
        eval = evaluate_tools(embeddings=embeddings, label_path='wiki/Wiki_labels.txt')
        eval.plot_embeddings()
    return embeddings

def line_run():
    from kon.model.embedding.util.util_tool import read_graph
    import os
    print(os.getcwd())
    Graph = read_graph('wiki/Wiki_edgelist.txt')
    line = Line(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        negative_ratio=1,
        batch_size=128,
        log_dir='logs/0/',
        epoch=100,
    )
    embeddings = line.transform()
    from kon.model.embedding.util.evaluate import evaluate_tools
    tool = evaluate_tools(embeddings, label_path='wiki/Wiki_labels.txt')
    tool.plot_embeddings()

def node2vec_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    node_vec = node2vec(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=1,
        p=0.25,
        q=4
    )

    embeddings = node_vec.transform()
    eval_tool = evaluate_tools(embeddings, label_path='wiki/Wiki_labels.txt')
    eval_tool.plot_embeddings()

def sdne_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')
    sden_model = sdne(
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
        log_dir='logs/0/',
        hidden_size_list=[256, 128],
        l1=1e-5,
        l2=1e-4
    )
    sden_model.train()
    embeddings = sden_model.get_embeddings()

    from kon.model.embedding.util.evaluate import evaluate_tools
    eval_tool = evaluate_tools(embeddings, label_path='wiki/Wiki_labels.txt')
    eval_tool.plot_embeddings()


def model_test(build_name,edgelist_path='wiki/Wiki_edgelist.txt',embedding=8):
    if build_name=='deepwalk':
        embedding=deep_walk_run(edgelist_path)
    elif build_name=='line':
        line_run()
    elif build_name=='node2vec':
        node2vec_run()
    elif build_name=='sdne':
        sdne_run()
    elif build_name=='all':
        deep_walk_run(edgelist_path)
        line_run()
        node2vec_run()
        sdne_run()

    return embedding

if __name__=='__main__':
    model_test('deepwalk')


