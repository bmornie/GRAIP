import sys
import os
import networkx as nx
from generator.generator import GraphGenerator


if __name__ == '__main__':
    graph_path = sys.argv[1]
    S = int(sys.argv[2])
    n_g = int(sys.argv[3])
    graphs = int(sys.argv[4])
    savepath = sys.argv[5]
    
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    G = nx.read_gml(graph_path, label=None)
    
    gen = GraphGenerator(G=G, max_gl_size=n_g)
    gen.get_properties(N=S)
    gen.generate(model='GRAIP', graphs=graphs, savepath=savepath)