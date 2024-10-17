# GRAIP: GRAphlet-based Incrementel generator for Probabilistic networks

This is the official Python implementation of GRAIP, as described in  

**Generating Random Graphs With Prescribed Graphlet Frequency Bounds Derived From Probabilistic Networks**  
by Bram Mornie, Didier Colle, Pieter Audenaert and Mario Pickavet.  
(Currently under review for publication.)  

### Requirements  
To generate graphs with GRAIP, you need the following dependencies: (listed versions were used during development and testing)  
- Python: 3.12  
- numpy: 1.26.4 (v2+ is fine for GRAIP, but pyemd does not work with it!)  
- matplotlib: 3.9.2  
- networkx: 3.3
  
Additionally, if you want to run `evaluate.py` and `test.py`, you also need:
- scipy: 1.14.1
- scikit-learn: 1.5.2
- pyemd: 1.0.0

### Installation  
**1. Clone the repository**  
```bash
git clone https://github.com/bmornie/GRAIP.git
cd GRAIP
```
**2. Install required dependencies**  
```bash
pip install -r requirements.txt
```

### Test run
The `test.py` script provides an example of what can be done using our code. It includes: sampling, graph generation and evaluation of the generated graphs.
```bash
python test.py
```

### Basic usage
You can quicly generate one or multiple graphs with GRAIP (default settings) by doing
```bash
python main.py graph_path S n_g graphs savepath
```
with:
- graph_path: the location of the probabilistic target graph (should be in gml format)
- S: the number of sample graphs constructed during sampling
- n_g: the maximum order of graphlets taken into account (should be 3, 4 or 5)
- graphs: the number of graphs to be generated
- savepath: location where generated graphs are stored. Graphs are saved in edge list format under "savepath/GRAIP_graph[n].txt" for n = 0, .., graphs-1
  
Example usage:
```bash
python main.py data/Haloferax_volcanii.gml 1000 4 10 new_graphs
```
