# GRAIP: GRAphlet-based Incrementel generator for Probabilistic networks

This is the official Python implementation of GRAIP, as described in  

**Generating Random Graphs With Prescribed Graphlet Frequency Bounds Derived From Probabilistic Networks**  
by Bram Mornie, Didier Colle, Pieter Audenaert and Mario Pickavet.  
(Currently under review for publication.)  

### Requirements  
To generate graphs with GRAIP, you need the following dependencies: (listed versions were used during development and testing)  
- Python: 3.12  
- numpy: 2.1.2
- matplotlib: 3.9.2  
- networkx: 3.3
- PyYAML: 6.0.2
  
Additionally, if you want to run `evaluate.py` and `test.py`, you also need:
- scikit-learn: 1.5.2

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
To generate graphs with GRAIP (or other models, see below) specify the parameters in `config.yaml` and run
```bash
python main.py
```
This only generates the graphs, 
