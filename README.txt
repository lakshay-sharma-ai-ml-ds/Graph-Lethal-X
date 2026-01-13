-------------------------------------------------------------------------------------------------- 
                                      Graph-Lethal X
             (Explainable Geometric Deep Learning for Synthetic Lethality Discovery)
--------------------------------------------------------------------------------------------------

Graph-Lethal X is a pipeline designed to discover synthetic lethal (SL) gene pairs by integrating 
CRISPR screens with Protein-Protein Interaction (PPI) networks using advanced Graph Attention 
Networks (GAT).

Key Features:
--------------
- Geometric Deep Learning : Uses dual-attention GAT layers to navigate complex biological topologies.
- Explainable AI (XAI) : Leverages GNNExplainer to identify the "Lethal Subgraph" behind every prediction.
- Bayesian Reliability : Quantifies prediction uncertainty using Monte Carlo Dropout to ensure biological trust.

Project Structure:
------------------
- data_prep.py : Simulates STRING (PPI) and CRISPR (SL) datasets.
- main.py : The core GNN engine, training loop, and interpretability suite.

Results Summary:
----------------
The model successfully reduces dense PPI "hairballs" into clear, actionable lethal pathways.
-  Target Node**: Gene 0
-  High-Confidence Partners**: Genes 76, 20, 43
-  Validation Metrics**: Includes ROC-AUC, PR-AUC, and Confusion Matrices.

Setup:
------
1. Install dependencies: `pip install torch torch-geometric pandas numpy networkx matplotlib seaborn scikit-learn`
2. Run `python data_prep.py`
3. Run `python main.py`