import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
import os

RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_graph():
    ppi = pd.read_csv('ppi_network.csv')
    feats = pd.read_csv('gene_features.csv', index_col=0)
    sl = pd.read_csv('sl_labels.csv')
    gene_map = {gene: i for i, gene in enumerate(feats.index)}
    edge_index = torch.tensor([[gene_map[r['protein1']], gene_map[r['protein2']]] for _, r in ppi.iterrows()], dtype=torch.long).t().contiguous()
    x = torch.tensor(feats.values, dtype=torch.float)
    edge_label_index = torch.tensor([[gene_map[r['gene_a']], gene_map[r['gene_b']]] for _, r in sl.iterrows()], dtype=torch.long).t().contiguous()
    y = torch.tensor(sl['label'].values, dtype=torch.float)
    return Data(x=x, edge_index=edge_index), edge_label_index, y

class GraphLethalX(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, dropout=0.2)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x).squeeze(-1)

def analyze_results(explanation, data, model, edge_label_index, y_true):
    print("\n📊 Saving Research Findings to /results folder...")
    
    feat_imp = explanation.node_mask.flatten()
    if torch.is_tensor(feat_imp): feat_imp = feat_imp.numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(feat_imp)), feat_imp, color='teal')
    plt.title("Biological Feature Importance")
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
    plt.close() 

    model.train() 
    preds_logits = []
    for _ in range(20):
        with torch.no_grad():
            z = model(data.x, data.edge_index)
            logits = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2
            preds_logits.append(logits.numpy())
    preds_logits = np.array(preds_logits)
    mean_probs = 1 / (1 + np.exp(-preds_logits.mean(axis=0)))
    std_dev = preds_logits.std(axis=0) 
    plt.figure(figsize=(8, 5))
    plt.scatter(mean_probs, std_dev, alpha=0.4, color='purple')
    plt.title("Reliability Map: Probability vs. Uncertainty")
    plt.savefig(os.path.join(RESULTS_DIR, 'uncertainty_map.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(mean_probs[y_true == 0], color='blue', label='Non-Lethal', kde=True)
    sns.histplot(mean_probs[y_true == 1], color='red', label='Lethal', kde=True)
    plt.legend()
    plt.subplot(1, 2, 2)
    y_pred_bin = (mean_probs > 0.5).astype(int)
    sns.heatmap(confusion_matrix(y_true, y_pred_bin), annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_metrics.png'))
    plt.close()

    results_df = pd.DataFrame({'Prob': mean_probs, 'Uncertainty': std_dev, 'Label': y_true})
    results_df.to_csv(os.path.join(RESULTS_DIR, 'lethal_predictions.csv'), index=False)

# Main Execution
def main():
    data, edge_label_index, y = load_graph()
    model = GraphLethalX(data.num_node_features, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(51):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        logits = (z[edge_label_index[0]] + z[edge_label_index[1]]) / 2
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    print("Explaining...")
    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=100),
        explanation_type='model', node_mask_type='attributes',
        edge_mask_type='object', model_config=dict(mode='binary_classification', task_level='node', return_type='raw'),
    )
    explanation = explainer(data.x, data.edge_index, index=0)
    
    # Save Subgraph Image
    mask = explanation.edge_mask.numpy() if torch.is_tensor(explanation.edge_mask) else explanation.edge_mask
    indices = np.where(mask >= np.percentile(mask, 98))[0]
    G = nx.Graph()
    for i in indices:
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        G.add_edge(u, v)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    colors = ['red' if n == 0 else 'lightgreen' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='orange', width=2)
    plt.savefig(os.path.join(RESULTS_DIR, 'lethal_subgraph.png'))
    plt.close()

    analyze_results(explanation, data, model, edge_label_index, y.numpy())
    print(f"DONE! Check the '{RESULTS_DIR}' folder for all images and CSVs.")

if __name__ == "__main__":
    main()