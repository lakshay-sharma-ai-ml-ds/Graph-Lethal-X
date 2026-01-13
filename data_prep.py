import pandas as pd
import numpy as np

def generate_data():
    # 1. Simulate PPI Network (STRING-like)
    # Genes, interactions, and a confidence score
    genes = [f"GENE_{i}" for i in range(100)]
    ppi_data = {
        'protein1': np.random.choice(genes, 500),
        'protein2': np.random.choice(genes, 500),
        'combined_score': np.random.uniform(0.4, 0.9, 500)
    }
    ppi_df = pd.DataFrame(ppi_data)
    ppi_df.to_csv('ppi_network.csv', index=False)

    # 2. Simulate CRISPR Synthetic Lethality (Labels)
    # 1 = Synthetic Lethal, 0 = Non-lethal
    crispr_data = {
        'gene_a': np.random.choice(genes, 200),
        'gene_b': np.random.choice(genes, 200),
        'label': np.random.randint(0, 2, 200)
    }
    crispr_df = pd.DataFrame(crispr_data)
    crispr_df.to_csv('sl_labels.csv', index=False)

    # 3. Simulate Gene Features (Expression/Mutations)
    features = np.random.randn(100, 16) # 16-dimensional feature vector
    feat_df = pd.DataFrame(features, index=genes)
    feat_df.to_csv('gene_features.csv')

    print("✅ Data files generated: ppi_network.csv, sl_labels.csv, gene_features.csv")

if __name__ == "__main__":
    generate_data()