
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


bioactivities_df = pd.read_csv('final_filtered_bioactivities.csv')
fingerprints_df = pd.read_csv('fingerprints_dataset.csv')

fingerprints = [DataStructs.CreateFromBitString(fp) for fp in fingerprints_df['Morgan']]


def tanimoto_distance_matrix(fps):
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
    return dists

distance_matrix = tanimoto_distance_matrix(fingerprints)


cutoff = 0.2
clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)


singletons = [cluster for cluster in clusters if len(cluster) == 1]
print(f'Liczba klastrów jednoelementowych (singletonów): {len(singletons)}')


cluster_sizes = [len(cluster) for cluster in clusters]
plt.bar(range(len(cluster_sizes)), cluster_sizes)
plt.xlabel('Indeks klastra')
plt.ylabel('Liczba związków')
plt.title('Rozmiary klastrów')
plt.show()


largest_cluster_idx = np.argmax(cluster_sizes)
largest_cluster_index = cluster_sizes.index(max(cluster_sizes))
largest_cluster_index = cluster_sizes.index(max(cluster_sizes))
selected_indices = list(clusters[largest_cluster_index][:10])
selected_molecules = bioactivities_df.iloc[selected_indices]

mols = [Chem.MolFromSmiles(smiles) for smiles in selected_molecules['canonical_smiles']]
img = Draw.MolsToGridImage(mols, molsPerRow=5)
img.show()


selected_molecules['ROMol'] = mols
selected_molecules.to_csv('selected_cluster_molecules.csv', index=False)

print('Wybrane struktury zapisano jako selected_cluster_molecules.sdf')
