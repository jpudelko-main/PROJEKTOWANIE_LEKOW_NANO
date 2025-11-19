
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, Draw, Descriptors, MACCSkeys, rdFingerprintGenerator


substancje = {
    'NAZWA': ['CODEINE', 'ASPIRIN', 'MORPHINE', 'OXYCODONE', 'GEFITINIB'],
    'SMILES': [
        'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O', 
        'CC(=O)OC1=CC=CC=C1C(=O)O',                                     
        'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O', 
        'CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O',  
        'COC1=C2C=CC(Cl)=C1OCC2CN1CCN(CC1)C1=CC=C(CN(C)C)C=N1'           
    ]
}


df_substancje = pd.DataFrame(substancje)
PandasTools.AddMoleculeColumnToFrame(df_substancje, 'SMILES', 'Molecule')


df_substancje['MACCS'] = df_substancje['Molecule'].apply(MACCSkeys.GenMACCSKeys)


morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
df_substancje['Morgan'] = df_substancje['Molecule'].apply(morgan_generator.GetFingerprint)


query_fp_MACCS = df_substancje.loc[df_substancje['NAZWA']=='GEFITINIB', 'MACCS'].iloc[0]
codeine_fp_MACCS = df_substancje.loc[df_substancje['NAZWA']=='CODEINE', 'MACCS'].iloc[0]

tanimoto_MACCS = DataStructs.TanimotoSimilarity(query_fp_MACCS, codeine_fp_MACCS)
print(f"Tanimoto MACCS (Gefitinib vs Codeine): {tanimoto_MACCS:.3f}")

query_fp_Morgan = df_substancje.loc[df_substancje['NAZWA']=='GEFITINIB', 'Morgan'].iloc[0]
codeine_fp_Morgan = df_substancje.loc[df_substancje['NAZWA']=='CODEINE', 'Morgan'].iloc[0]

tanimoto_Morgan = DataStructs.TanimotoSimilarity(query_fp_Morgan, codeine_fp_Morgan)
print(f"Tanimoto Morgan (Gefitinib vs Codeine): {tanimoto_Morgan:.3f}")

df_substancje['Tanimoto_MACCS'] = DataStructs.BulkTanimotoSimilarity(query_fp_MACCS, df_substancje['MACCS'].tolist())
df_substancje['Tanimoto_Morgan'] = DataStructs.BulkTanimotoSimilarity(query_fp_Morgan, df_substancje['Morgan'].tolist())

print(df_substancje[['NAZWA', 'Tanimoto_MACCS', 'Tanimoto_Morgan']])

plt.scatter(df_substancje['Tanimoto_MACCS'], df_substancje['Tanimoto_Morgan'])
plt.xlabel('Tanimoto MACCS')
plt.ylabel('Tanimoto Morgan')
plt.title('MACCS vs Morgan (Tanimoto)')
for i, name in enumerate(df_substancje['NAZWA']):
    plt.text(df_substancje['Tanimoto_MACCS'][i], df_substancje['Tanimoto_Morgan'][i], name)
plt.show()

dice_list_MACCS = [DataStructs.DiceSimilarity(query_fp_MACCS, fp) for fp in df_substancje['MACCS']]
dice_list_Morgan = [DataStructs.DiceSimilarity(query_fp_Morgan, fp) for fp in df_substancje['Morgan']]

df_substancje['Dice_MACCS'] = dice_list_MACCS
df_substancje['Dice_Morgan'] = dice_list_Morgan

print(df_substancje[['NAZWA', 'Dice_MACCS', 'Dice_Morgan']])

df = pd.read_csv('final_filtered_bioactivities.csv')


print("Kolumny w df:", df.columns.tolist())


PandasTools.AddMoleculeColumnToFrame(df, 'canonical_smiles', 'Molecule')


query_smiles = df_substancje.loc[df_substancje['NAZWA']=='GEFITINIB', 'SMILES'].iloc[0]
query_mol = Chem.MolFromSmiles(query_smiles)

query_MACCS = MACCSkeys.GenMACCSKeys(query_mol)
query_Morgan = morgan_generator.GetFingerprint(query_mol)


df['MACCS'] = df['Molecule'].apply(MACCSkeys.GenMACCSKeys)
df['Morgan'] = df['Molecule'].apply(morgan_generator.GetFingerprint)


df['Tanimoto_MACCS'] = DataStructs.BulkTanimotoSimilarity(query_MACCS, df['MACCS'].tolist())
df['Tanimoto_Morgan'] = DataStructs.BulkTanimotoSimilarity(query_Morgan, df['Morgan'].tolist())

df['Dice_MACCS'] = [DataStructs.DiceSimilarity(query_MACCS, fp) for fp in df['MACCS']]
df['Dice_Morgan'] = [DataStructs.DiceSimilarity(query_Morgan, fp) for fp in df['Morgan']]


df.to_csv('fingerprints_dataset.csv', index=False)


top5 = df.sort_values(by='Tanimoto_Morgan', ascending=False).head(5)
print(top5[['canonical_smiles', 'Tanimoto_Morgan', 'pIC50']])

img = PandasTools.FrameToGridImage(top5, molsPerRow=5, molCol='ROMol', legendsCol='pIC50')
img.show()

