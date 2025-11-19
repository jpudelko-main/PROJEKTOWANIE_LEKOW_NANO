# Importowanie potrzebnych bibliotek
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools

substancje = {
    'NAZWA': ['CODEINE', 'ASPIRIN', 'MORPHINE', 'OXYCODONE'],
    'SMILES': [
        'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O',
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O',
        'CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O'
    ]
}
df_substancje = pd.DataFrame(substancje)


def oblicz_wlasciwosci(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MW': Descriptors.MolWt(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'LogP': Descriptors.MolLogP(mol)
    }

wlasciwosci = df_substancje['SMILES'].apply(oblicz_wlasciwosci).apply(pd.Series)
df_substancje = pd.concat([df_substancje, wlasciwosci], axis=1)

progi = {'MW': 500, 'HBA': 10, 'HBD': 5, 'LogP': 5}

for prop in progi:
    plt.figure()
    plt.bar(df_substancje['NAZWA'], df_substancje[prop])
    plt.axhline(y=progi[prop], color='red', linestyle='--', label='Lipinski limit')
    plt.title(f'{prop}')
    plt.ylabel(prop)
    plt.legend()
    plt.show()

def czy_lipinski(smiles):
    props = oblicz_wlasciwosci(smiles)
    naruszenia = sum([
        props['MW'] > 500,
        props['HBA'] > 10,
        props['HBD'] > 5,
        props['LogP'] > 5
    ])
    return naruszenia <= 1

df_ex1 = pd.read_csv('egfr_bioactivities.csv')

df_ex1['Lipinski_pass'] = df_ex1['canonical_smiles'].apply(czy_lipinski)



def _define_radial_axes_angles(n_axes):
    x_angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]
    x_angles += x_angles[:1]
    return x_angles

lipinski_pass_df = df_ex1[df_ex1['Lipinski_pass']]
props_mean = lipinski_pass_df['canonical_smiles'].apply(oblicz_wlasciwosci).apply(pd.Series).mean()
props_std = lipinski_pass_df['canonical_smiles'].apply(oblicz_wlasciwosci).apply(pd.Series).std()

props_scaled = props_mean / pd.Series(progi) * 5
props_scaled_std_up = (props_mean + props_std) / pd.Series(progi) * 5
props_scaled_std_down = (props_mean - props_std) / pd.Series(progi) * 5

angles = _define_radial_axes_angles(len(progi))
labels = list(progi.keys())

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, [5]*len(labels) + [5], color='lightblue', alpha=0.4, label='Lipinski Ro5 region')
ax.plot(angles, list(props_scaled) + [props_scaled[0]], label='Mean', color='blue')
ax.plot(angles, list(props_scaled_std_up) + [props_scaled_std_up[0]], label='Mean + STD', linestyle='--', color='orange')
ax.plot(angles, list(props_scaled_std_down) + [props_scaled_std_down[0]], label='Mean - STD', linestyle='--', color='orange')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()