import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from IPython.display import display


bioactivities_df = pd.read_csv('egfr_bioactivities.csv')
unwanted_substructures_df = pd.read_csv('unwanted_substructures.csv')


pains_catalog = FilterCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

def is_pains(mol):
    """Sprawdza, czy dany związek pasuje do filtru PAINS."""
    return pains_catalog.HasMatch(mol)


PandasTools.AddMoleculeColumnToFrame(bioactivities_df, smilesCol='canonical_smiles', molCol='ROMol')


bioactivities_df['is_pains'] = bioactivities_df['ROMol'].apply(lambda mol: is_pains(mol) if mol else False)
bioactivities_df_filtered = bioactivities_df[~bioactivities_df['is_pains']].copy()


unwanted_smarts = unwanted_substructures_df['smarts'].tolist()
unwanted_mols = [Chem.MolFromSmarts(smarts) for smarts in unwanted_smarts if Chem.MolFromSmarts(smarts)]

def has_unwanted_substructure(mol):
    """Sprawdza, czy dany związek zawiera niechcianą podstrukturę."""
    return any(mol.HasSubstructMatch(sub_mol) for sub_mol in unwanted_mols)


bioactivities_df_filtered.loc[:, 'has_unwanted_substructure'] = bioactivities_df_filtered['ROMol'].apply(
    lambda mol: has_unwanted_substructure(mol) if mol else False)

final_filtered_df = bioactivities_df_filtered[~bioactivities_df_filtered['has_unwanted_substructure']]
final_filtered_df.to_csv('final_filtered_bioactivities.csv', index=False)

pains_mols = bioactivities_df[bioactivities_df['is_pains']]['ROMol'].dropna().tolist()[:3]
unwanted_mols = bioactivities_df_filtered[bioactivities_df_filtered['has_unwanted_substructure']]['ROMol'].dropna().tolist()[:3]

if pains_mols:
    draw_pains = Draw.MolsToGridImage(pains_mols, molsPerRow=3, subImgSize=(200, 200))
    display(draw_pains)
else:
    print("Brak związków PAINS do wizualizacji.")

if unwanted_mols:
    draw_unwanted = Draw.MolsToGridImage(unwanted_mols, molsPerRow=3, subImgSize=(200, 200))
    display(draw_unwanted)
else:
    print("Brak związków z niechcianymi podstrukturami do wizualizacji.")


from collections import Counter

all_substructures = []
for mol in final_filtered_df['ROMol'].dropna():
    for atom in mol.GetAtoms():
        fragment = Chem.MolToSmarts(mol)
        all_substructures.append(fragment)

most_common_substructure = Counter(all_substructures).most_common(1)[0]

# Wyniki
print("Najczęstsza podstruktura:", most_common_substructure)
