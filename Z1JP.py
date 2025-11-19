

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client

targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity

uniprot_id = "P00533"


target = targets_api.get(target_components__accession=uniprot_id).only(
    "target_chembl_id", "organism", "pref_name", "target_type"
)


chembl_id = target[0]["target_chembl_id"]


bioactivities = bioactivities_api.filter(
    target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B"
).only(
    "molecule_chembl_id", "standard_value", "standard_units", "standard_type"
)

df_bioactivities = pd.DataFrame(bioactivities)


df_bioactivities = df_bioactivities.dropna(subset=["standard_value"])
df_bioactivities = df_bioactivities[df_bioactivities['standard_units'] == 'nM']
df_bioactivities = df_bioactivities.drop_duplicates("molecule_chembl_id")
df_bioactivities.rename(columns={"standard_value": "IC50"}, inplace=True)


compound_ids = df_bioactivities['molecule_chembl_id'].tolist()
compounds = compounds_api.filter(molecule_chembl_id__in=compound_ids).only(
    "molecule_chembl_id", "molecule_structures"
)

df_compounds = pd.DataFrame(compounds)

df_compounds = df_compounds.dropna(subset=['molecule_structures'])
df_compounds['canonical_smiles'] = df_compounds['molecule_structures'].apply(lambda x: x['canonical_smiles'])
df_compounds = df_compounds[['molecule_chembl_id', 'canonical_smiles']]
df_compounds = df_compounds.drop_duplicates("molecule_chembl_id")


df_merged = pd.merge(df_bioactivities, df_compounds, on="molecule_chembl_id")
df_merged["IC50"] = pd.to_numeric(df_merged["IC50"], errors='coerce')
df_merged.dropna(subset=['IC50'], inplace=True)

df_merged["pIC50"] = df_merged["IC50"].apply(lambda x: -math.log10(x * 1e-9))


plt.hist(df_merged["pIC50"], bins=30, edgecolor='black')
plt.xlabel("pIC50")
plt.ylabel("Frequency")
plt.title("Histogram pIC50 dla EGFR")
plt.show()


print(df_merged.nlargest(5, 'pIC50'))


df_merged.to_csv("egfr_bioactivities.csv", index=False)


print(df_merged.head())
