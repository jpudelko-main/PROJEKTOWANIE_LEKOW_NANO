# 1. Import bibliotek
import random
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import AllChem, Draw, rdFMCS, PandasTools
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D


molecules_df = pd.read_csv("selected_cluster_molecules.csv")
molecules = [Chem.MolFromSmiles(smi) for smi in molecules_df['canonical_smiles']]

mcs_result = rdFMCS.FindMCS(molecules)
mcs_smarts = mcs_result.smartsString
print(f"MCS SMARTS: {mcs_smarts}")
print(f"Liczba atomów: {mcs_result.numAtoms}")
print(f"Liczba wiązań: {mcs_result.numBonds}")

mcs_molecule = Chem.MolFromSmarts(mcs_smarts)
Draw.MolToFile(mcs_molecule, "mcs_structure.svg")
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D

def highlight_molecules(molecules, mcs, number, label=True, same_orientation=True, **kwargs):
    molecules = deepcopy(molecules[:number])
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    matching = [mol.GetSubstructMatch(pattern) for mol in molecules]

    legends = None
    if label:
        legends = [
            mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{i+1}"
            for i, mol in enumerate(molecules)
        ]

    if same_orientation and matching[0]:
        mol, match = molecules[0], matching[0]
        AllChem.Compute2DCoords(mol)
        coords = [mol.GetConformer().GetAtomPosition(x) for x in match]
        coords2D = [Geometry.Point2D(pt.x, pt.y) for pt in coords]
        for mol, match in zip(molecules[1:], matching[1:]):
            if not match:
                continue
            coord_dict = {match[i]: coord for i, coord in enumerate(coords2D)}
            AllChem.Compute2DCoords(mol, coordMap=coord_dict)

    drawer = rdMolDraw2D.MolDraw2DSVG(200 * number, 200)
    drawer.DrawMolecules(
        molecules,
        highlightAtoms=matching,
        legends=legends,
        **kwargs,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open("highlighted_molecules.svg", "w") as f:
        f.write(svg)

    return SVG(svg)


mcs_result_thresh = rdFMCS.FindMCS(molecules, threshold=0.8)
print(f"MCS (80%) SMARTS: {mcs_result_thresh.smartsString}")
print(f"Liczba atomów (80%): {mcs_result_thresh.numAtoms}")
print(f"Liczba wiązań (80%): {mcs_result_thresh.numBonds}")

mcs_thresh_molecule = Chem.MolFromSmarts(mcs_result_thresh.smartsString)
Draw.MolToFile(mcs_thresh_molecule, "mcs_structure_80.svg")
highlight_molecules(molecules, mcs_result_thresh, 5)

mcs_result_rings = rdFMCS.FindMCS(molecules, completeRingsOnly=True)
print(f"MCS (tylko pierścienie) SMARTS: {mcs_result_rings.smartsString}")
print(f"Liczba atomów (tylko pierścienie): {mcs_result_rings.numAtoms}")
print(f"Liczba wiązań (tylko pierścienie): {mcs_result_rings.numBonds}")

mcs_rings_molecule = Chem.MolFromSmarts(mcs_result_rings.smartsString)
Draw.MolToFile(mcs_rings_molecule, "mcs_structure_rings.svg")
highlight_molecules(molecules, mcs_result_rings, 5)

comparison = Draw.MolsToGridImage(
    [mcs_molecule, mcs_thresh_molecule, mcs_rings_molecule],
    legends=["MCS pełny", "MCS 80%", "MCS pierścienie"],
    molsPerRow=3,
    subImgSize=(250, 250)
)
comparison.save("mcs_comparison.png")


egfr_df = pd.read_csv("egfr_bioactivities.csv")
egfr_filtered_df = egfr_df[egfr_df['pIC50'] > 9]
egfr_molecules = [Chem.MolFromSmiles(smi) for smi in egfr_filtered_df['canonical_smiles']]


random.seed(42)
sampled_molecules = random.sample(egfr_molecules, min(50, len(egfr_molecules)))


mcs_sampled = rdFMCS.FindMCS(sampled_molecules)
print(f"MCS EGFR (50 cząst.) SMARTS: {mcs_sampled.smartsString}")
print(f"Liczba atomów EGFR: {mcs_sampled.numAtoms}")
print(f"Liczba wiązań EGFR: {mcs_sampled.numBonds}")

highlight_molecules(sampled_molecules, mcs_sampled, 5)
