# Import bibliotek
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys

import random


seed = 22
random.seed(seed)
np.random.seed(seed)


from warnings import filterwarnings
filterwarnings("ignore")


data = pd.read_csv('final_filtered_bioactivities.csv') 


data = data[['molecule_chembl_id', 'canonical_smiles', 'pIC50']]


data['active'] = data['pIC50'].apply(lambda x: 1 if x >= 6.3 else 0)


def smiles_to_maccs(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return list(MACCSkeys.GenMACCSKeys(mol))

data['maccs'] = data['canonical_smiles'].apply(smiles_to_maccs)
data.dropna(inplace=True)

X = np.array(list(data['maccs']))
y = np.array(data['active'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=seed),
    "SVM": svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True, random_state=seed),
    "ANN": MLPClassifier(hidden_layer_sizes=(5, 3), random_state=seed)
}


for name, model in models.items():
    print(f"\n{name} model:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

  
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"AUC: {auc:.2f}")

  
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves')
plt.legend()
plt.grid()
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=seed)

for name, model in models.items():
    aucs = []
    for train_index, test_index in kf.split(X):
        X_tr, X_val = X[train_index], X[test_index]
        y_tr, y_val = y[train_index], y[test_index]

        model.fit(X_tr, y_tr)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_val_proba))

    print(f"\n{name} - Mean AUC (5-fold CV): {np.mean(aucs):.2f} ± {np.std(aucs):.2f}")
