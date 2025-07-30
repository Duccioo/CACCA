import pandas as pd
from rdkit import Chem

# NUOVO: Importiamo AllChem per i fingerprint
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

# 1. Caricamento dei Dati (invariato)
try:
    df = pd.read_csv("assets/mol_test.csv")
except FileNotFoundError:
    print("Errore: il file 'test.csv' non è stato trovato.")
    print(
        "Assicurati che il file si trovi nella stessa directory dello script o fornisci il percorso corretto."
    )
    exit()

df["SMILES"] = df["SMILES"].str.strip().str.strip('"')

# 2. Generazione delle Feature (Descrittori + Fingerprint) - MODIFICATO
descriptor_names = ["LogP", "TPSA", "NumHDonors", "NumHAcceptors", "ExactMolWt"]
metadata_list = []
descriptor_list = []
fingerprint_list = []

for index, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            descriptors = [
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.ExactMolWt(mol),
            ]
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            metadata_list.append([row["MOL"], row["PAPER"]])
            descriptor_list.append(descriptors)
            fingerprint_list.append(np.array(fp))
        except Exception as e:
            print(f"Errore su {row['MOL']}: {e}")
    else:
        print(f"SMILES non valido per: {row['MOL']}")

descriptor_array = np.array(descriptor_list)
fingerprint_array = np.array(fingerprint_list)
processed_df = pd.DataFrame(metadata_list, columns=["MOL", "PAPER"])

## 3. Standardizzazione separata
# Standardizziamo i descrittori e i fingerprint separatamente
scaler_desc = StandardScaler()
scaled_descriptors = scaler_desc.fit_transform(descriptor_array)

scaler_fp = StandardScaler()
scaled_fingerprints = scaler_fp.fit_transform(fingerprint_array)

# 4. Compressione dei Fingerprint con PCA
# Riduciamo le 1024 dimensioni dei fingerprint a un numero gestibile, es. 10
# Queste 10 "super-feature" rappresentano l'informazione strutturale più importante
n_fp_components = 10
pca_fp = PCA(n_components=n_fp_components, random_state=42)
compressed_fingerprints = pca_fp.fit_transform(scaled_fingerprints)

print(
    f"\nCompressi i fingerprint da {fingerprint_array.shape[1]} a {n_fp_components} dimensioni."
)

# 5. Creazione della Matrice di Feature Bilanciata
# Combiniamo i 5 descrittori standardizzati con i 10 fingerprint compressi
balanced_features = np.hstack((scaled_descriptors, compressed_fingerprints))

print(
    f"Creata matrice di feature bilanciata con {balanced_features.shape[1]} colonne totali."
)
# Nota: NON è necessario ri-standardizzare 'balanced_features', perché i suoi componenti sono già scalati.

# 6. Applicazione di UMAP sulla matrice bilanciata
# Usiamo i parametri ottimizzati dal Passo 1
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=4,  # Valore basso adatto a piccoli dataset
    min_dist=0.1,  # Permette cluster densi
)
embedding = reducer.fit_transform(balanced_features)
umap_df = pd.DataFrame(data=embedding, columns=["UMAP1", "UMAP2"])

# Unisci i risultati della UMAP con i metadati
final_umap_df = pd.concat([processed_df, umap_df], axis=1)

# 8. Visualizzazione UMAP (codice di plotting invariato)
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x="UMAP1",
    y="UMAP2",
    hue="PAPER",
    data=final_umap_df,
    palette="viridis",
    s=100,
    alpha=0.8,
)
for i in range(final_umap_df.shape[0]):
    plt.text(
        final_umap_df["UMAP1"][i] + 0.1,
        final_umap_df["UMAP2"][i],
        final_umap_df["MOL"][i],
        fontsize=9,
    )
plt.title("UMAP dello Spazio Chimico (Descrittori + Fingerprint Morgan)", fontsize=16)
plt.xlabel("Componente UMAP 1", fontsize=12)
plt.ylabel("Componente UMAP 2", fontsize=12)
plt.legend(title="Gruppo (PAPER)")
plt.savefig("umap_plot_with_fingerprints.png", dpi=300)
plt.close()

print(
    "\nGrafici salvati come 'pca_plot_with_fingerprints.png' e 'umap_plot_with_fingerprints.png'"
)
