import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# NUOVO: Importiamo KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

# --- Sezioni 1-5 (Caricamento, Feature Engineering, Bilanciamento) ---
# QUESTA PARTE RESTA IDENTICA AL CODICE PRECEDENTE
# Il risultato finale di questa parte è la matrice 'balanced_features'
# ... (omettiamo il codice per brevità, il risultato è 'balanced_features' e 'processed_df') ...

# --- Esecuzione del codice precedente fino alla creazione delle feature bilanciate ---
df = pd.read_csv("assets/mol_test.csv")
df["SMILES"] = df["SMILES"].str.strip().str.strip('"')

descriptor_names = ["LogP", "TPSA", "NumHDonors", "NumHAcceptors", "ExactMolWt"]
metadata_list, descriptor_list, fingerprint_list = [], [], []

for index, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol:
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

descriptor_array = np.array(descriptor_list)
fingerprint_array = np.array(fingerprint_list)
processed_df = pd.DataFrame(metadata_list, columns=["MOL", "PAPER"])

scaler_desc = StandardScaler()
scaled_descriptors = scaler_desc.fit_transform(descriptor_array)
scaler_fp = StandardScaler()
scaled_fingerprints = scaler_fp.fit_transform(fingerprint_array)

pca_fp = PCA(n_components=10, random_state=42)
compressed_fingerprints = pca_fp.fit_transform(scaled_fingerprints)

balanced_features = np.hstack((scaled_descriptors, compressed_fingerprints))
# --- Fine del codice precedente ---


# --- NUOVA SEZIONE: APPLICAZIONE DI K-MEANS ---

# 6. Trovare il 'k' ottimale con il Metodo del Gomito
inertia = []
K = range(2, 11)  # Testiamo da 2 a 10 cluster
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(balanced_features)
    inertia.append(kmeans.inertia_)

# Visualizzazione del grafico a gomito
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, "bx-")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Inerzia")
plt.title("Metodo del Gomito per la Scelta di k")
plt.savefig("kmeans_elbow_plot.png", dpi=300)
plt.close()

# Dal grafico, scegliamo un valore per k dove si forma il "gomito".
# Guardando il grafico, un valore di k=5 o k=6 sembra ragionevole. Usiamo k=5.
OPTIMAL_K = 6
print(f"\nScelto k={OPTIMAL_K} basandosi sul metodo del gomito.")

# 7. Esecuzione di K-Means con il k ottimale
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init="auto")
# Applichiamo il clustering alla nostra matrice di feature bilanciate
kmeans_labels = kmeans.fit_predict(balanced_features)

# Aggiungiamo le etichette dei cluster trovati al nostro DataFrame
processed_df["KMeans_Cluster"] = kmeans_labels


# 8. Esecuzione di UMAP (per la visualizzazione)
# Questa parte non cambia, serve solo a creare le coordinate 2D per il grafico
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=4, min_dist=0.1)
embedding = reducer.fit_transform(balanced_features)
umap_df = pd.DataFrame(data=embedding, columns=["UMAP1", "UMAP2"])

# Uniamo tutto in un DataFrame finale per la visualizzazione
final_df = pd.concat([processed_df.reset_index(drop=True), umap_df], axis=1)

# 9. Visualizzazione UMAP colorata per Cluster K-Means
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 10))

# MODIFICATO: Usiamo 'KMeans_Cluster' per la colorazione (hue)
scatter = sns.scatterplot(
    x="UMAP1",
    y="UMAP2",
    hue="KMeans_Cluster",  # <- MODIFICA CHIAVE
    data=final_df,
    palette="deep",  # Usiamo una palette diversa per distinguere dal grafico precedente
    s=120,
    alpha=0.9,
)

for i in range(final_df.shape[0]):
    plt.text(
        final_df["UMAP1"][i] + 0.05,
        final_df["UMAP2"][i],
        final_df["MOL"][i],
        fontsize=9,
    )

plt.title(f"Visualizzazione UMAP colorata da K-Means (k={OPTIMAL_K})", fontsize=16)
plt.xlabel("Componente UMAP 1", fontsize=12)
plt.ylabel("Componente UMAP 2", fontsize=12)
plt.legend(title="Cluster K-Means")
plt.savefig("umap_kmeans_clusters.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nGrafici salvati come 'kmeans_elbow_plot.png' e 'umap_kmeans_clusters.png'")

# Stampa i membri di ciascun cluster per un'analisi dettagliata
for i in range(OPTIMAL_K):
    print(f"\n--- Membri del Cluster {i} ---")
    cluster_members = final_df[final_df["KMeans_Cluster"] == i]["MOL"].tolist()
    print(", ".join(cluster_members))
