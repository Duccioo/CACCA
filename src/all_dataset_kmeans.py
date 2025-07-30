import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

# --- 1. Caricamento Dati e Feature Engineering (MODIFICATO) ---

try:
    # Assicurati di salvare il tuo nuovo file CSV come 'new_data.csv'
    df = pd.read_csv('dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore.csv') 
except FileNotFoundError:
    print("Errore: il file 'new_data.csv' non è stato trovato.")
    exit()

# MODIFICA CHIAVE: L'approccio ora è molto più diretto.

metadata_list = []
descriptor_list = []
fingerprint_list = []

# Iteriamo su ogni riga del nuovo DataFrame
# Seleziona 10.000 righe a caso dal DataFrame originale
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)

for index, row in df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        try:
            # A. LEGGIAMO i descrittori già calcolati e scalati dal file
            # Non li calcoliamo più!
            precalculated_descriptors = row[['ExactMW', 'LogP', 'NumHBD', 'NumHBA', 'TPSA']].values
            
            # B. CALCOLIAMO solo i fingerprint Morgan, come prima
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            
            # C. SALVIAMO i dati
            # Non avendo un nome, creiamo un ID generico per ogni molecola
            metadata_list.append([f"Mol_{index}"]) 
            descriptor_list.append(precalculated_descriptors)
            fingerprint_list.append(np.array(fp))
            
        except Exception as e:
            print(f"Errore sulla molecola con indice {index}: {e}")
    else:
        print(f"SMILES non valido per la molecola con indice: {index}")

# Convertiamo le liste in array NumPy
descriptor_array = np.array(descriptor_list)
fingerprint_array = np.array(fingerprint_list)
# Il nostro DataFrame di metadati ora contiene solo un ID
processed_df = pd.DataFrame(metadata_list, columns=["MOL_ID"])


# --- 2. Bilanciamento delle Feature (leggermente modificato) ---

# Dato che i descrittori sono GIA' SCALATI, StandardScaler non farebbe nulla di dannoso,
# ma per correttezza logica, potremmo saltare questo passaggio.
# Lo manteniamo per rendere il codice più robusto nel caso in cui i dati non fossero perfettamente scalati.
scaler_desc = StandardScaler()
scaled_descriptors = scaler_desc.fit_transform(descriptor_array)

# I fingerprint DEVONO essere scalati perché sono appena stati generati (valori 0/1)
scaler_fp = StandardScaler()
scaled_fingerprints = scaler_fp.fit_transform(fingerprint_array)

# Il resto della logica di bilanciamento (PCA sui fingerprint) è IDENTICA
n_fp_components = 10 
pca_fp = PCA(n_components=n_fp_components, random_state=42)
compressed_fingerprints = pca_fp.fit_transform(scaled_fingerprints)
balanced_features = np.hstack((scaled_descriptors, compressed_fingerprints))

print(f"\nCreata matrice di feature bilanciata con {balanced_features.shape[0]} molecole e {balanced_features.shape[1]} colonne.")


# --- 3. K-Means (IDENTICO) ---
# Il metodo del gomito e il clustering funzionano sulla matrice 'balanced_features', quindi non cambiano.

inertia = []
# Scegliamo un range appropriato per il numero di dati
max_k = min(11, len(df)) # Non possiamo avere più cluster che dati
K = range(2, max_k)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(balanced_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Inerzia')
plt.title('Metodo del Gomito per la Scelta di k')
plt.show()
plt.close()

# Scegli un valore dal grafico. Mettiamo un default, ma modificalo in base al tuo grafico.
OPTIMAL_K = 5 
print(f"Usando k={OPTIMAL_K}. Modifica questa variabile se il grafico a gomito suggerisce un valore diverso.")

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(balanced_features)
processed_df['KMeans_Cluster'] = kmeans_labels


# --- 4. UMAP e Visualizzazione Finale (MODIFICATO) ---

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=4, min_dist=0.1)
embedding = reducer.fit_transform(balanced_features)
umap_df = pd.DataFrame(data=embedding, columns=["UMAP1", "UMAP2"])

final_df = pd.concat([processed_df.reset_index(drop=True), umap_df], axis=1)

# MODIFICA CHIAVE NELLA VISUALIZZAZIONE
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 10))

# La colorazione (hue) ora è basata sui cluster trovati da K-Means, non più su 'PAPER'
scatter = sns.scatterplot(
    x="UMAP1", y="UMAP2", hue="KMeans_Cluster", data=final_df, 
    palette="viridis", s=120, alpha=0.9
)

# Le etichette ora usano il nostro ID generico 'MOL_ID'
for i in range(final_df.shape[0]):
    plt.text(final_df["UMAP1"][i] + 0.05, final_df["UMAP2"][i], final_df["MOL_ID"][i], fontsize=9)

plt.title(f"UMAP e Clustering K-Means sul Nuovo Dataset (k={OPTIMAL_K})", fontsize=16)
plt.xlabel("Componente UMAP 1", fontsize=12)
plt.ylabel("Componente UMAP 2", fontsize=12)
plt.legend(title="Cluster K-Means")
plt.savefig("new_data_umap_clusters.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\nGrafico finale salvato come 'new_data_umap_clusters.png'")

for i in range(OPTIMAL_K):
    cluster_mols = final_df[final_df['KMeans_Cluster'] == i]['MOL_ID'].tolist()
    print(f"\n--- Cluster {i} --- ({len(cluster_mols)} molecole)")
    # Stampiamo anche gli SMILES per capire la chimica
    cluster_smiles = df.loc[final_df[final_df['KMeans_Cluster'] == i].index, 'SMILES'].tolist()
    for j, mol_id in enumerate(cluster_mols):
        print(f"  {mol_id}: {cluster_smiles[j]}")