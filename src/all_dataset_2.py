import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

# --- 1. Caricamento dei due Dataset ---
try:
    # Dataset grande per l'addestramento dei modelli
    df_train = pd.read_csv('dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore.csv') 
    # Dataset piccolo da visualizzare
    df_test = pd.read_csv('assets/mol_test.csv')
    df_test["SMILES"] = df_test["SMILES"].str.strip().str.strip('"')
except FileNotFoundError as e:
    print(f"Errore nel caricamento dei file: {e}")
    exit()

print(f"Dataset di addestramento caricato con {len(df_train)} molecole.")
print(f"Dataset di test caricato con {len(df_test)} molecole.")

# --- 2. Feature Engineering sul Dataset di TRAINING ---
# Questa sezione serve a creare la matrice di feature e ad ADDESTRARE gli scaler e la PCA

print("\n--- Processando il dataset di TRAINING... ---")
# Seleziona casualmente 10.000 righe dal dataset di training
df_train = df_train.sample(n=10000, random_state=42).reset_index(drop=True)

train_desc_list, train_fp_list = [], []
for _, row in df_train.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol:
        train_desc_list.append(row[['ExactMW', 'LogP', 'NumHBD', 'NumHBA', 'TPSA']].values)
        train_fp_list.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)))

train_desc_array = np.array(train_desc_list)
train_fp_array = np.array(train_fp_list)

# ADDESTRIAMO gli scaler SUL SOLO DATASET DI TRAINING
scaler_desc = StandardScaler().fit(train_desc_array)
scaler_fp = StandardScaler().fit(train_fp_array)

# Applichiamo la trasformazione
scaled_train_desc = scaler_desc.transform(train_desc_array)
scaled_train_fp = scaler_fp.transform(train_fp_array)

# ADDESTRIAMO la PCA sui fingerprint di training
pca_fp = PCA(n_components=10, random_state=42).fit(scaled_train_fp)
compressed_train_fp = pca_fp.transform(scaled_train_fp)

# Matrice finale di feature per il training
balanced_features_train = np.hstack((scaled_train_desc, compressed_train_fp))
print(f"Matrice di training creata con forma: {balanced_features_train.shape}")


# --- 3. Feature Engineering sul Dataset di TEST ---
# Qui usiamo gli scaler e la PCA GIA' ADDESTRATI per trasformare i dati di test

print("\n--- Processando il dataset di TEST... ---")
test_metadata_list, test_desc_list, test_fp_list = [], [], []
for _, row in df_test.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol:
        # Calcoliamo i descrittori da zero per il set di test
        descriptors = [Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.ExactMolWt(mol)]
        test_desc_list.append(descriptors)
        test_fp_list.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)))
        test_metadata_list.append(row['MOL'])

test_desc_array = np.array(test_desc_list)
test_fp_array = np.array(test_fp_list)
test_metadata_df = pd.DataFrame(test_metadata_list, columns=["MOL"])

# **PUNTO CRUCIALE**: Usiamo .transform(), NON .fit_transform()
scaled_test_desc = scaler_desc.transform(test_desc_array)
scaled_test_fp = scaler_fp.transform(test_fp_array)
compressed_test_fp = pca_fp.transform(scaled_test_fp)

# Matrice finale di feature per il test (nello stesso spazio di quelle di training)
balanced_features_test = np.hstack((scaled_test_desc, compressed_test_fp))
print(f"Matrice di test creata con forma: {balanced_features_test.shape}")


# --- 4. Addestramento dei modelli su TRAINING e Predizione su TEST ---

# Scegliamo k usando il metodo del gomito sui dati di training
inertia = []
K = range(2, 11)
for k in K:
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(balanced_features_train)
    inertia.append(kmeans_elbow.inertia_)

# Scegliamo k=5 come in precedenza (o il valore suggerito dal tuo nuovo grafico a gomito)
OPTIMAL_K = 5 

# ADDESTRIAMO KMeans SUL TRAINING
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto').fit(balanced_features_train)
# PREVEDIAMO i cluster per il TEST
test_cluster_labels = kmeans.predict(balanced_features_test)
test_metadata_df['KMeans_Cluster'] = test_cluster_labels

# ADDESTRIAMO UMAP SUL TRAINING
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit(balanced_features_train)
# TRASFORMIAMO le coordinate per il TEST
test_embedding = reducer.transform(balanced_features_test)
test_umap_df = pd.DataFrame(data=test_embedding, columns=["UMAP1", "UMAP2"])


# --- 5. Visualizzazione finale (solo dati di TEST) ---

final_df = pd.concat([test_metadata_df.reset_index(drop=True), test_umap_df], axis=1)

print("\n--- Risultati del Clustering per il set di Test ---")
for i in range(OPTIMAL_K):
    cluster_members = final_df[final_df['KMeans_Cluster'] == i]['MOL'].tolist()
    if cluster_members:
        print(f"  Cluster {i}: {', '.join(cluster_members)}")


plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 10))

scatter = sns.scatterplot(
    x="UMAP1", y="UMAP2", hue="KMeans_Cluster", data=final_df, 
    palette="deep", s=120, alpha=0.9
)
for i in range(final_df.shape[0]):
    plt.text(final_df["UMAP1"][i] + 0.05, final_df["UMAP2"][i], final_df["MOL"][i], fontsize=9)

plt.title(f"Posizione delle Molecole di Test nella Mappa Globale (k={OPTIMAL_K})", fontsize=16)
plt.xlabel("Componente UMAP 1")
plt.ylabel("Componente UMAP 2")
plt.legend(title="Cluster Predetto")
plt.savefig("test_data_on_global_map.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\nGrafico finale salvato come 'test_data_on_global_map.png'")