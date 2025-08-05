import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import argparse
import os
import pathlib


# Configurazione opzioni
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Chemical space analysis using PCA and UMAP"
    )
    parser.add_argument(
        "--mol-file",
        type=str,
        default=pathlib.Path("assets", "mol_test_2.csv"),
        help="Path to the CSV file containing SMILES and metadata (default: assets/mol_test_2.csv)",
    )
    parser.add_argument(
        "--use-fingerprints",
        action="store_true",
        help="Include Morgan fingerprints in the analysis (default: descriptors only)",
    )
    parser.add_argument(
        "--n-fp-components",
        type=int,
        default=10,
        help="Number of PCA components to compress fingerprints (default: 10)",
    )
    return parser.parse_args()


# 6. Funzioni per creare i plot con seaborn
def create_pca_plot(data, title_suffix, filename_suffix, explained_variance):
    """Crea un plot PCA utilizzando seaborn con stile migliorato"""

    # Configurazione stile seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10})

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot principale con seaborn
    scatter = sns.scatterplot(
        data=data,
        x="PC1",
        y="PC2",
        hue="PAPER",
        palette="viridis",
        s=120,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        ax=ax,
    )

    # Aggiunta etichette per ogni punto
    for i in range(data.shape[0]):
        ax.annotate(
            data["MOL"].iloc[i],
            (data["PC1"].iloc[i], data["PC2"].iloc[i]),
            textcoords="offset points",
            fontsize=7,
            alpha=0.7,
        )

    # Personalizzazione del plot
    pc1_var = explained_variance[0] * 100
    pc2_var = explained_variance[1] * 100
    total_var = sum(explained_variance) * 100

    ax.set_title(
        f"PCA of Chemical Space ({title_suffix})\nTotal Explained Variance: {total_var:.1f}%",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=12, fontweight="bold")

    # Miglioramento della legenda
    legend = ax.legend(
        title="Group",
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    # Griglia più sottile
    ax.grid(True, alpha=0.3)

    # Salvataggio
    plt.tight_layout()
    plt.savefig(f"pca_plot_{filename_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return f"pca_plot_{filename_suffix}.png"


def create_umap_plot(data, title_suffix, filename_suffix):
    """Crea un plot UMAP utilizzando seaborn con stile migliorato"""

    # Configurazione stile seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10})

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot principale con seaborn
    scatter = sns.scatterplot(
        data=data,
        x="UMAP1",
        y="UMAP2",
        hue="PAPER",
        palette="viridis",
        s=120,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        ax=ax,
    )

    # Aggiunta etichette per ogni punto
    for i in range(data.shape[0]):
        ax.annotate(
            data["MOL"].iloc[i],
            (data["UMAP1"].iloc[i], data["UMAP2"].iloc[i]),
            textcoords="offset points",
            fontsize=7,
            alpha=0.7,
        )

    # Personalizzazione del plot
    ax.set_title(
        f"UMAP of Chemical Space ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("UMAP-1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP-2", fontsize=12, fontweight="bold")

    # Miglioramento della legenda
    legend = ax.legend(
        title="Group",
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    # Griglia più sottile
    ax.grid(True, alpha=0.3)

    # Salvataggio
    plt.tight_layout()
    plt.savefig(f"umap_plot_{filename_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return f"umap_plot_{filename_suffix}.png"


def main():

    # Creazione della cartella di output
    os.makedirs("output", exist_ok=True)

    # Parsing degli argomenti
    args = parse_arguments()
    MOL_FILE = args.mol_file
    USE_FINGERPRINTS = args.use_fingerprints
    N_FP_COMPONENTS = args.n_fp_components

    print(
        f"Mode: {'Descriptors + Fingerprints' if USE_FINGERPRINTS else 'Descriptors Only'}"
    )
    if USE_FINGERPRINTS:
        print(f"PCA components for fingerprints: {N_FP_COMPONENTS}")

    # 1. Caricamento dei Dati
    try:
        df = pd.read_csv(MOL_FILE)
    except FileNotFoundError:
        print(f"Error: '{MOL_FILE}' not found.")
        print("Please ensure the file is in the correct directory.")
        exit()

    df["SMILES"] = df["SMILES"].str.strip().str.strip('"')

    # 2. Generazione delle Feature
    descriptor_names = ["LogP", "TPSA", "NumHDonors", "NumHAcceptors", "ExactMolWt"]
    metadata_list = []
    descriptor_list = []
    fingerprint_list = []

    print("Processing molecules...")
    for index, row in df.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                # Calcolo descrittori chimici
                descriptors = [
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.ExactMolWt(mol),
                ]

                # Calcolo fingerprint solo se richiesto
                if USE_FINGERPRINTS:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fingerprint_list.append(np.array(fp))

                metadata_list.append([row["MOL"], row["PAPER"].strip()])
                descriptor_list.append(descriptors)

            except Exception as e:
                print(f"Error on {row['MOL']}: {e}")
        else:
            print(f"Invalid SMILES for: {row['MOL']}")

    descriptor_array = np.array(descriptor_list)
    processed_df = pd.DataFrame(metadata_list, columns=["MOL", "PAPER"])

    print(f"Successfully processed {len(descriptor_list)} molecules.")

    # 3. Preparazione delle Feature
    # Standardizziamo sempre i descrittori
    scaler_desc = StandardScaler()
    scaled_descriptors = scaler_desc.fit_transform(descriptor_array)

    if USE_FINGERPRINTS:
        # Standardizzazione e compressione dei fingerprint
        fingerprint_array = np.array(fingerprint_list)

        # Compressione dei fingerprint con PCA
        pca_fp = PCA(n_components=N_FP_COMPONENTS, random_state=42)
        compressed_fingerprints = pca_fp.fit_transform(fingerprint_array)

        print(
            f"Compressed fingerprints from {fingerprint_array.shape[1]} to {N_FP_COMPONENTS} dimensions."
        )

        # Combinazione descrittori + fingerprint compressi
        final_features = np.hstack((scaled_descriptors, compressed_fingerprints))
        feature_type = "Descriptors + Fingerprints"

    else:
        # Solo descrittori
        final_features = scaled_descriptors
        feature_type = "Descriptors Only"

    print(f"Final feature matrix: {final_features.shape[1]} features ({feature_type})")

    # 4. Applicazione di PCA
    print("Applying PCA...")
    pca_2d = PCA(n_components=2, random_state=42)
    pca_embedding = pca_2d.fit_transform(final_features)
    pca_df = pd.DataFrame(data=pca_embedding, columns=["PC1", "PC2"])

    # Unisci i risultati della PCA con i metadati
    final_pca_df = pd.concat([processed_df, pca_df], axis=1)

    print(
        f"PCA explained variance: PC1={pca_2d.explained_variance_ratio_[0]:.3f}, PC2={pca_2d.explained_variance_ratio_[1]:.3f}"
    )
    print(f"Total explained variance: {sum(pca_2d.explained_variance_ratio_):.3f}")

    # 5. Applicazione di UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=4,  # Valore basso adatto a piccoli dataset
        min_dist=0.1,  # Permette cluster densi
    )
    embedding = reducer.fit_transform(final_features)
    umap_df = pd.DataFrame(data=embedding, columns=["UMAP1", "UMAP2"])

    # Unisci i risultati della UMAP con i metadati
    final_umap_df = pd.concat([processed_df, umap_df], axis=1)

    # 7. Creazione dei plot
    if USE_FINGERPRINTS:
        # Crea entrambi i grafici PCA e UMAP
        pca_filename = create_pca_plot(
            final_pca_df,
            "Descriptors + Morgan Fingerprints",
            "with_fingerprints",
            pca_2d.explained_variance_ratio_,
        )
        umap_filename = create_umap_plot(
            final_umap_df, "Descriptors + Morgan Fingerprints", "with_fingerprints"
        )
        print(f"\nPlots saved:")
        print(f"- PCA: '{pca_filename}'")
        print(f"- UMAP: '{umap_filename}'")
    else:
        # Crea entrambi i grafici PCA e UMAP
        pca_filename = create_pca_plot(
            final_pca_df,
            "Chemical Descriptors Only",
            "descriptors_only",
            pca_2d.explained_variance_ratio_,
        )
        umap_filename = create_umap_plot(
            final_umap_df, "Chemical Descriptors Only", "descriptors_only"
        )
        print(f"\nPlots saved:")
        print(f"- PCA: '{pca_filename}'")
        print(f"- UMAP: '{umap_filename}'")

    print(f"\nAnalysis completed using: {feature_type}")
    print(f"Total number of features used: {final_features.shape[1]}")
    print(f"Number of molecules analyzed: {final_features.shape[0]}")


if __name__ == "__main__":
    main()
