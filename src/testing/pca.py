import argparse
import pathlib
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Funzioni di Supporto ---


def parse_arguments() -> argparse.Namespace:
    """Configura e analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(
        description="Analisi dello spazio chimico tramite PCA e UMAP con plotting di qualità.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mol-file",
        type=pathlib.Path,
        default=pathlib.Path("assets", "mol_test.csv"),
        help="Percorso del file CSV contenente SMILES e metadati.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("output_analysis"),
        help="Directory dove salvare i grafici e i risultati.",
    )
    parser.add_argument(
        "--use-fingerprints",
        action="store_true",
        help="Includi i Morgan fingerprints nell'analisi (oltre ai descrittori).",
    )
    parser.add_argument(
        "--n-fp-components",
        type=int,
        default=10,
        help="Numero di componenti PCA per comprimere i fingerprints.",
    )
    # NUOVI/AGGIORNATI ARGOMENTI
    parser.add_argument(
        "--ref-smiles",
        type=str,
        default="CN(CCO)C(=O)C(c1ccccc1)c2ccccc2",
        help="SMILES di una molecola da evidenziare nei grafici.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Aggiungi etichette con il nome della molecola su ogni punto (sconsigliato per molti dati).",
    )
    return parser.parse_args()


def generate_features(
    df: pd.DataFrame, use_fingerprints: bool, n_fp_components: int
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """
    Calcola le feature (descrittori e/o fingerprints) dal DataFrame di molecole.

    Returns:
        Una tupla contenente:
        - L'array finale delle feature, pronto per la riduzione dimensionale.
        - Un DataFrame con i metadati delle molecole processate con successo.
        - Una stringa che descrive il tipo di feature utilizzate.
    """
    descriptor_list, fingerprint_list, metadata_list = [], [], []

    print("INFO: Processing molecules...")
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if not mol:
            print(f"WARN: Invalid SMILES for molecule {row['MOL']}. Skipping.")
            continue
        try:
            descriptor_list.append(
                [
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.ExactMolWt(mol),
                ]
            )
            if use_fingerprints:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprint_list.append(np.array(fp))
            metadata_list.append([row["MOL"], row["PAPER"]])
        except Exception as e:
            print(f"ERROR: Could not process molecule {row['MOL']}: {e}")

    if not descriptor_list:
        raise ValueError("No molecules were successfully processed.")

    processed_df = pd.DataFrame(metadata_list, columns=["MOL", "PAPER"])
    print(f"INFO: Successfully processed {len(processed_df)} molecules.")

    scaled_descriptors = StandardScaler().fit_transform(np.array(descriptor_list))

    if use_fingerprints and fingerprint_list:
        fp_array = np.array(fingerprint_list)
        pca_fp = PCA(n_components=n_fp_components, random_state=42)
        compressed_fingerprints = pca_fp.fit_transform(fp_array)
        final_features = np.hstack((scaled_descriptors, compressed_fingerprints))
        feature_type = "Descriptors + Morgan FP"
    else:
        final_features = scaled_descriptors
        feature_type = "Descriptors Only"

    print(f"INFO: Final feature matrix shape: {final_features.shape} ({feature_type})")
    return final_features, processed_df, feature_type


def create_publication_plot(
    data: pd.DataFrame,
    method: str,  # 'PCA' or 'UMAP'
    feature_type: str,
    output_dir: pathlib.Path,
    annotate: bool,
    ref_mol_name: Optional[str] = None,
    explained_variance: Optional[List[float]] = None,
) -> None:
    """
    Crea e salva un grafico di alta qualità per PCA o UMAP, con opzioni per annotazione ed evidenziazione.
    """
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = sns.scatterplot(
        data=data,
        x=f"{method}1",
        y=f"{method}2",
        hue="Origin",
        style="Origin",
        palette="deep",
        s=80,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    if method == "PCA":
        pc1_var, pc2_var = explained_variance[0] * 100, explained_variance[1] * 100
        ax.set_xlabel(f"Principal Component 1 ({pc1_var:.1f}%)", fontsize=14)
        ax.set_ylabel(f"Principal Component 2 ({pc2_var:.1f}%)", fontsize=14)
        ax.set_title(
            f"PCA of Chemical Space\n({feature_type})", fontsize=18, weight="bold"
        )
    else:
        ax.set_xlabel("UMAP Dimension 1", fontsize=14)
        ax.set_ylabel("UMAP Dimension 2", fontsize=14)
        ax.set_title(
            f"UMAP of Chemical Space\n({feature_type})", fontsize=18, weight="bold"
        )

    # EVIDENZIAZIONE MOLECOLA DI RIFERIMENTO
    if ref_mol_name:
        ref_point = data[data["MOL"] == ref_mol_name]
        if not ref_point.empty:
            ax.scatter(
                ref_point[f"{method}1"],
                ref_point[f"{method}2"],
                s=250,
                facecolors="none",
                edgecolors="red",
                linewidths=2.5,
                label="Reference",
            )
            print(f"INFO: Highlighted '{ref_mol_name}' in the {method} plot.")

    # ANNOTAZIONI (OPZIONALI)
    if annotate:
        print(f"INFO: Annotating points in the {method} plot...")
        for _, row in data.iterrows():
            ax.text(
                x=row[f"{method}1"],
                y=row[f"{method}2"],
                s=row["MOL"],
                ha="center",
                va="bottom",
                fontdict={"size": 7, "alpha": 0.75},
            )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Source", frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    filename = f"{method.lower()}_plot_{'fp' if 'FP' in feature_type else 'desc'}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"INFO: Plot saved to '{output_path}'")
    plt.close(fig)


# --- Funzione Principale ---


def main():
    """Flusso principale dello script."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Output will be saved to: '{args.output_dir}'")

    try:
        df = pd.read_csv(args.mol_file)
        df["SMILES"] = df["SMILES"].str.strip().str.strip('"')

    except FileNotFoundError:
        print(f"FATAL: File not found at '{args.mol_file}'. Please check the path.")
        return

    # Trova il nome della molecola di riferimento, se specificata
    ref_mol_name = None
    if args.ref_smiles:
        ref_row = df[df["SMILES"] == args.ref_smiles]
        if not ref_row.empty:
            ref_mol_name = ref_row.iloc[0]["MOL"]
            print(
                f"INFO: Reference SMILES found. Will highlight molecule: '{ref_mol_name}'"
            )
        else:
            print(
                f"WARN: Reference SMILES '{args.ref_smiles}' not found in the dataset."
            )

    final_features, processed_df, feature_type = generate_features(
        df, args.use_fingerprints, args.n_fp_components
    )
    processed_df["Origin"] = processed_df["PAPER"].apply(
        lambda x: "In Paper" if x == 1 else "Not In Paper"
    )

    # Applicazione di PCA
    print("\nINFO: Applying PCA...")
    pca_2d = PCA(n_components=2, random_state=42)
    pca_embedding = pca_2d.fit_transform(final_features)
    pca_df = pd.DataFrame(pca_embedding, columns=["PCA1", "PCA2"])
    final_pca_df = pd.concat([processed_df.reset_index(drop=True), pca_df], axis=1)
    print(f"INFO: PCA explained variance: {pca_2d.explained_variance_ratio_}")

    # Applicazione di UMAP
    print("INFO: Applying UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
    umap_embedding = reducer.fit_transform(final_features)
    umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
    final_umap_df = pd.concat([processed_df.reset_index(drop=True), umap_df], axis=1)

    # Creazione dei grafici
    print("\nINFO: Generating plots...")
    create_publication_plot(
        data=final_pca_df,
        method="PCA",
        feature_type=feature_type,
        output_dir=args.output_dir,
        annotate=args.annotate,
        ref_mol_name=ref_mol_name,
        explained_variance=pca_2d.explained_variance_ratio_,
    )

    create_publication_plot(
        data=final_umap_df,
        method="UMAP",
        feature_type=feature_type,
        output_dir=args.output_dir,
        annotate=args.annotate,
        ref_mol_name=ref_mol_name,
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
