# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # NUOVO: Importa Seaborn
import torch
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.spatial import distance

from model.CAE import CVAE, ConditionalAutoencoder, OLDConditionalAutoencoder
from generatore2 import load_vocab

PAD, EOS, SOS = "_", "E", "X"


def load_model_from_folder(
    folder, num_prop: int = 5, num_check="best"
) -> ConditionalAutoencoder:
    """Load the latest model checkpoint from a given folder."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    vocab_file = Path(folder) / "vocab.pkl"
    print(f"Loading vocabulary from {vocab_file} …")
    if vocab_file.exists():
        char, vocab, int_to_char, pad_idx, sos_idx, eos_idx = load_vocab(vocab_file)
    ckpts = [p for p in Path(folder).glob("model_*.pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {folder!s}")
    best_ckpt = [p for p in ckpts if p.stem.endswith("best")]
    if best_ckpt and num_check == "best":
        latest = best_ckpt[0]
    elif isinstance(num_check, int):
        latest = [p for p in ckpts if p.stem.endswith(str(num_check))]
        if latest:
            latest = latest[0]
        else:
            raise FileNotFoundError(
                f"No checkpoint ending with '{num_check}' found in {folder!s}"
            )
    else:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
    params = Path(folder) / "args.json"
    if params.exists():
        with open(params) as f:
            train_args = argparse.Namespace(**json.load(f))
    try:
        print(
            f"[INFO] Loading {train_args.model_type} model from {latest.as_posix()} …"
        )
        if train_args.model_type == "CVAE":
            model = CVAE(
                len(vocab),
                num_prop,
                train_args.latent_size,
                train_args.emb_size,
                train_args.hidden_size,
                train_args.n_rnn_layer,
            )
        else:
            model = ConditionalAutoencoder(
                len(vocab),
                num_prop,
                train_args.latent_size,
                train_args.emb_size,
                train_args.hidden_size,
                train_args.n_rnn_layer,
            )
    except:
        model = OLDConditionalAutoencoder(
            len(vocab),
            num_prop,
            train_args.latent_size,
            train_args.emb_size,
            train_args.hidden_size,
            train_args.n_rnn_layer,
        )
    model.load_state_dict(torch.load(latest, map_location=torch.device("cpu")))
    return model, vocab, train_args


def smiles_dict_to_tensor(
    smiles_df: pd.DataFrame, char2idx: dict[str, int], seq_length: int
) -> Tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    np.ndarray,
    List[str],
    List[str],
    List[int],
]:
    """Processa un DataFrame e restituisce tensori e liste di metadati."""
    xs, cs, ls, valid_smiles, logp_vals, names, paper_vals = [], [], [], [], [], [], []
    for row in smiles_df.itertuples():
        name = row.MOL
        smi = row.SMILES
        paper_status = row.PAPER
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) >= seq_length - 2:
            print(f"[WARN] Skipping invalid or too‑long SMILES: {smi} (name: {name})")
            continue
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp_vals.append(logp)
        token_ids = [char2idx.get(ch) for ch in smi if ch in char2idx]
        if not token_ids:
            print(
                f"[WARN] Skipping SMILES with no recognised chars: {smi} (name: {name})"
            )
            continue
        padded = (
            [char2idx[SOS]]
            + token_ids
            + [char2idx[EOS]] * (seq_length - len(token_ids) - 1)
        )
        xs.append(padded)
        cs.append([mw, logp, hbd, hba, tpsa])
        ls.append(len(token_ids) + 1)
        valid_smiles.append(smi)
        names.append(name)
        paper_vals.append(paper_status)
    if not xs:
        raise RuntimeError("No valid SMILES to process – aborting.")
    return (
        torch.as_tensor(xs, dtype=torch.long),
        torch.tensor(cs, dtype=torch.float32),
        torch.as_tensor(ls, dtype=torch.long),
        np.array(logp_vals, dtype=np.float32),
        valid_smiles,
        names,
        paper_vals,
    )


def load_smiles_from_csv(
    file_path: str,
    filter_not_in_paper: bool = False,
    n_rows: int | None = None,
    random_sample: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Carica un DataFrame di pandas da file CSV."""
    smiles_file = Path(file_path)
    df = pd.read_csv(smiles_file, skipinitialspace=True)
    if filter_not_in_paper:
        df = df[df["PAPER"] == 1].copy()
    if n_rows is not None:
        if random_sample:
            df = df.sample(n=n_rows, random_state=random_seed).reset_index(drop=True)
        else:
            df = df.head(n_rows).reset_index(drop=True)
    return df


###############################################################################
# NUOVA FUNZIONE DI PLOTTING
###############################################################################


def create_publication_plot(
    plot_df: pd.DataFrame,
    save_dir: Path,
    plot_type: str,
    annotate: bool = False,
    ref_name: str | None = None,
) -> None:
    """
    Crea e salva un grafico UMAP di qualità per la pubblicazione utilizzando Seaborn.

    Args:
        plot_df (pd.DataFrame): DataFrame contenente le coordinate UMAP e i metadati.
        save_dir (Path): Directory dove salvare il grafico.
        plot_type (str): Tipo di plot ('logp' o 'paper_status').
        annotate (bool): Se annotare i punti con i loro nomi.
        ref_name (str | None): Nome della molecola di riferimento da evidenziare.
    """
    # Imposta lo stile di Seaborn per un look professionale
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 8))

    if plot_type == "logp":
        # Plot colorato per logP
        scatter = sns.scatterplot(
            data=plot_df,
            x="UMAP-1",
            y="UMAP-2",
            hue="logP",
            palette="plasma",  # Un colormap percettivamente uniforme
            s=50,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        title = "UMAP Projection of Latent Space (Colored by logP)"
        filename = "umap_plot_logp.png"
        # Aggiungi una colorbar
        norm = plt.Normalize(plot_df["logP"].min(), plot_df["logP"].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        scatter.get_legend().remove()
        cbar = plt.colorbar(sm, ax=scatter.axes)
        cbar.set_label("logP", rotation=270, labelpad=20)

    elif plot_type == "paper_status":
        # Plot colorato per lo status 'PAPER'
        # Usa sia colore che stile del marcatore per massima chiarezza
        scatter = sns.scatterplot(
            data=plot_df,
            x="UMAP-1",
            y="UMAP-2",
            hue="Origin",  # Usa la colonna descrittiva
            style="Origin",  # Usa anche marcatori diversi
            s=60,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        title = "UMAP Projection of Latent Space (by Origin)"
        filename = "umap_plot_origin.png"
        plt.legend(title="Origin", frameon=True)

    else:
        raise ValueError("plot_type must be 'logp' or 'paper_status'")

    # Evidenzia il punto di riferimento, se specificato
    if ref_name:
        ref_point = plot_df[plot_df["Name"] == ref_name]
        if not ref_point.empty:
            plt.scatter(
                ref_point["UMAP-1"],
                ref_point["UMAP-2"],
                s=150,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="Reference Molecule",
            )
            # Aggiungi questa linea per assicurarti che la legenda "Reference Molecule" appaia
            if plot_type == "paper_status":
                handles, labels = scatter.get_legend_handles_labels()
                # Aggiungi la handle del punto di riferimento se non è già lì
                if "Reference Molecule" not in labels:
                    ref_handle = plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="none",
                        markeredgecolor="red",
                        markeredgewidth=2,
                        markersize=10,
                        label="Reference Molecule",
                    )
                    handles.append(ref_handle)
                    labels.append("Reference Molecule")
                plt.legend(handles=handles, labels=labels, title="Origin")

    plt.title(title, fontsize=18, weight="bold")
    plt.xlabel("UMAP-1", fontsize=14)
    plt.ylabel("UMAP-2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Aggiungi annotazioni (usare con cautela, può sovraffollare il grafico)
    if annotate:
        for i, row in plot_df.iterrows():
            plt.annotate(
                row["Name"],
                (row["UMAP-1"], row["UMAP-2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )

    out_path = save_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Publication-ready plot saved to {out_path.as_posix()}")
    plt.close()  # Chiudi la figura per liberare memoria


###############################################################################
# Main routine - MODIFICATA
###############################################################################


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    model, vocab, train_args = load_model_from_folder(args.save_dir)
    model.to(device)
    model.eval()

    molecules_df = load_smiles_from_csv("assets/mol_test.csv")
    x, c, l, logp_vals, valid_smiles, names, paper_vals = smiles_dict_to_tensor(
        molecules_df, vocab, train_args.max_seq_length
    )
    x, c, l = x.to(device), c.to(device), l.to(device)

    scale_file = Path(args.smiles_file)
    if scale_file.exists():
        with open(scale_file) as f:
            scale_data = json.load(f)
            print(f"[INFO] Loaded scaling metrics from {scale_file.as_posix()}")
            if scale_data["Scaling method"] == "zscore":
                scale_metrics = scale_data["Scale metrics"]
                mean = torch.tensor(
                    scale_metrics[0], device=device, dtype=torch.float32
                )
                std = torch.tensor(scale_metrics[1], device=device, dtype=torch.float32)
                c = (c - mean) / std

    with torch.no_grad():
        latent = model(x, c, l)[0].cpu().numpy()
        print(
            f"[INFO] Encoded {latent.shape[0]} molecules → latent_dim={latent.shape[1]}"
        )

    ref_name = None  # Inizializza il nome di riferimento
    if args.ref_smiles:
        try:
            ref_index = valid_smiles.index(args.ref_smiles)
            ref_embedding = latent[ref_index]
            ref_name = names[ref_index]  # Ottieni il nome della molecola di riferimento
            print(
                f"[INFO] Reference SMILES found at index {ref_index} (Name: {ref_name})"
            )
        except ValueError:
            raise ValueError(
                f"Reference SMILES '{args.ref_smiles}' not found in the input list."
            )

        VI = np.linalg.inv(np.cov(latent, rowvar=False))
        print("[INFO] Computing distances from reference embedding …")
        dists = {
            "euclidean": [distance.euclidean(ref_embedding, z) for z in latent],
            "cosine": [distance.cosine(ref_embedding, z) for z in latent],
            "manhattan": [distance.cityblock(ref_embedding, z) for z in latent],
            "correlation": [distance.correlation(ref_embedding, z) for z in latent],
            "mahalanobis": [distance.mahalanobis(ref_embedding, z, VI) for z in latent],
            "chebyshev": [distance.chebyshev(ref_embedding, z) for z in latent],
        }

        df = pd.DataFrame(
            {
                "Name": names,
                "SMILES": valid_smiles,
                "PAPER": paper_vals,
                "logP": logp_vals,
                **dists,
            }
        )
        out_csv = save_dir / "distance_polymer2.csv"
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Distance table saved to {out_csv.as_posix()}")

    print("[INFO] Running UMAP …")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=train_args.seed,
    )
    embedding = reducer.fit_transform(latent)

    # --- NUOVA SEZIONE PLOTTING ---
    # Crea un DataFrame per il plotting, che è il modo migliore per usare Seaborn
    plot_df = pd.DataFrame(
        {
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            "logP": logp_vals,
            "PAPER": paper_vals,
            "Name": names,
            "SMILES": valid_smiles,
        }
    )
    # Aggiungi una colonna descrittiva per la legenda
    plot_df["Origin"] = plot_df["PAPER"].apply(
        lambda x: "In Paper" if x == 1 else "Not In Paper"
    )

    # Crea e salva il primo plot (colorato per logP)
    create_publication_plot(
        plot_df=plot_df,
        save_dir=save_dir,
        plot_type="logp",
        annotate=args.annotate,
        ref_name=ref_name,
    )

    # Crea e salva il secondo plot (colorato per status)
    create_publication_plot(
        plot_df=plot_df,
        save_dir=save_dir,
        plot_type="paper_status",
        annotate=args.annotate,
        ref_name=ref_name,
    )


def parse_cli() -> argparse.Namespace:
    # ... (il resto del codice rimane invariato)
    parser = argparse.ArgumentParser(
        description="Encode SMILES and generate UMAP plots."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/model_preCondRNN_v4.2",
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--smiles_file",
        type=str,
        default="dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore_scaling_metrics.json",
        help="Path to the scaling metrics JSON file.",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=5, help="UMAP n_neighbors parameter."
    )
    parser.add_argument(
        "--min_dist", type=float, default=0.3, help="UMAP min_dist parameter."
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with their names (use with caution).",
    )
    parser.add_argument(
        "--ref_smiles",
        type=str,
        default="CN(CCO)C(=O)C(c1ccccc1)c2ccccc2",
        help="Reference SMILES for distance computation and highlighting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_cli())
