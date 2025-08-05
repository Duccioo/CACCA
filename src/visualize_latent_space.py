# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.spatial import distance

# ---
from model.CAE import CVAE, ConditionalAutoencoder, OLDConditionalAutoencoder
from generatore2 import load_vocab

PAD, EOS, SOS = "_", "E", "X"  # special tokens expected by the model


def load_model_from_folder(folder, num_prop: int = 5, num_check="best") -> ConditionalAutoencoder:
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
            raise FileNotFoundError(f"No checkpoint ending with '{num_check}' found in {folder!s}")
    else:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    params = Path(folder) / "args.json"
    if params.exists():
        with open(params) as f:
            train_args = argparse.Namespace(**json.load(f))
    try:
        print(f"[INFO] Loading {train_args.model_type} model from {latest.as_posix()} …")
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
    smiles_df: pd.DataFrame,
    char2idx: dict[str, int],
    seq_length: int,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, np.ndarray, List[str], List[str], List[int]]:
    """Processa un DataFrame e restituisce tensori e metadati."""
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
            print(f"[WARN] Skipping SMILES with no recognised chars: {smi} (name: {name})")
            continue

        padded = [char2idx[SOS]] + token_ids + [char2idx[EOS]] * (seq_length - len(token_ids) - 1)
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
# NUOVA FUNZIONE DI PLOTTING PER PUBBLICAZIONI
###############################################################################


def create_umap_plot(
    data: pd.DataFrame,
    output_path: Path,
    annotate: bool = False,
    ref_name: str | None = None,
) -> None:
    """
    Crea e salva un grafico UMAP di alta qualità usando Seaborn.

    Args:
        data (pd.DataFrame): DataFrame contenente le coordinate UMAP e i metadati
                             per il plotting (es. 'UMAP-1', 'UMAP-2', 'logP', 'Name', 'In Paper').
        output_path (Path): Percorso dove salvare l'immagine del grafico.
        annotate (bool): Se True, annota i punti con i loro nomi.
        ref_name (str | None): Nome della molecola di riferimento da evidenziare.
    """
    # Imposta lo stile di Seaborn per una qualità da pubblicazione
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Crea la figura
    plt.figure(figsize=(10, 8))

    # Definisci una palette di colori e marcatori per una chiara distinzione
    palette = {"Yes": "#0173b2", "No": "#de8f05"}
    markers = {"Yes": "o", "No": "X"}

    # Crea lo scatter plot con Seaborn
    ax = sns.scatterplot(
        data=data,
        x="UMAP-1",
        y="UMAP-2",
        hue="In Paper",  # Colora in base alla colonna 'In Paper'
        style="In Paper", # Cambia marcatore in base alla colonna 'In Paper'
        palette=palette,
        markers=markers,
        s=100,  # Dimensione dei punti
        alpha=0.8,
        edgecolor="w",  # Bordo bianco per migliore separazione
        linewidth=0.5,
    )

    # Evidenzia la molecola di riferimento, se specificata
    if ref_name and ref_name in data["Name"].values:
        ref_point = data[data["Name"] == ref_name]
        plt.scatter(
            ref_point["UMAP-1"],
            ref_point["UMAP-2"],
            marker="*",
            s=400,
            facecolor="#d55e00", # Colore arancione scuro per la stella
            edgecolor="black",
            linewidth=1.5,
            label=f"Reference: {ref_name}",
            zorder=3, # Assicura che sia disegnato sopra gli altri punti
        )

    # Aggiungi annotazioni se richiesto (utile per pochi punti)
    if annotate:
        for i, point in data.iterrows():
            plt.text(
                point["UMAP-1"] + 0.05,
                point["UMAP-2"] + 0.05,
                point["Name"],
                fontsize=7,
                alpha=0.8,
            )

    # Imposta titoli e label
    plt.title("UMAP Projection of Molecular Latent Space", fontsize=16, weight="bold")
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)

    # Personalizza la leggenda
    handles, labels = ax.get_legend_handles_labels()
    
    # Aggiungi la label per la stella di riferimento, se presente
    if ref_name and ref_name in data["Name"].values:
        # Trova la label di riferimento e spostala alla fine se necessario
        ref_idx = [i for i, label in enumerate(labels) if label.startswith("Reference")]
        if ref_idx:
            idx = ref_idx[0]
            handles.append(handles.pop(idx))
            labels.append(labels.pop(idx))
            
    # Ricrea la leggenda con un titolo e una posizione migliori
    plt.legend(handles=handles, labels=labels, title="Molecule Status", loc="best", frameon=True, shadow=True)


    # Salva la figura con alta risoluzione
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Publication-quality plot saved to {output_path.as_posix()}")
    plt.close() # Chiude la figura per liberare memoria


###############################################################################
# Routine Principale Modificata
###############################################################################


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab, train_args = load_model_from_folder(args.save_dir)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 1. Pre‑process SMILES and encode
    # ------------------------------------------------------------------
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
                mean = torch.tensor(scale_metrics[0], device=device, dtype=torch.float32)
                std = torch.tensor(scale_metrics[1], device=device, dtype=torch.float32)
                c = (c - mean) / std

    with torch.no_grad():
        latent = model(x, c, l)[0].cpu().numpy()
        print(f"[INFO] Encoded {latent.shape[0]} molecules → latent_dim={latent.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Optional: compute distances from reference SMILES
    # ------------------------------------------------------------------
    ref_name = None
    if args.ref_smiles:
        try:
            ref_index = valid_smiles.index(args.ref_smiles)
            ref_embedding = latent[ref_index]
            ref_name = names[ref_index] # Ottieni il nome della molecola di riferimento
            print(f"[INFO] Reference SMILES '{args.ref_smiles}' (Name: {ref_name}) found at index {ref_index}")

            VI = np.linalg.inv(np.cov(latent, rowvar=False))
            dists = {
                "euclidean": [distance.euclidean(ref_embedding, z) for z in latent],
                "mahalanobis": [distance.mahalanobis(ref_embedding, z, VI) for z in latent],
            }
            df = pd.DataFrame({"Name": names, "SMILES": valid_smiles, "PAPER": paper_vals, "logP": logp_vals, **dists})
            out_csv = Path(args.save_dir) / "distance_polymer2.csv"
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Distance table saved to {out_csv.as_posix()}")

        except ValueError:
            print(f"[WARN] Reference SMILES '{args.ref_smiles}' not found. Skipping distance calculation and plot highlighting.")
        except np.linalg.LinAlgError:
            print("[WARN] Could not compute inverse covariance matrix (singular). Skipping Mahalanobis distance.")


    # ------------------------------------------------------------------
    # 3. UMAP dimensionality reduction
    # ------------------------------------------------------------------
    print("[INFO] Running UMAP …")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=train_args.seed)
    embedding = reducer.fit_transform(latent)

    # ------------------------------------------------------------------
    # 4. Preparazione dati e chiamata alla nuova funzione di plotting
    # ------------------------------------------------------------------
    # Crea un DataFrame per il plotting per una gestione più pulita dei dati
    plot_df = pd.DataFrame({
        "UMAP-1": embedding[:, 0],
        "UMAP-2": embedding[:, 1],
        "logP": logp_vals,
        "Name": names,
        # Crea una colonna più descrittiva per la legenda
        "In Paper": ["Yes" if p == 1 else "No" for p in paper_vals]
    })
    
    # Determina il nome del file di output e chiama la funzione di plotting
    out_path = Path(args.save_dir) / "umap_plot_publication.png"
    
    # Trova il nome del riferimento se lo SMILES è stato fornito
    ref_mol_name = None
    if args.ref_smiles:
        try:
            ref_idx = valid_smiles.index(args.ref_smiles)
            ref_mol_name = names[ref_idx]
        except ValueError:
            pass # Già gestito sopra

    create_umap_plot(
        data=plot_df,
        output_path=out_path,
        annotate=args.annotate,
        ref_name=ref_mol_name # Passa il nome della molecola di riferimento
    )


###############################################################################
# CLI
###############################################################################


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode molecules and generate a publication-quality UMAP plot.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/model_preCondRNN_v4.2",
        help="Directory containing model checkpoints and for saving outputs.",
    )
    parser.add_argument(
        "--smiles_file",
        type=str,
        default="dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore_scaling_metrics.json",
        help="JSON file with scaling metrics.",
    )
    parser.add_argument("--n_neighbors", type=int, default=5, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min_dist", type=float, default=0.3, help="UMAP min_dist parameter.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with molecule names.")
    parser.add_argument(
        "--ref_smiles",
        type=str,
        default="CN(CCO)C(=O)C(c1ccccc1)c2ccccc2",
        help="SMILES string to use as a reference for distance computation and highlighting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Esempio di come lanciare lo script (da riga di comando sarebbe senza `[]`)
    # args = parse_cli([])
    # run(args)
    run(parse_cli())