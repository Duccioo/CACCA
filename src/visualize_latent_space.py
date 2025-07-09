# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import json

import pandas as pd
from scipy.spatial import distance


# ---
from model.CAE import ConditionalAutoencoder , OLDConditionalAutoencoder
from generatore2 import load_vocab


PAD, EOS, SOS = "_", "E", "X"  # special tokens expected by the model


def load_model_from_folder(folder, num_prop: int = 5, num_check = "best") -> ConditionalAutoencoder:
    """Load the latest model checkpoint from a given folder."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    # load vocab.pkl file
    vocab_file = Path(folder) / "vocab.pkl"
    print(f"Loading vocabulary from {vocab_file} …")
    if vocab_file.exists():
        char, vocab, int_to_char, pad_idx, sos_idx, eos_idx = load_vocab(vocab_file)
        
    ckpts = [p for p in Path(folder).glob("model_*.pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {folder!s}")
    # Cerca un file che termina con 'best.pt'
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
        print(f"[INFO] Loading model from {latest.as_posix()} …")
        model = ConditionalAutoencoder(len(vocab), num_prop, train_args.latent_size, train_args.emb_size, train_args.hidden_size, train_args.n_rnn_layer)
        model.load_state_dict(torch.load(latest, map_location=torch.device("cpu")))
    except:
        model = OLDConditionalAutoencoder(len(vocab), num_prop, train_args.latent_size, train_args.emb_size, train_args.hidden_size, train_args.n_rnn_layer)
        model.load_state_dict(torch.load(latest, map_location=torch.device("cpu")))
    return model, vocab, train_args


# Sostituisci anche questa funzione
def smiles_dict_to_tensor(
    smiles_df: pd.DataFrame, # MODIFICA: Accetta un DataFrame
    char2idx: dict[str, int],
    seq_length: int,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, np.ndarray, List[str], List[str], List[int]]: # MODIFICA: Aggiunto List[int] per i valori di 'PAPER'
    """
    MODIFICA: Processa un DataFrame invece di un dizionario.
    Restituisce anche i valori della colonna 'PAPER'.
    """
    xs, cs, ls, valid_smiles, logp_vals, names, paper_vals = [], [], [], [], [], [], []

    # MODIFICA: Itera sulle righe del DataFrame
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
        ls.append(len(token_ids) + 1)  # +1 for SOS
        valid_smiles.append(smi)
        names.append(name)
        paper_vals.append(paper_status) # MODIFICA: Salva il valore di PAPER

    if not xs:
        raise RuntimeError("No valid SMILES to process – aborting.")

    return (
        torch.as_tensor(xs, dtype=torch.long),
        torch.tensor(cs, dtype=torch.float32),
        torch.as_tensor(ls, dtype=torch.long),
        np.array(logp_vals, dtype=np.float32),
        valid_smiles,
        names,
        paper_vals, # MODIFICA: Restituisce i valori di PAPER
    )

# Sostituisci questa funzione
def load_smiles_from_csv(file_path: str, filter_not_in_paper: bool=False) -> pd.DataFrame:
    """
    MODIFICA: Ora restituisce un DataFrame di pandas invece di un dizionario.
    Questo permette di mantenere tutte le informazioni necessarie (MOL, SMILES, PAPER).
    """
    smiles_file = Path(file_path)
    df = pd.read_csv(smiles_file, skipinitialspace=True)

    if filter_not_in_paper:
        df = df[df['PAPER'] == 1].copy() # .copy() per evitare SettingWithCopyWarning
    
    return df


###############################################################################
# Main routine
###############################################################################

# E infine, sostituisci la funzione principale run()
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab, train_args = load_model_from_folder(args.save_dir)
    model.to(device)
    model.eval()  # set to evaluation mode

    
    # ------------------------------------------------------------------
    # 3. Pre‑process SMILES and encode
    # ------------------------------------------------------------------
    # MODIFICA: Ora carichiamo il DataFrame
    molecules_df = load_smiles_from_csv("assets/mol_test.csv")
    # MODIFICA: Passiamo il DataFrame e riceviamo anche i valori di 'paper'
    x, c, l, logp_vals, valid_smiles, names, paper_vals = smiles_dict_to_tensor(molecules_df, vocab, train_args.max_seq_length)
    x, c, l = x.to(device), c.to(device), l.to(device)
    
    
    # load dataset scale metric from json file
    scale_file = Path(args.smiles_file)
    if scale_file.exists():
        with open(scale_file) as f:
            scale_data = json.load(f)
            print(f"[INFO] Loaded scaling metrics from {scale_file.as_posix()}")
            if scale_data["Scaling method"] == "zscore":
                scale_metrics = scale_data['Scale metrics']
                mean = torch.tensor(scale_metrics[0], device=device, dtype=torch.float32)
                std = torch.tensor(scale_metrics[1], device=device, dtype=torch.float32)
                c = (c - mean) / std

    with torch.no_grad():
        latent = model(x, c, l)[0].cpu().numpy()
        print(f"[INFO] Encoded {latent.shape[0]} molecules → latent_dim={latent.shape[1]}")
    
    
    # ------------------------------------------------------------------
    # 4. Optional: compute distances from reference SMILES
    # ------------------------------------------------------------------
    if args.ref_smiles:
        try:
            ref_index = valid_smiles.index(args.ref_smiles)
            ref_embedding = latent[ref_index]
            print(f"[INFO] Reference SMILES found at index {ref_index}")
        except ValueError:
            raise ValueError(f"Reference SMILES '{args.ref_smiles}' not found in the input list.")

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
        
        # MODIFICA 1: Aggiunta della colonna 'PAPER' al DataFrame di output
        df = pd.DataFrame(
            {
                "Name": names,
                "SMILES": valid_smiles,
                "PAPER": paper_vals, # Ecco la nuova colonna
                "logP": logp_vals,
                "euclidean": dists["euclidean"],
                "cosine": dists["cosine"],
                "manhattan": dists["manhattan"],
                "correlation": dists["correlation"],
                "mahalanobis": dists["mahalanobis"],
                "chebyshev": dists["chebyshev"],
            }
        )
        out_csv = Path(args.save_dir) / "distance_polymer2.csv"
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Distance table saved to {out_csv.as_posix()}")

    # ------------------------------------------------------------------
    # 4. UMAP dimensionality reduction
    # ------------------------------------------------------------------
    print("[INFO] Running UMAP …")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=train_args.seed)
    embedding = reducer.fit_transform(latent)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=logp_vals, cmap="viridis", s=70, alpha=0.9)
    plt.colorbar(scatter, label="logP")
    plt.title("2‑D UMAP projection of the latent space")
    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.grid(True)

    if args.annotate:
        # MODIFICA 2: Annotazione con il nome ('MOL') invece che con lo SMILES
        for i, name in enumerate(names):
            plt.annotate(
                name, (embedding[i, 0], embedding[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8
            )
    name_file = Path(args.save_dir) / "umap_plot.png"
    out_path = name_file
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to {out_path.as_posix()}")


###############################################################################
# CLI
###############################################################################


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir", type=str, default="saved_models/model_preCondRNN_v4.2", help="Directory containing model_*.pt checkpoints"
    )

    parser.add_argument(
        "--smiles_file",
        type=str,
        default="dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore_scaling_metrics.json",
        help="CSV file with SMILES and name",
    )
    
    parser.add_argument("--n_neighbors", type=int, default=5, help="UMAP n_neighbors parameter")
    parser.add_argument("--min_dist", type=float, default=0.3, help="UMAP min_dist parameter")

    parser.add_argument("--annotate", action="store_true", help="Draw the SMILES string next to every point")

    parser.add_argument(
        "--ref_smiles",
        type=str,
        default="OCCN(C(=O)C(c1ccccc1)c1ccccc1)",
        help="SMILES string to use as reference for distance computation",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load SMILES list (either built‑in examples or from file)
    # ------------------------------------------------------------------
 
    return args


if __name__ == "__main__":
    run(parse_cli())
