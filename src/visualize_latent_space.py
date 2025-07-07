# -*- coding: utf-8 -*-
"""
visualize_latent_space_fixed.py
--------------------------------
Standalone script to project the latent space of a *ConditionalAutoencoder* 🧬
into 2‑D using UMAP and colour the points by logP.

Major improvements over the original version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Removed brittle external dependencies** (‵ZINC‵ dataset class and
   ‵generatore2.py‵).  You can now load a pre‑computed vocabulary from a
   plain text/NumPy file **or** build one on‑the‑fly from the SMILES list
   you want to visualise.
2. **Added missing model arguments** (the original script crashed because
   *emb_size* was not passed to the model constructor).
3. **Replaced the non‑existent `model.encode()` call** with the latent
   tensor returned by the model’s `forward` pass.
4. **Extra CLI flags** –‐ `--smiles_file` to read SMILES from a file and
   `--vocab_file` to reuse the vocabulary employed during training.
5. **Graceful error handling & progress messages** for an easier
   interactive experience.

Run it with something like::

    python visualize_latent_space_fixed.py \
        --save_dir my_checkpoints \
        --smiles_file example.smi \
        --vocab_file vocab.npy

or just rely on the default built‑in SMILES list for a quick smoke test.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ----------------------------- local imports ----------------------------- #
from model_new import ConditionalAutoencoder  # noqa: E402
from generatore2 import load_vocab
###############################################################################
# Utility helpers
###############################################################################

import pandas as pd
from scipy.spatial import distance


PAD, EOS, SOS = "_", "E", "X"  # special tokens expected by the model


def build_vocab_from_smiles(smiles_list: List[str]) -> Tuple[dict[str, int], dict[int, str]]:
    """Construct a minimal vocabulary directly from *smiles_list*.

    NOTE 📢  Use this **only** for quick visualisations.  It *will not*
    match the vocabulary the model was trained with, therefore generated
    reconstructions/decodings may be meaningless.
    """
    special = [PAD, EOS, SOS]
    chars = sorted({c for s in smiles_list for c in s if c not in special})
    full = special + chars
    char2idx = {c: i for i, c in enumerate(full)}
    idx2char = {i: c for i, c in enumerate(full)}
    return char2idx, idx2char

def smiles_to_tensor(
    smiles_list: List[str],
    char2idx: dict[str, int],
    seq_length: int,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, np.ndarray, List[str]]:
    """Tokenise SMILES → tensors and compute logP for colouring the plot."""
    xs, cs, ls, valid, logp_vals = [], [], [], [], []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) >= seq_length - 2:
            print(f"[WARN] Skipping invalid or too‑long SMILES: {smi}")
            continue

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba= rdMolDescriptors.CalcNumHBA(mol)
        tpsa= rdMolDescriptors.CalcTPSA(mol)


        logp_vals.append(logp)

        token_ids = [char2idx.get(ch) for ch in smi if ch in char2idx]
        if not token_ids:
            print(f"[WARN] Skipping SMILES with no recognised chars: {smi}")
            continue

        padded = [char2idx[SOS]] + token_ids + [char2idx[EOS]] * (seq_length - len(token_ids) - 1)
        xs.append(padded)
        #cs.append([logp])  # conditioning vector (shape: 1)
        cs.append([mw, logp, hbd, hba, tpsa])
        ls.append(len(token_ids) + 1)  # +1 for SOS
        valid.append(smi)

    if not xs:
        raise RuntimeError("No valid SMILES to process – aborting.")

    return (
        torch.as_tensor(xs, dtype=torch.long),
        torch.tensor(cs, dtype=torch.float32),
        torch.as_tensor(ls, dtype=torch.long),
        np.array(logp_vals, dtype=np.float32),
        valid,
    )

###############################################################################
# Main routine
###############################################################################

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load or build vocabulary
    # ------------------------------------------------------------------
    if args.vocab_file:
        print(f"[INFO] Loading vocabulary from {args.vocab_file} …")
        char, vocab, int_to_char, pad_idx, sos_idx, eos_idx = load_vocab(args.vocab_file)
    else:
        print("[INFO] Building vocabulary from input SMILES …")
        vocab, _ = build_vocab_from_smiles(args.smiles)

    vocab_size = len(vocab)
    print(f"[INFO] Vocab size = {vocab_size}")

    # ------------------------------------------------------------------
    # 2. Build & load model
    # ------------------------------------------------------------------
    # NOTE: you *must* instantiate the model with the same hyper‑parameters
    #       that were used during training.  Adjust if necessary.
    train_args = argparse.Namespace(
        latent_size=args.latent_size,
        unit_size=args.unit_size,
        n_rnn_layer=args.n_layers,
        num_prop=5,            # logP
        batch_size=128,
        emb_size=args.emb_size,
    )

    model = ConditionalAutoencoder(vocab_size, train_args).to(device)

    # automatically pick the checkpoint with the highest epoch number
    ckpts = [p for p in Path(args.save_dir).glob("model_*.pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.save_dir!s}")
    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
    print(f"[INFO] Loading weights from {latest.name}")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()

    # ------------------------------------------------------------------
    # 3. Pre‑process SMILES and encode
    # ------------------------------------------------------------------
    x, c, l, logp_vals, valid_smiles = smiles_to_tensor(args.smiles, vocab, args.seq_length)
    x, c, l = x.to(device), c.to(device), l.to(device)

    with torch.no_grad():
        # forward() returns (z, logits, dec_out)
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

        print("[INFO] Computing distances from reference embedding …")
        dists = {
            "euclidean": [distance.euclidean(ref_embedding, z) for z in latent],
            "cosine": [distance.cosine(ref_embedding, z) for z in latent],
            "manhattan": [distance.cityblock(ref_embedding, z) for z in latent],
        }

        df = pd.DataFrame({
            "SMILES": valid_smiles,
            "logP": logp_vals,
            "euclidean": dists["euclidean"],
            "cosine": dists["cosine"],
            "manhattan": dists["manhattan"]
        })
        out_csv = Path(args.out).with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Distance table saved to {out_csv.as_posix()}")


    print(f"[INFO] Encoded {latent.shape[0]} molecules → latent_dim={latent.shape[1]}")

    # ------------------------------------------------------------------
    # 4. UMAP dimensionality reduction
    # ------------------------------------------------------------------
    print("[INFO] Running UMAP …")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
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

    # annotate each point with its SMILES (optional – can get crowded!)
    if args.annotate:
        for i, smi in enumerate(valid_smiles):
            plt.annotate(smi, (embedding[i, 0], embedding[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    out_path = Path(args.out).with_suffix(".png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to {out_path.as_posix()}")

###############################################################################
# CLI
###############################################################################


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise the latent space of a ConditionalAutoencoder")

    parser.add_argument("--save_dir", type=str, default="save", help="Directory containing model_*.pt checkpoints")
    parser.add_argument("--seq_length", type=int, default=120, help="Maximum sequence length used during training")

    parser.add_argument("--smiles_file", type=str, default=None, help="Plain‑text file with one SMILES per line (overrides the built‑in list)")
    parser.add_argument("--vocab_file", type=str, default="save", help="NumPy .npy file with the vocabulary used during training")

    parser.add_argument("--latent_size", type=int, default=200)
    parser.add_argument("--unit_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--emb_size", type=int, default=256)

    parser.add_argument("--n_neighbors", type=int, default=5, help="UMAP n_neighbors parameter")
    parser.add_argument("--min_dist", type=float, default=0.3, help="UMAP min_dist parameter")

    parser.add_argument("--annotate", action="store_true", help="Draw the SMILES string next to every point")
    parser.add_argument("--out", type=str, default="latent_space_umap.png", help="Output image filename")
    parser.add_argument("--ref_smiles", type=str, default="OCCN(C(=O)C(c1ccccc1)c1ccccc1)", help="SMILES string to use as reference for distance computation")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load SMILES list (either built‑in examples or from file)
    # ------------------------------------------------------------------
    if args.smiles_file:
        with open(args.smiles_file) as f:
            args.smiles = [l.strip() for l in f if l.strip()]
    else:
        args.smiles = [  # quick demo list
            'CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl', #Metoclopramide
            'CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O', #Indomethacin
            "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                # Ibuprofen
            "c1ccccc1",                      # Benzene
            "O=C(O)c1ccccc1C(=O)O",          # Phthalic acid
            "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Chrysene
            "C1CCCCC1",                      # Cyclohexane
            "CCO",                           # Ethanol
            "CNC(=O)N(C)C",                  # Dimethylurea
            #"CC(=O)O",                       # Acetic acid
            #"C1=CC=C(C=C1)C(C(C(=O)O)N)C(C(=O)O)N",  # Example with heteroatoms
            "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O",  # Oestradiol
            "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)C",  # Pyramidone
            "CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C",
            "CS(=O)(=O)NC1=C(C=C(C=C1)[N+](=O)[O-])OC2=CC=CC=C2",
            "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
            

            #"C1=CC=C(C=C1)CC2=CC=CC=C2", 
           # "c1ccccc1C1NC=CO1", 
            "OCCN(C(=O)C(c1ccccc1)c1ccccc1)" # POLYMER (BOOOOOOH)
        ]
    return args


if __name__ == "__main__":
    run(parse_cli())
