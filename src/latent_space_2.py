# latent_space_2
# -*- coding: utf-8 -*-
"""
visualize_latent_space_fixed.py
--------------------------------
Standalone script to project the latent space of a *ConditionalAutoencoder* 🧬
into 2‑D using UMAP, colour the points by logP **and compute distances from a reference molecule**.

New features added
~~~~~~~~~~~~~~~~~~
* **Distance calculation**: choose a reference SMILES (or default to the first one) and obtain
  the top‑k nearest neighbours according to one or more metrics (Euclidean, cosine, etc.).
  Results are printed and optionally saved to CSV.
* **CLI flags**
  • `--ref_smiles` / `--ref_idx` choose the reference molecule  
  • `--metrics` comma‑separated list of metrics understood by `scipy.spatial.distance.cdist`  
  • `--top_k` number of closest molecules to report  
  • `--dist_out` optional CSV file to store the distances

Run an example::

    python visualize_latent_space_fixed.py \
        --save_dir save/ \
        --smiles_file molecules.smi \
        --vocab_file vocab.npy \
        --ref_smiles "CCO" \
        --metrics euclidean,cosine \
        --top_k 15 \
        --dist_out neighbours.csv
"""
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

# distance metrics
try:
    from scipy.spatial import distance as ssd  # most metrics

    _HAVE_SCIPY = True
except ImportError:  # graceful fallback
    _HAVE_SCIPY = False
    print("[WARN] SciPy not found – only Euclidean distance available.")

# ----------------------------- local imports ----------------------------- #
from model_new import ConditionalAutoencoder  # noqa: E402
from generatore2 import load_vocab  # noqa: E402

###############################################################################
# Utility helpers
###############################################################################

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
    """Tokenise SMILES → tensors and compute molecular descriptors."""
    xs, cs, ls, valid, logp_vals = [], [], [], [], []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) >= seq_length - 2:
            print(f"[WARN] Skipping invalid or too‑long SMILES: {smi}")
            continue

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)

        logp_vals.append(logp)

        token_ids = [char2idx.get(ch) for ch in smi if ch in char2idx]
        if not token_ids:
            print(f"[WARN] Skipping SMILES with no recognised chars: {smi}")
            continue

        padded = [char2idx[SOS]] + token_ids + [char2idx[EOS]] * (seq_length - len(token_ids) - 1)
        xs.append(padded)
        cs.append([mw, logp, hbd, hba, tpsa])  # conditioning vector (5 props)
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


def _compute_distances(
    latent: np.ndarray,
    ref_idx: int,
    metrics: list[str],
) -> dict[str, np.ndarray]:
    """Return a dict {metric: distances} (length = N)."""
    ref_vec = latent[ref_idx : ref_idx + 1]
    d: dict[str, np.ndarray] = {}
    for m in metrics:
        if m == "euclidean" or not _HAVE_SCIPY:
            d[m] = np.linalg.norm(latent - ref_vec, axis=1)
        else:
            d[m] = ssd.cdist(ref_vec, latent, metric=m)[0]
    return d


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load or build vocabulary
    # ------------------------------------------------------------------
    if args.vocab_file:
        print(f"[INFO] Loading vocabulary from {args.vocab_file} …")
        char, vocab, int_to_char, pad_idx, sos_idx, eos_idx = load_vocab(args.vocab_file)
        char2idx = vocab
    else:
        print("[INFO] Building vocabulary from input SMILES …")
        char2idx, _ = build_vocab_from_smiles(args.smiles)

    vocab_size = len(char2idx)
    print(f"[INFO] Vocab size = {vocab_size}")

    # ------------------------------------------------------------------
    # 2. Build & load model
    # ------------------------------------------------------------------
    train_args = argparse.Namespace(
        latent_size=args.latent_size,
        unit_size=args.unit_size,
        n_rnn_layer=args.n_layers,
        num_prop=5,  # conditioning vector length
        batch_size=128,
        emb_size=args.emb_size,
    )

    model = ConditionalAutoencoder(vocab_size, train_args).to(device)

    ckpts = list(Path(args.save_dir).glob("model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.save_dir!s}")
    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
    print(f"[INFO] Loading weights from {latest.name}")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()

    # ------------------------------------------------------------------
    # 3. Pre‑process SMILES and encode
    # ------------------------------------------------------------------
    x, c, l, logp_vals, valid_smiles = smiles_to_tensor(args.smiles, char2idx, args.seq_length)
    x, c, l = x.to(device), c.to(device), l.to(device)

    with torch.no_grad():
        latent = model(x, c, l)[0].cpu().numpy()

    print(f"[INFO] Encoded {latent.shape[0]} molecules → latent_dim={latent.shape[1]}")

    # ------------------------------------------------------------------
    # 4. UMAP & plot
    # ------------------------------------------------------------------
    print("[INFO] Running UMAP …")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
    embedding = reducer.fit_transform(latent)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=logp_vals, cmap="viridis", s=70, alpha=0.9)
    plt.colorbar(scatter, label="logP")
    plt.title("2‑D UMAP projection of the latent space")
    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.grid(True)

    if args.annotate:
        for i, smi in enumerate(valid_smiles):
            plt.annotate(
                smi, (embedding[i, 0], embedding[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8
            )

    out_path = Path(args.out).with_suffix(".png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to {out_path.as_posix()}")

    # ------------------------------------------------------------------
    # 5. Distance calculation
    # ------------------------------------------------------------------
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        metrics = ["euclidean"]

    if args.ref_smiles:
        try:
            ref_idx = valid_smiles.index(args.ref_smiles)
        except ValueError:
            print(f"[WARN] Reference SMILES {args.ref_smiles} not found – defaulting to index 0")
            ref_idx = 0
    else:
        ref_idx = args.ref_idx if args.ref_idx is not None else 0

    print(f"[INFO] Using reference molecule #{ref_idx}: {valid_smiles[ref_idx]}")
    dist_dict = _compute_distances(latent, ref_idx, metrics)

    for m in metrics:
        order = np.argsort(dist_dict[m])
        print(f"\n=== Nearest neighbours by {m} (top {args.top_k}) ===")
        for rank in range(1, min(args.top_k + 1, len(order))):  # skip rank 0 (itself)
            idx = order[rank]
            print(f"{rank:2d}. dist={dist_dict[m][idx]:.4f} | {valid_smiles[idx]}")
    # optional CSV output
    if args.dist_out:
        import csv

        with open(args.dist_out, "w", newline="") as fw:
            writer = csv.writer(fw)
            header = ["smiles"] + [f"dist_{m}" for m in metrics]
            writer.writerow(header)
            for i, smi in enumerate(valid_smiles):
                row = [smi] + [dist_dict[m][i] for m in metrics]
                writer.writerow(row)
        print(f"[INFO] Distance matrix saved to {args.dist_out}")


###############################################################################
# CLI
###############################################################################


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the latent space of a ConditionalAutoencoder and compute distances"
    )

    parser.add_argument(
        "--save_dir", type=str, default="save", help="Directory containing model_*.pt checkpoints"
    )
    parser.add_argument(
        "--seq_length", type=int, default=120, help="Maximum sequence length used during training"
    )

    parser.add_argument(
        "--smiles", type=str, default="", help="Comma-separated list of SMILES strings to encode"
    )
    parser.add_argument(
        "--ref_smiles", type=str, default="", help="Reference SMILES string for distance computation"
    )
    parser.add_argument(
        "--ref_idx", type=int, default=None, help="Reference molecule index for distance computation"
    )
    parser.add_argument(
        "--metrics", type=str, default="euclidean", help="Comma-separated list of distance metrics"
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to report")
    parser.add_argument("--dist_out", type=str, default="", help="Output CSV file for distance matrix")
    parser.add_argument("--out", type=str, default="out", help="Output directory for plots")

    return parser.parse_args()
