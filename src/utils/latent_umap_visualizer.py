#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
latent_umap_visualizer.py
-------------------------------------
Visualizza in 2‑D/3‑D lo spazio latente di un Autoencoder condizionale
utilizzando UMAP e opzionalmente esegue un clustering K‑means **sullo
spazio latente originale** (non sulla proiezione UMAP).

Dipendenze principali:
  * torch
  * numpy
  * umap-learn
  * matplotlib
  * rdkit-pypi
  * pandas
  * scipy

Esempio di utilizzo:
    python latent_umap_visualizer.py \
        --smiles-file data/smiles.txt \
        --save-dir checkpoints/ \
        --seq-length 128 \
        --latent-size 256 \
        --unit-size 512 \
        --n-layers 3 \
        --emb-size 64 \
        --n-neighbors 15 \
        --min-dist 0.10 \
        --annotate \
        --out results/umap_latent

Il file PNG del grafico verrà salvato in ``results/umap_latent.png`` e,
se fornito ``--ref-smiles``, viene generato anche un CSV con le distanze
metriche dal riferimento.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import pickle

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
from scipy.spatial import distance

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# -------------------------------------------------------------------------
from model.CAE import ConditionalAutoencoder  # noqa: E402
from generatore2 import load_vocab
# -------------------------------------------------------------------------

# Token speciali attesi dal modello (adatta se necessario)
PAD, EOS, SOS = "_", "E", "X"

# -------------------------------------------------------------------------
# Funzioni di utilità
# -------------------------------------------------------------------------
def build_vocab_from_smiles(smiles_list: List[str]) -> Tuple[dict[str, int], dict[int, str]]:
    special = [PAD, EOS, SOS]
    chars = sorted({c for s in smiles_list for c in s if c not in special})
    full = special + chars
    char2idx = {c: i for i, c in enumerate(full)}
    idx2char = {i: c for i, c in enumerate(full)}
    return char2idx, idx2char

def load_vocab_from_file(save_dir: Path):
    vocab_path = Path(save_dir) / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"{vocab_path} non trovato – hai salvato il vocab durante il training?")
    with vocab_path.open("rb") as f:
        data = pickle.load(f)
    char = data["char"]
    vocab = data["vocab"]
    int_to_char = {i: c for c, i in vocab.items()}
    pad_idx = vocab["_"]
    sos_idx = vocab["X"]
    eos_idx = vocab["E"]
    return vocab, int_to_char, pad_idx, sos_idx, eos_idx

def smiles_to_tensor(smiles_list: List[str], char2idx: dict[str, int], seq_length: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, np.ndarray, List[str]]:
    xs, cs, ls, valid, logp_vals = [], [], [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) >= seq_length - 2:
            print(f"[WARN] Scarto SMILES non valida o troppo lunga: {smi}")
            continue
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp_vals.append(logp)
        token_ids = [char2idx.get(ch) for ch in smi if ch in char2idx]
        if not token_ids:
            print(f"[WARN] Nessun carattere riconosciuto in: {smi}")
            continue
        padded = [char2idx[SOS]] + token_ids + [char2idx[EOS]] * (seq_length - len(token_ids) - 1)
        xs.append(padded)
        cs.append([mw, logp, hbd, hba, tpsa])
        ls.append(len(token_ids) + 1)
        valid.append(smi)
    if not xs:
        raise RuntimeError("Nessuno SMILES valido – operazione abortita.")
    return (
        torch.as_tensor(xs, dtype=torch.long),
        torch.tensor(cs, dtype=torch.float32),
        torch.as_tensor(ls, dtype=torch.long),
        np.array(logp_vals, dtype=np.float32),
        valid,
    )

def kmeans_clustering(embedding: np.ndarray, smiles: list[str], k: int = 2, out_prefix="cluster_kmeans") -> None:
    from sklearn.cluster import KMeans
    if len(embedding) != len(smiles):
        raise ValueError("embedding e smiles devono avere la stessa lunghezza")
    print(f"[INFO] Eseguo K-means con k={k} su {len(smiles)} molecole …")
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(embedding)
    df = pd.DataFrame({"SMILES": smiles, "cluster": labels})
    df.to_csv(f"{out_prefix}.csv", index=False)
    print(f"[INFO] Clustering salvato in {out_prefix}.csv")
    if embedding.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=70, edgecolors="k")
        plt.title(f"Cluster K-means (k={k}) nello spazio latente")
        plt.xlabel("Latente‑1")
        plt.ylabel("Latente‑2")
        plt.grid(True)
        plt.colorbar(scatter, label="Cluster")
        plt.savefig(f"{out_prefix}_2d.png", dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot salvato in {out_prefix}_2d.png")
        plt.close()

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device rilevato: {device}")
    if args.vocab_file:
        print(f"[INFO] Carico vocabolario da {args.vocab_file} …")
        char2idx, idx2char, pad_idx, sos_idx, eos_idx = load_vocab_from_file(args.vocab_file)
    else:
        print("[INFO] Costruisco vocabolario a partire dagli SMILES input …")
        char2idx, idx2char = build_vocab_from_smiles(args.smiles)
        pad_idx, sos_idx, eos_idx = char2idx[PAD], char2idx[SOS], char2idx[EOS]
    vocab_size = len(char2idx)
    print(f"[INFO] Dimensione vocabolario = {vocab_size}")
    train_args = argparse.Namespace(
        latent_size=args.latent_size,
        unit_size=args.unit_size,
        n_rnn_layer=args.n_layers,
        num_prop=5,
        batch_size=128,
        emb_size=args.emb_size,
    )
    model = ConditionalAutoencoder(vocab_size, train_args).to(device)
    ckpts = sorted(Path(args.save_dir).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"Nessun checkpoint trovato in {args.save_dir!s}")
    latest = ckpts[-1]
    print(f"[INFO] Carico pesi da {latest.name}")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()
    x, c, l, logp_vals, valid_smiles = smiles_to_tensor(args.smiles, char2idx, args.seq_length)
    x, c, l = x.to(device), c.to(device), l.to(device)
    if args.smiles_file.suffix == ".csv":
        df = pd.read_csv(args.smiles_file, sep=";")
        smiles_to_label = dict(zip(df["SMILES"], df.get("Type", "Unknown")))
        labels = [smiles_to_label.get(smi, "Unknown") for smi in valid_smiles]
    else:
        labels = ["Unlabeled"] * len(valid_smiles)
    with torch.no_grad():
        latent = model(x, c, l)[0].cpu().numpy()
        print(f"[INFO] Encodate {latent.shape[0]} molecole → dim_latente={latent.shape[1]}")
    if args.ref_smiles:
        try:
            ref_index = valid_smiles.index(args.ref_smiles)
            ref_embedding = latent[ref_index]
            print(f"[INFO] SMILES di riferimento trovato all'indice {ref_index}")
        except ValueError as e:
            raise ValueError(f"Lo SMILES di riferimento '{args.ref_smiles}' non è nella lista di input.") from e
        print("[INFO] Calcolo distanze dal riferimento …")
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
            "manhattan": dists["manhattan"],
        })
        out_csv = Path(args.out).with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Tabella distanze salvata in {out_csv.as_posix()}")
    print("[INFO] Lancio UMAP …")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=3, random_state=42)
    embedding = reducer.fit_transform(latent)
    kmeans_clustering(latent, valid_smiles)
    
    from matplotlib.lines import Line2D
    unique_labels = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    colors = [label_to_idx[lab] for lab in labels]
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap="tab20",
        s=70,
        alpha=0.9,
        edgecolors="k",
    )
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
            markerfacecolor=plt.cm.tab20(label_to_idx[lab] / len(unique_labels)),
            markeredgecolor='k', label=lab)
        for lab in unique_labels
    ]
    plt.legend(handles=handles, title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.colorbar(scatter, label="logP")
    plt.title("Proiezione 2‑D dello spazio latente (UMAP)")
    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if args.annotate:
        for i, smi in enumerate(valid_smiles):
            plt.annotate(smi, (embedding[i, 0], embedding[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    out_path = Path(args.out).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Grafico salvato in {out_path.as_posix()}")
    plt.close()
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=logp_vals, cmap="viridis", s=70, alpha=0.9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    plt.colorbar(sc, label="logP")
    plt.title("Proiezione 3D dello spazio latente (UMAP)")
    plt.savefig(Path(args.out).with_suffix(".3d.png"), dpi=300, bbox_inches="tight")
    plt.close()
    import plotly.express as px
    fig = px.scatter_3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        color=labels,
        hover_name=valid_smiles,
        title="Latent space UMAP 3D (cluster per Type)",
    )
    fig.write_html("umap_3d_by_type.html")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualizza lo spazio latente di un autoencoder con UMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles-file", type=Path, default='/mnt/beegfs/home/giulio/calamita/Database_drugs.csv', help="Percorso al file di testo con uno SMILES per riga.")
    group.add_argument("--smiles", nargs="+", help="Lista di SMILES passata direttamente da CLI.")
    p.add_argument("--vocab-file", type=Path, default='/mnt/beegfs/home/giulio/calamita/', help="JSON con vocabolario char→index.")
    p.add_argument("--save-dir", type=Path, required=True, default='/mnt/beegfs/home/giulio/calamita/', help="Directory con i checkpoint del modello.")
    p.add_argument("--seq-length", type=int, default=128, help="Lunghezza massima sequenza in input.")
    p.add_argument("--latent-size", type=int, default=200, help="Dimensione spazio latente.")
    p.add_argument("--unit-size", type=int, default=512, help="Dimensione unità GRU/LSTM.")
    p.add_argument("--n-layers", type=int, default=3, help="Numero layer RNN encoder/decoder.")
    p.add_argument("--emb-size", type=int, default=256, help="Dimensionalità embedding caratteri.")
    p.add_argument("--n-neighbors", type=int, default=15, help="Numero di vicini per UMAP.")
    p.add_argument("--min-dist", type=float, default=0.10, help="Parametro min_dist di UMAP.")
    p.add_argument("--annotate", action="store_true", help="Annota il punto con lo SMILES.")
    p.add_argument("--ref-smiles", help="SMILES di riferimento per il calcolo distanze.")
    p.add_argument("--out", type=Path, default=Path("latent_umap"), help="Prefisso output (senza estensione).")
    return p.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.smiles_file.suffix == ".csv":
        df = pd.read_csv(cli_args.smiles_file, sep=';')
        cli_args.smiles = df["SMILES"].dropna().astype(str).tolist()
    run(cli_args)
