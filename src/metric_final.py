# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import umap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


# ---
from model.CAE import ConditionalAutoencoder, OLDConditionalAutoencoder, CVAE
from generatore2 import load_vocab


PAD, EOS, SOS = "_", "E", "X"  # special tokens expected by the model


def load_model_from_folder(
    folder, num_prop: int = 5, num_check="best"
) -> ConditionalAutoencoder:
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


def calculate_ef_at_k(
    pairwise_dists: np.ndarray, labels: np.ndarray, ref_idx: int, k: int
) -> float:
    """Calcola l'Enrichment Factor (EF@k)"""
    ref_distances = pairwise_dists[ref_idx]

    # Escludi il riferimento stesso ordinando le distanze
    sorted_indices = np.argsort(ref_distances)[1:]  # [0] è il rif. stesso

    top_k_indices = sorted_indices[:k]

    print("////////////////////////////////////")
    print(np.argsort(ref_distances))
    print(ref_distances[top_k_indices])
    print(labels[top_k_indices])
    print("////////////////////////////////////")

    # n+ è il numero totale di adsorbati verificati (label 1)
    n_plus = np.sum(labels == 1)
    # M è il numero totale di molecole (escluso il riferimento)
    M = len(labels) - 1

    if n_plus == 0 or M == 0:
        return 0.0

    # Hits@k: numero di adsorbati verificati tra i top k
    hits_at_k = np.sum(labels[top_k_indices] == 1)

    return (hits_at_k / k) / (n_plus / M) if k > 0 and n_plus > 0 else 0.0


def calculate_pp_hit_at_k(
    pairwise_dists: np.ndarray, labels: np.ndarray, k: int
) -> float:
    """Calcola il Positive-Positive hit (PP-hit@k)"""
    positive_indices = np.where(labels == 1)[0]
    if len(positive_indices) < 2:
        return 0.0  # Non si può calcolare con meno di 2 positivi

    num_positives_with_neighbor = 0
    for i in positive_indices:
        # Ordina gli indici in base alla distanza dal punto i
        # Escludi il punto stesso ([1:])
        sorted_indices = np.argsort(pairwise_dists[i])[1:]

        # Prendi i k vicini più prossimi
        k_nearest_neighbors = sorted_indices[:k]

        # Controlla se almeno uno dei vicini è anch'esso positivo
        if any(labels[neighbor_idx] == 1 for neighbor_idx in k_nearest_neighbors):
            num_positives_with_neighbor += 1

    return num_positives_with_neighbor / len(positive_indices)


def calculate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calcola lo Silhouette Score solo sui dati etichettati (0 e 1)"""
    labeled_mask = (labels == 0) | (labels == 1)
    if np.sum(labeled_mask) < 2 or len(np.unique(labels[labeled_mask])) < 2:
        return np.nan  # Non calcolabile se non ci sono almeno 2 campioni o 2 classi

    labeled_embeddings = embeddings[labeled_mask]
    labeled_labels = labels[labeled_mask]

    return silhouette_score(labeled_embeddings, labeled_labels)


# ---
# Nuove funzioni per la generazione degli embedding
# ---


def get_rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    """Calcola i 5 descrittori RDKit per una lista di SMILES."""
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # In caso di SMILES non valido, ritorna un vettore di zeri
            descriptors.append(np.zeros(5))
            continue
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        mw = Descriptors.ExactMolWt(mol)
        descriptors.append([logp, tpsa, hbd, hba, mw])
    return np.array(descriptors)


def get_morgan_fingerprints(smiles_list: List[str], n_bits: int = 2048) -> np.ndarray:
    """Calcola i Morgan Fingerprints per una lista di SMILES."""
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp = np.zeros(n_bits, dtype=int)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        fingerprints.append(fp)
    return np.array(fingerprints)


def smiles_dict_to_tensor(
    smiles_df: pd.DataFrame,  # MODIFICA: Accetta un DataFrame
    char2idx: dict[str, int],
    seq_length: int,
) -> Tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    np.ndarray,
    List[str],
    List[str],
    List[int],
]:  # MODIFICA: Aggiunto List[int] per i valori di 'PAPER'
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
        ls.append(len(token_ids) + 1)  # +1 for SOS
        valid_smiles.append(smi)
        names.append(name)
        paper_vals.append(paper_status)  # MODIFICA: Salva il valore di PAPER

    if not xs:
        raise RuntimeError("No valid SMILES to process – aborting.")

    return (
        torch.as_tensor(xs, dtype=torch.long),
        torch.tensor(cs, dtype=torch.float32),
        torch.as_tensor(ls, dtype=torch.long),
        np.array(logp_vals, dtype=np.float32),
        valid_smiles,
        names,
        paper_vals,  # MODIFICA: Restituisce i valori di PAPER
    )


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
        palette="deep",
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

    return f"umap_plot_{filename_suffix}_test_0.png"


def run_analysis():
    """Funzione principale che esegue l'intera analisi."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 1. Caricamento e Preparazione Dati ---
    print("[INFO] 1. Caricamento e preparazione dei dati...")
    try:
        molecules_df = pd.read_csv("assets/mol_test.csv")
    except FileNotFoundError:
        print("[ERRORE] File 'assets/mol_test.csv' non trovato. Assicurati che esista.")
        return

    # Mappatura della colonna PAPER a etichette numeriche
    # Assumiamo: 1 -> adsorbato (positivo), 0 -> non-adsorbato (negativo), altro -> non etichettato (-1)
    # Questa mappatura è cruciale per le metriche.
    molecules_df["LABEL"] = molecules_df["PAPER"].apply(lambda x: 1 if x == 1 else 0)

    # Pulisci gli SMILES da spazi e virgolette che possono causare errori di parsing
    molecules_df["SMILES"] = molecules_df["SMILES"].str.strip().str.strip('"')

    valid_smiles = molecules_df["SMILES"].tolist()
    names = molecules_df["MOL"].tolist()
    labels = molecules_df["LABEL"].to_numpy()

    # Identifica l'indice del polimero di riferimento
    ref_name = "PolyPhOx"
    ref_idx = names.index(ref_name)
    print(f"[INFO] Polimero di riferimento '{ref_name}' trovato all'indice {ref_idx}.")

    test_name = "Fluoxetine"
    test_idx = names.index(test_name)
    print(f"[INFO] Polimero di test '{test_name}' trovato all'indice {test_idx}.")

    test_name2 = "Pyramidone"
    test_idx2 = names.index(test_name2)
    print(f"[INFO] Polimero di test '{test_name2}' trovato all'indice {test_idx2}.")

    test_name3 = "Ibuprofen"
    test_idx3 = names.index(test_name3)
    print(f"[INFO] Polimero di test '{test_name3}' trovato all'indice {test_idx3}.")

    # --- 2. Definizione e Generazione degli Embedding ---
    print("\n[INFO] 2. Generazione degli embedding con vari metodi...")

    embeddings_collection = {}

    # Metodo A: ConditionalAutoencoder (CAE)
    print("\n--- Metodo: ConditionalAutoencoder (CAE) ---")
    # Usa la tua logica per caricare il modello e ottenere i dati
    # Per questo esempio, useremo un placeholder
    model_folder = "saved_models/model_preCondRNN_v4.2"
    cae_model, vocab, train_args = load_model_from_folder(model_folder)
    cae_model.to(device)

    x, c, l, logp_vals, valid_smiles, names, paper_vals = smiles_dict_to_tensor(
        molecules_df, vocab, train_args.max_seq_length
    )

    # load dataset scale metric from json file
    scale_file = Path(
        "dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore_scaling_metrics.json"
    )

    x, c, l = x.to(device), c.to(device), l.to(device)
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

    x, c, l = x.to(device), c.to(device), l.to(device)

    with torch.no_grad():
        cae_latent = cae_model(x, c, l)[0].cpu().numpy()
    embeddings_collection["CAE"] = cae_latent

    # usa umap per ridurre la dimensionalità degli embedding CAE
    umap_cae = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.1, random_state=42)
    cae_latent = umap_cae.fit_transform(cae_latent)
    embeddings_collection["UMAP_CAE"] = cae_latent

    print(f"[OK] Generati embedding CAE di forma: {cae_latent.shape}")

    # Preparazione per metodi classici
    descriptors = get_rdkit_descriptors(valid_smiles)
    fingerprints = get_morgan_fingerprints(valid_smiles, n_bits=1024)

    # Standardizza i dati prima di PCA/UMAP
    scaler_desc = StandardScaler()
    scaled_descriptors = scaler_desc.fit_transform(descriptors)

    # Metodo B: PCA su Descrittori
    print("\n--- Metodo: PCA su Descrittori Chimici ---")
    pca_desc = PCA(n_components=2)  # 5 descrittori -> 5 componenti
    embeddings_collection["PCA_Descriptors"] = pca_desc.fit_transform(
        scaled_descriptors
    )
    print(
        f"[OK] Generati embedding di forma: {embeddings_collection['PCA_Descriptors'].shape}"
    )

    # Metodo C: UMAP su Descrittori
    print("\n--- Metodo: UMAP su Descrittori Chimici ---")
    umap_desc = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3, random_state=42)
    embeddings_collection["UMAP_Descriptors"] = umap_desc.fit_transform(
        scaled_descriptors
    )
    print(
        f"[OK] Generati embedding di forma: {embeddings_collection['UMAP_Descriptors'].shape}"
    )

    # Metodo D: PCA Misto (Fingerprints + Descrittori)
    print("\n--- Metodo: PCA Misto (FP ridotti + Descrittori) ---")
    pca_fp = PCA(n_components=10, random_state=42)
    reduced_fp = pca_fp.fit_transform(fingerprints)
    mixed_features = np.concatenate([scaled_descriptors, reduced_fp], axis=1)
    pca_mixed = PCA(n_components=2, random_state=42)  # Riduciamo le 15 features finali
    embeddings_collection["PCA_Mixed"] = pca_mixed.fit_transform(mixed_features)
    print(
        f"[OK] Generati embedding di forma: {embeddings_collection['PCA_Mixed'].shape}"
    )

    # Metodo E: UMAP Misto (Fingerprints + Descrittori)
    print("\n--- Metodo: UMAP Misto (FP ridotti + Descrittori) ---")
    # Usiamo gli stessi dati misti del metodo D
    umap_mixed = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.1, random_state=42)
    umap_mixed_embeddings = umap_mixed.fit_transform(mixed_features)
    embeddings_collection["UMAP_Mixed"] = umap_mixed_embeddings

    # Crea un DataFrame per il plotting
    umap_plot_df = pd.DataFrame(umap_mixed_embeddings, columns=["UMAP1", "UMAP2"])
    umap_plot_df["MOL"] = names
    umap_plot_df["PAPER"] = molecules_df[
        "PAPER"
    ]  # Assumendo che l'ordine sia lo stesso

    create_umap_plot(umap_plot_df, "Misto (FP + Descrittori)", "mixed")
    print(
        f"[OK] Generati embedding di forma: {embeddings_collection['UMAP_Mixed'].shape}"
    )

    # --- 3. Calcolo Metriche e Salvataggio Risultati ---
    print("\n[INFO] 3. Calcolo delle metriche per ogni metodo...")

    results = []
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    k_values_ef = [3, 5, 10]
    k_values_pp = [1, 3, 5]

    for method_name, embeddings in embeddings_collection.items():
        print(f"\n--- Valutazione per: {method_name} ---")

        # Salva gli embedding
        embedding_df = pd.DataFrame(embeddings, index=names)
        embedding_filename = output_dir / f"embeddings_{method_name}.csv"
        embedding_df.to_csv(embedding_filename)
        print(f"  [SAVE] Embedding salvati in: {embedding_filename}")

        # Calcola la matrice delle distanze pairwise
        pairwise_dists = cdist(embeddings, embeddings, metric="euclidean")
        print(f"  [INFO] Matrice delle distanze pairwise calcolata per {method_name}.")

        # Salva le distanze dal polimero di riferimento
        distances_from_ref = pairwise_dists[ref_idx]
        distance_df = pd.DataFrame(
            {
                "Name": names,
                "SMILES": valid_smiles,
                f"Distance_from_{ref_name}": distances_from_ref,
            }
        )
        distance_df = distance_df.sort_values(
            by=f"Distance_from_{ref_name}", ascending=True
        )

        distance_filename = output_dir / f"distances_{method_name}.csv"
        distance_df.to_csv(distance_filename, index=False)
        print(f"  [SAVE] Distanze salvate in: {distance_filename}")

        print("-----------------------------------")
        # Calcola le metriche
        method_results = {"Method": method_name}

        for k in k_values_ef:
            ef_score = calculate_ef_at_k(pairwise_dists, labels, ref_idx, k)
            method_results[f"EF@{k}"] = ef_score
            print(f"  [METRIC] EF@{k}: {ef_score:.4f}")

        for k in k_values_pp:
            pp_score = calculate_pp_hit_at_k(pairwise_dists, labels, k)
            method_results[f"PP-hit@{k}"] = pp_score
            print(f"  [METRIC] PP-hit@{k}: {pp_score:.4f}")

        silhouette = calculate_silhouette(embeddings, labels)
        method_results["Silhouette"] = silhouette
        print(f"  [METRIC] Silhouette Score: {silhouette:.4f}")

        results.append(method_results)

    # --- 4. Report Finale ---
    print("\n[INFO] 4. Report riassuntivo finale.")
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Method")

    # Stampa a schermo
    print("\n" + "=" * 80)
    print("RISULTATI FINALI DI CONFRONTO")
    print("=" * 80)
    print(results_df.to_string(float_format="%.4f"))
    print("=" * 80)

    # Salva il report finale in un CSV
    report_filename = output_dir / "metrics_summary_report.csv"
    results_df.to_csv(report_filename)
    print(f"\n[SUCCESS] Analisi completata. Report salvato in: {report_filename}")


if __name__ == "__main__":
    # In questa versione, gli argomenti da linea di comando sono stati rimossi
    # per semplicità e per focalizzarsi sulla logica principale.
    # Si possono reintegrare se necessario.
    run_analysis()
