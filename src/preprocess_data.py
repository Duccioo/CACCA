import csv
import multiprocessing as mp
import pathlib
from argparse import ArgumentParser
from typing import List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA

from tqdm import tqdm
import requests


ZINC_DOWNLOAD_URL = "https://github.com/jaechanglim/CVAE/blob/master/smiles.txt?raw=true"

# =============================================================================
# 1. FUNZIONI DI CALCOLO E SCALING (le tue, leggermente arricchite)
# =============================================================================


def calc_props(smiles: str) -> Optional[Tuple]:
    """Calcola 6 proprietà; ritorna None se lo SMILES non è valido."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Canonizzazione dello SMILES per avere una rappresentazione unica
        smiles_can = Chem.MolToSmiles(mol, canonical=True)
        return (
            smiles_can,
            ExactMolWt(mol),
            MolLogP(mol),
            CalcNumHBD(mol),
            CalcNumHBA(mol),
            CalcTPSA(mol),
        )
    except Exception:
        # Cattura altri possibili errori di RDKit
        return None


def scale_props_list(props_list: List[Tuple[float, ...]], method: str) -> List[Tuple[float, ...]]:
    """Esegue lo scaling col metodo scelto su una lista di tuple numeriche."""
    if method not in {"zscore", "minmax"}:
        raise ValueError("Il metodo di scaling deve essere 'zscore' o 'minmax'")

    arr = np.asarray(props_list, dtype=float)

    if method == "zscore":
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        std[std == 0] = 1.0  # Evita divisione per zero
        scaled = (arr - mean) / std
        # print(f"📈  Proprietà scalate con Z-score (media={mean.round(2)}, std={std.round(2)})")
    else:  # minmax
        min_val = arr.min(axis=0)
        range_val = arr.max(axis=0) - min_val
        range_val[range_val == 0] = 1.0  # Evita divisione per zero
        scaled = (arr - min_val) / range_val
        # print(f"📈  Proprietà scalate con Min-Max (min={min_val.round(2)}, range={range_val.round(2)})")

    return [tuple(row) for row in scaled], (mean, std) if method == "zscore" else (min_val, range_val)


def read_smiles(path: pathlib.Path, smiles_column: Optional[str]) -> List[str]:
    """Legge SMILES da file .txt (uno per riga) o da .csv."""
    path = pathlib.Path(path)
    smiles = []

    print(f"📖  Lettura SMILES da: {path.name}")
    if path.suffix.lower() == ".csv":
        if not smiles_column:
            raise ValueError("Per i file CSV è necessario specificare --smiles-column.")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(smiles_column)
                if val and val.strip():
                    smiles.append(val.strip())
    else:  # Assume .txt o .smi
        with path.open(encoding="utf-8") as f:
            smiles = [line.strip() for line in f if line.strip()]

    if not smiles:
        print(f"⚠️  Nessuno SMILES trovato in {path.name}. Procedo a scaricare da {ZINC_DOWNLOAD_URL}.")
        try:
            response = requests.get(ZINC_DOWNLOAD_URL)
            response.raise_for_status()  # Lancia un errore se il download fallisce
            smiles = [line.strip() for line in response.text.splitlines() if line.strip()]
        except requests.exceptions.RequestException as e:
            raise IOError(f"Errore durante il download del dataset: {e}")

    print(f"    -> Trovate {len(smiles)} stringhe SMILES.")
    return smiles


# =============================================================================
# 2. FUNZIONE PRINCIPALE DI PRE-PROCESSING
# =============================================================================


def run_preprocessing(
    smiles_list: List[str],
    save_path: pathlib.Path,
    scale_method: Optional[str] = None,
    ncpu: int = 1,
) -> List[Tuple]:
    """
    Funzione orchestratrice che calcola, scala e salva le proprietà molecolari.
    """
    if not smiles_list:
        print("⚠️  La lista di SMILES è vuota. Nessuna operazione eseguita.")
        return []

    # Esegui il calcolo delle proprietà in parallelo
    results = []
    with mp.Pool(ncpu) as pool:
        pbar = tqdm(total=len(smiles_list), desc="🧪  Calcolando proprietà", unit="smiles")
        for result in pool.imap_unordered(calc_props, smiles_list, chunksize=1000):
            if result is not None:
                results.append(result)
            pbar.update(1)
        pbar.close()

    if not results:
        print("❌  Nessuna molecola valida trovata. Nessun file di output verrà creato.")
        return []

    print(f"✅  Processate {len(smiles_list)} SMILES, trovate {len(results)} molecole valide.")

    # Separa SMILES canonici e proprietà numeriche
    smiles_can_list = [r[0] for r in results]
    props_only = [r[1:] for r in results]

    # Scala le proprietà se richiesto
    if scale_method:
        scaled_props, scale_metric = scale_props_list(props_only, method=scale_method)
        # salvo le metriche di scaling per eventuali usi futuri in un file .txt
        with open(save_path.parent / (save_path.stem + "_scaling_metrics.txt"), "w") as f:
            f.write(f"Scaling method: {scale_method}\n")
            f.write(f"Scale metrics: {scale_metric}\n")
    else:
        scaled_props = props_only

    # Unisci di nuovo SMILES e proprietà (scalate o meno)
    final_results = list(zip(smiles_can_list, *zip(*scaled_props)))

    # Salva su file CSV
    print(f"💾  Salvataggio dei risultati in: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["SMILES", "ExactMW", "LogP", "NumHBD", "NumHBA", "TPSA"]
        writer.writerow(header)
        writer.writerows(final_results)

    print(f"🎉  Pre-processing completato! Salvate {len(final_results)} molecole in {save_path.name}")
    return final_results


# =============================================================================
# 3. BLOCCO DI ESECUZIONE (per uso da riga di comando)
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Script per il pre-processing di file SMILES.")

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Percorso del file di input (.txt, .smi, o .csv).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Percorso del file di output CSV dove salvare i risultati.",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default=None,
        help="Nome della colonna contenente gli SMILES (richiesto solo per file .csv).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["zscore", "minmax"],
        default=None,
        help="Metodo di scaling da applicare alle proprietà (opzionale).",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Numero di core della CPU da utilizzare per il calcolo.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Forza il ricalcolo anche se il file di output esiste già."
    )

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_file)
    if args.output_file is None:
        output_path = input_path.parent / (input_path.stem + "_preprocessed" + f"_scale-{args.scale}" + ".csv")
        output_path = pathlib.Path(output_path)
    else:
        output_path = pathlib.Path(args.output_file)

    # Controlla se il file di output esiste già e non si vuole forzare il ricalcolo
    if output_path.exists() and not args.force:
        print(
            f"ℹ️  Il file di output '{output_path.name}' esiste già. Per sovrascriverlo, usa l'opzione --force."
        )
        print("Nessuna operazione eseguita.")
    else:
        # 1. Leggi gli SMILES dal file di input
        smiles_to_process = read_smiles(input_path, args.smiles_column)

        # 2. Esegui il pre-processing
        run_preprocessing(
            smiles_list=smiles_to_process,
            save_path=output_path,
            scale_method=args.scale,
            ncpu=args.ncpu,
        )
