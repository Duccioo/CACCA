#!/usr/bin/env python
# smiles_props.py
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA

import argparse
import csv
import multiprocessing as mp
from pathlib import Path


# ────────────────────────────────────────────────────────────────────
def calc_props(smiles: str):
    """Calcola 6 proprietà; ritorna None se SMILES non valido."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles_can = Chem.MolToSmiles(mol)
    return (
        smiles_can,
        ExactMolWt(mol),
        MolLogP(mol),
        CalcNumHBD(mol),
        CalcNumHBA(mol),
        CalcTPSA(mol),
    )


def read_smiles(path: Path, smiles_column: str | None) -> list[str]:
    """Legge SMILES da .txt (uno per riga) oppure da .csv."""
    if path.suffix.lower() == ".csv":
        if smiles_column is None:
            raise ValueError("Per i CSV specifica --smiles_column con il nome della colonna SMILES")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return [row[smiles_column].strip() for row in reader if row.get(smiles_column)]
    else:  # assume txt
        with path.open() as f:
            return [l.strip() for l in f if l.strip()]


def main(args):
    in_path  = Path(args.input_filename)
    out_path = Path(args.output_filename)

    smiles_list = read_smiles(in_path, args.smiles_column)
    print(f"📄  Letti {len(smiles_list)} SMILES da {in_path.name}")

    # Pool parallelo
    ncpu = max(1, args.ncpus)
    with mp.Pool(ncpu) as pool:
        results = pool.map(calc_props, smiles_list)

    # Scrittura
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["SMILES", "ExactMW", "logP", "HBD", "HBA", "TPSA"])  # header
        valid = 0
        for row in results:
            if row is not None:
                writer.writerow(row)
                valid += 1

    print(f"✅  Salvate {valid} molecole valide in {out_path.name}")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola proprietà RDKit da SMILES")

    parser.add_argument("--input_filename",  type=str, default="smiles.txt",
                        help="File .txt (uno SMILES per riga) oppure .csv")
    parser.add_argument("--output_filename", type=str, default="smiles_prop.txt",
                        help="File TSV di output")
    parser.add_argument("--smiles_column",   type=str, default=None,
                        help="Nome colonna SMILES se l'input è un CSV")
    parser.add_argument("--ncpus", type=int, default=1,
                        help="Numero di CPU da usare (default 1)")

    main(parser.parse_args())
