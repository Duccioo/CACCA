# generate_smiles.py
import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from model.CAE import ConditionalAutoencoder
from utils import decode_smiles_from_indexes


def load_vocab(save_dir: Path):
    vocab_path = Path(save_dir) / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"{vocab_path} non trovato – hai salvato il vocab durante il training?")
    with vocab_path.open("rb") as f:
        data = pickle.load(f)
    char = data["char"]
    vocab = data["vocab"]
    int_to_char = {i: c for c, i in vocab.items()}
    pad_idx = vocab["_"]  # attenzione: deve corrispondere al token PAD creato prima
    sos_idx = vocab["X"]
    eos_idx = vocab["E"]
    return char, vocab, int_to_char, pad_idx, sos_idx, eos_idx


def build_model(vocab_size: int, latent_size: int, emb_size: int = 256):
    class DummyArgs:
        emb_size = 256
        batch_size = 1  # non serve più in inference
        latent_size = 200
        unit_size = 512
        n_rnn_layer = 3
        lr = 1e-4
        num_prop = 5

    return ConditionalAutoencoder(vocab_size, DummyArgs())


def load_latest_checkpoint(model: torch.nn.Module, save_dir: Path, device: torch.device):
    ckpts = sorted(save_dir.glob("model_*.pt"))
    if not ckpts:
        raise RuntimeError(f"Nessun checkpoint trovato in {save_dir}")
    ckpt_path = ckpts[-1]  # ultimo = più recente (nomi ordinati)
    print(f"🧩  Carico pesi da: {ckpt_path.name}")
    model.restore(str(ckpt_path), map_location=device)
    model.eval()


@torch.no_grad()
def generate_batch(model, vocab, int_to_char, sos_idx, eos_idx, args, device):
    """Genera una singola molecola e la valida."""
    # ---- vettore condizioni casuali -------------------------------------------------
    target_MolWt = np.random.uniform(args.MolWt_min, args.MolWt_max)
    target_logP = np.random.uniform(args.logP_min, args.logP_max)
    target_HBD = np.random.randint(args.HBD_min, args.HBD_max + 1)
    target_HBA = np.random.randint(args.HBA_min, args.HBA_max + 1)
    target_TPSA = np.random.uniform(args.TPSA_min, args.TPSA_max)

    c_vals = [target_MolWt, target_logP, target_HBD, target_HBA, target_TPSA]
    c = torch.tensor([c_vals], dtype=torch.float32, device=device)

    # ---- campionamento latente + autoregressivo -------------------------------------
    z = torch.randn(1, args.latent_size, device=device)
    start_token = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    generated = model.sample(
        latent_vector=z,
        c=c,
        start_token=start_token,
        seq_length=args.seq_length,
        eos_idx=eos_idx,  # <-- ora è un int
        pad_idx=vocab["_"],
        temperature=0.8,
        device=device,
    )

    smiles = decode_smiles_from_indexes(generated[0].tolist(), int_to_char)

    # ---- validazione RDKit -----------------------------------------------------------
    mol = Chem.MolFromSmiles(smiles.strip("_"))
    if mol is None:
        return None, None, "SMILES non valido"
    calc_logp = Descriptors.MolLogP(mol)
    in_range = args.logP_min <= calc_logp <= args.logP_max
    return smiles, calc_logp, "OK" if in_range else "fuori range"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌐  Device: {device}\n")

    save_dir = Path(args.save_dir)
    char, vocab, int_to_char, pad_idx, sos_idx, eos_idx = load_vocab(save_dir)
    model = build_model(len(char), args.latent_size, args.emb_size).to(device)
    load_latest_checkpoint(model, save_dir, device)

    print(f"🎯 Target logP ∈ [{args.logP_min}, {args.logP_max}] – Genero {args.num_samples} molecole …\n")

    valid = []
    attempts = 0
    MAX_ATTEMPTS = 1000

    while len(valid) < args.num_samples and attempts < MAX_ATTEMPTS:
        attempts += 1
        smi, logp, status = generate_batch(model, vocab, int_to_char, sos_idx, eos_idx, args, device)

        if smi is None:
            print("⚠️  SMILES non valido")
        elif status == "OK":
            valid.append((smi, logp))
            print(f"✓ {smi}  (logP = {logp:.2f})")
        else:
            print(f"✗ {smi}  (logP = {logp:.2f} – fuori range)")

    print(f"\n✅ Generati {len(valid)} SMILES validi in {attempts} tentativi.\n")

    if args.output_file:
        with open(args.output_file, "w") as f:
            for smi, logp in valid:
                f.write(f"{smi}\t{logp:.2f}\n")
        print(f"💾  Salvato in {args.output_file}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Range proprietà
    parser.add_argument("--MolWt_min", type=float, default=200.0)
    parser.add_argument("--MolWt_max", type=float, default=400.0)
    parser.add_argument("--logP_min", type=float, default=0.0)
    parser.add_argument("--logP_max", type=float, default=4.0)
    parser.add_argument("--HBD_min", type=int, default=0)
    parser.add_argument("--HBD_max", type=int, default=5)
    parser.add_argument("--HBA_min", type=int, default=0)
    parser.add_argument("--HBA_max", type=int, default=10)
    parser.add_argument("--TPSA_min", type=float, default=20.0)
    parser.add_argument("--TPSA_max", type=float, default=120.0)

    # Parametri modello / sampling
    parser.add_argument("--seq_length", type=int, default=120)
    parser.add_argument("--latent_size", type=int, default=200)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=5)

    # I/O
    parser.add_argument("--save_dir", type=str, default="save")
    parser.add_argument("--output_file", type=str, default="generated.smi")

    main(parser.parse_args())
