import torch
import numpy as np
import argparse
import os
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---
from model.CAE import ConditionalAutoencoder
from utils import decode_smiles_from_indexes


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocab & model info
    vocab_path = os.path.join(args.save_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Missing vocab.pkl in {args.save_dir}. Did you save it during training?")

    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    char = vocab_data["char"]
    vocab = vocab_data["vocab"]
    vocab_size = len(char)
    char_to_int = vocab
    int_to_char = {i: c for c, i in vocab.items()}

    # Define model config (must match training)
    class DummyArgs:
        emb_size = 256
        batch_size = 128
        latent_size = args.latent_size
        unit_size = 512
        n_rnn_layer = 3
        lr = 0.0001
        num_prop = 5  # logP, MolWt, HBD, HBA, TPSA

    model = ConditionalAutoencoder(vocab_size, DummyArgs()).to(device)

    # Load latest checkpoint
    model_files = [f for f in os.listdir(args.save_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise RuntimeError(f"No model found in {args.save_dir}")
    latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in model_files])
    ckpt_path = os.path.join(args.save_dir, f"model_{latest_epoch}.pt")
    print(f"Loading model from {ckpt_path}")
    model.restore(ckpt_path, map_location=device)

    # Generation
    print(f"\n🎯 Target: logP in range [{args.logP_min}, {args.logP_max}]")
    valid_molecules = []
    attempts = 0
    max_attempts = 500

    while len(valid_molecules) < args.num_samples and attempts < max_attempts:
        attempts += 1

        # Generate random conditional vector
        target_logP = np.random.uniform(args.logP_min, args.logP_max)
        target_MolWt = np.random.uniform(args.MolWt_min, args.MolWt_max)
        target_logP = np.random.uniform(args.logP_min, args.logP_max)
        target_HBD = np.random.randint(args.HBD_min, args.HBD_max + 1)
        target_HBA = np.random.randint(args.HBA_min, args.HBA_max + 1)
        target_TPSA = np.random.uniform(args.TPSA_min, args.TPSA_max)

        c_vals = [target_MolWt, target_logP, target_HBD, target_HBA, target_TPSA]

        c = torch.tensor([c_vals], dtype=torch.float32).to(device)

        z = torch.randn(1, DummyArgs.latent_size).to(device)
        start_token = torch.LongTensor([[char_to_int["X"]]]).to(device)
        end_token = torch.LongTensor([[char_to_int["E"]]]).to(device).squeeze(dim=0)

        # Autoregressive sampling from model
        generated_tensor = model.sample(
            z, c, start_token, eos_idx=end_token, seq_length=args.seq_length, device=device
        )
        smiles = decode_smiles_from_indexes(generated_tensor[0].tolist(), int_to_char)

        mol = Chem.MolFromSmiles(smiles.strip("_"))
        if mol:
            calc_logp = Descriptors.MolLogP(mol)
            if args.logP_min <= calc_logp <= args.logP_max:
                valid_molecules.append((smiles, calc_logp))
                print(f"✓ {smiles}  (logP = {calc_logp:.2f})")
            else:
                print(f"✗ {smiles}  (logP = {calc_logp:.2f} fuori range)")
        else:
            print("⚠️  SMILES non valido")

    print(f"\n✅ Generati {len(valid_molecules)} SMILES validi nel range logP desiderato.")

    # Opzionale: salvataggio dei risultati
    if args.output_file:
        with open(args.output_file, "w") as f:
            for smi, logp in valid_molecules:
                f.write(f"{smi}\t{logp:.2f}\n")
        print(f"\n💾 Risultati salvati in: {args.output_file}")


parser = argparse.ArgumentParser()

# Range per ogni proprietà
parser.add_argument("--MolWt_min", type=float, default=200.0)
parser.add_argument("--MolWt_max", type=float, default=400.0)

parser.add_argument("--logP_min", type=float, default=0.0)
parser.add_argument("--logP_max", type=float, default=4.0)

parser.add_argument("--HBD_min", type=float, default=0)
parser.add_argument("--HBD_max", type=float, default=5)

parser.add_argument("--HBA_min", type=float, default=0)
parser.add_argument("--HBA_max", type=float, default=10)

parser.add_argument("--TPSA_min", type=float, default=20.0)
parser.add_argument("--TPSA_max", type=float, default=120.0)
parser.add_argument("--emb_size", type=int, default=256, help="Emb size")
parser.add_argument("--seq_length", type=int, default=120)
parser.add_argument("--save_dir", type=str, default="save")
parser.add_argument("--latent_size", type=int, default=200)
parser.add_argument("--num_samples", type=int, default=5)

args = parser.parse_args()
main(args)
