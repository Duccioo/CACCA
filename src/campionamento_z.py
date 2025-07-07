# build_latent_dataset.py
import argparse, pickle, csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from model.CAE import ConditionalAutoencoder
from utils import tokenize_smiles  # la tua funzione di tokenizzazione con PAD/X/E


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset drug-like: (x, c, length, smiles)
# ──────────────────────────────────────────────────────────────────────────────
class DrugDataset(Dataset):
    def __init__(self, csv_path: Path, vocab, seq_length: int):
        self.rows = []
        with csv_path.open() as f:
            reader = csv.reader(f, delimiter="\t")
            for smi, *props in reader:
                self.rows.append((smi, np.array(props, dtype=np.float32)))
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        smi, prop = self.rows[idx]
        x_ids, y_ids, length = tokenize_smiles(smi, self.vocab, self.seq_length)
        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(prop,  dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
            smi
        )


def smiles_to_fp(smiles: str, fp_size=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    return np.asarray(fp, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Funzioni di utilità
# ──────────────────────────────────────────────────────────────────────────────
def load_vocab(save_dir: Path):
    with (save_dir / "vocab.pkl").open("rb") as f:
        data = pickle.load(f)
    return data["vocab"]


def load_cae(save_dir: Path, vocab_size: int, latent_size: int, emb_size: int, device):
    class Dummy:
        emb_size = emb_size
        batch_size = 1
        latent_size = latent_size
        unit_size = 512
        n_rnn_layer = 3
        num_prop = 5
        lr = 1e-3

    model = ConditionalAutoencoder(vocab_size, Dummy()).to(device)
    ckpt = sorted(save_dir.glob("model_*.pt"))[-1]
    print("📥  checkpoint:", ckpt.name)
    model.restore(str(ckpt), map_location=device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir  = Path(args.save_dir)
    dataset_p = Path(args.dataset)

    vocab = load_vocab(save_dir)
    model = load_cae(save_dir, len(vocab), args.latent_size, args.emb_size, device)

    ds  = DrugDataset(dataset_p, vocab, args.seq_length)
    dl  = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    Z, F, smiles_all = [], [], []

    with torch.no_grad():
        for x, c, length, smi_batch in dl:
            x, c, length = x.to(device), c.to(device), length.to(device)
            z, *_ = model(x, c, length)          # z : (B, latent)
            Z.append(z.cpu())

            for smi in smi_batch:
                fp = smiles_to_fp(smi)
                if fp is not None:
                    F.append(fp)
                    smiles_all.append(smi)

    Z = torch.cat(Z).numpy()                     # (N, latent)
    F = np.stack(F).astype(np.float32)           # (N, fp_size)

    out_dir = save_dir / "latent_dataset"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "Z.npy", Z)
    np.save(out_dir / "F.npy", F)
    with (out_dir / "smiles.txt").open("w") as f:
        f.writelines(s + "\n" for s in smiles_all)

    print("✅  Salvati", Z.shape[0], "embedding in", out_dir)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      type=str, required=True, help="TSV con SMILES e proprietà")
    p.add_argument("--save_dir",     type=str, default="save")
    p.add_argument("--seq_length",   type=int, default=120)
    p.add_argument("--latent_size",  type=int, default=200)
    p.add_argument("--emb_size",     type=int, default=256)
    p.add_argument("--batch_size",   type=int, default=256)
    main(p.parse_args())
