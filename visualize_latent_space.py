# src/visualize_latent_space.py

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors

from model import ConditionalAutoencoder
from utils import get_vocab  # Re-using the vocab creation from utils script

from torch_geometric.datasets import ZINC


def preprocess_smiles(smiles_list, char_to_int, seq_length):
    """Converts a list of SMILES strings to tokenized and padded tensors."""
    processed_x = []
    processed_c = []
    processed_l = []
    valid_smiles = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol and len(smiles) < seq_length - 2:
            # Calculate logP
            logp = Descriptors.MolLogP(mol)
            processed_c.append([logp])

            # Tokenize and pad, ignoring characters not in vocab
            tokenized = [char_to_int[c] for c in smiles if c in char_to_int]

            # Check if any characters were actually tokenized
            if not tokenized:
                print(f"Skipping SMILES with no valid characters in vocab: {smiles}")
                continue

            padded = (
                [char_to_int["X"]]
                + tokenized
                + [char_to_int["E"]] * (seq_length - len(tokenized) - 1)
            )
            processed_x.append(padded)
            processed_l.append(len(tokenized) + 1)  # +1 for 'X'
            valid_smiles.append(smiles)
        else:
            print(f"Skipping invalid or too long SMILES: {smiles}")

    return (
        torch.LongTensor(processed_x),
        torch.FloatTensor(processed_c),
        torch.LongTensor(processed_l),
        np.array(processed_c).flatten(),  # Return logP values for plotting
        valid_smiles,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Vocabulary ---
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "ZINC"
    )
    train_dataset = ZINC(data_path, subset=True, split="train")
    char_to_int, int_to_char, vocab_size = get_vocab(train_dataset)

    # --- Model ---
    train_args = argparse.Namespace(
        latent_size=200, unit_size=512, n_rnn_layer=3, num_prop=1, batch_size=128
    )
    model = ConditionalAutoencoder(vocab_size, train_args).to(device)

    model_files = [
        f
        for f in os.listdir(args.save_dir)
        if f.startswith("model_") and f.endswith(".pt")
    ]
    if not model_files:
        print(f"Error: No model files found in {args.save_dir}")
        return
    latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in model_files])
    model_path = os.path.join(args.save_dir, f"model_{latest_epoch}.pt")
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Input Data ---
    # Example SMILES strings
    input_smiles = [
        "CC(C)Cc1ccccc1",  # Ibuprofen (low logP)
        "c1ccccc1",  # Benzene
        "O=C(O)c1ccccc1C(=O)O",  # Phthalic acid
        "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Chrysene (high logP)
        "C1CCCCC1",  # Cyclohexane
        "CCO",  # Ethanol
        "CNC(=O)N(C)C",  # Dimethylurea
        "CC(=O)O",  # Acetic acid
        "C1=CC=C(C=C1)C(C(C(=O)O)N)C(C(=O)O)N",  # Example with more heteroatoms
        "C[C@]12CC[C@H]3[C@H](CC[C@@]4(C)C3CC[C@@H]4O)[C@@H]1CC=C2O",  # OESTRADIOL
        "CC1=C(C(=O)N(N1C)C)C2=CC=CC=C2",  # Pyramidone
        "C1=CC=C(C=C1)CC2=CC=CC=C2",
        "c1ccccc1C1NC=CO1",
        "CC(=O)NC1=CC=C(O)C=C1"
    ]

    # --- Get Latent Vectors ---
    x, c, l, logp_values, valid_smiles = preprocess_smiles(
        input_smiles, char_to_int, args.seq_length
    )
    x, c, l = x.to(device), c.to(device), l.to(device)

    with torch.no_grad():
        latent_vectors = model.encode(x, c, l).cpu().numpy()

    print(f"Successfully encoded {len(latent_vectors)} molecules.")

    # --- UMAP Reduction and Plotting ---
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=logp_values, cmap="viridis", s=50
    )
    plt.colorbar(scatter, label="logP")
    plt.title("2D UMAP projection of the Latent Space")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    # Add labels for each point
    for i, txt in enumerate(valid_smiles):
        plt.annotate(
            txt,
            (embedding[i, 0], embedding[i, 1]),
            xytext=(5, -5),
            textcoords="offset points",
        )

    plot_path = "latent_space_umap.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_length",
        help="max_seq_length used during training",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--save_dir", help="directory with saved models", type=str, default="save/"
    )
    args = parser.parse_args()
    main(args)
