# src/generate.py

import torch
import numpy as np
import argparse
import os
import collections
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors


from model import ConditionalAutoencoder
from utils import smiles_from_zinc_graph, decode_smiles_from_indexes
from torch_geometric.datasets import ZINC


def get_vocab(dataset):
    """Creates vocabulary from the training dataset."""
    print("Creating vocabulary...")
    all_smiles = []
    for data in tqdm(dataset):
        smiles = smiles_from_zinc_graph(data)
        if smiles:
            all_smiles.append(smiles)

    total_string = "".join(all_smiles)
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    chars = list(chars) + ["E", "X"]  # Add special characters
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_int, int_to_char, len(chars)


def sample(model, z, c, seq_length, device, char_to_int, int_to_char):
    """
    Samples a SMILES string from the model given a latent vector and a condition.
    """
    model.eval()
    with torch.no_grad():
        # Start with the 'X' token
        x = torch.LongTensor([[char_to_int["X"]]]).to(device)

        # Initialize hidden state for the decoder LSTM
        # We'll use the latent vector 'z' to form the initial hidden state.
        # The encoder's last hidden state is processed by a linear layer to get 'z'.
        # For generation, we can't perfectly reverse this, but we can initialize
        # the decoder's hidden state in a way that depends on 'z'.
        # A simple approach is to have a linear layer map z to the initial hidden state size.
        # Since the model doesn't have this layer, we'll initialize hidden state to zeros
        # and rely on the concatenated z to guide the generation at each step.
        hidden = None

        generated_indices = []

        for _ in range(seq_length):
            # The decode function expects a sequence, so we provide the sequence built so far
            # However, the current decode function is designed for teacher forcing.
            # We need a step-by-step generation logic.

            # Let's adapt the logic from the decode function for a single time step.
            z_expand = z.unsqueeze(1)
            c_expand = c.unsqueeze(1)
            x_embed = model.embedding_decode(x)

            inp = torch.cat([z_expand, x_embed, c_expand], dim=-1)

            output, hidden = model.decoder_rnn(inp, hidden)

            logits = model.output_linear(output.squeeze(1))
            probs = torch.softmax(logits, dim=-1)

            # Sample the next character
            next_char_idx = torch.multinomial(probs, 1).item()

            if next_char_idx == char_to_int["E"]:
                break  # Stop if end token is generated

            generated_indices.append(next_char_idx)
            x = torch.LongTensor([[next_char_idx]]).to(device)

        return decode_smiles_from_indexes(generated_indices, int_to_char)


def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Vocabulary ---
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "ZINC"
    )
    train_dataset = ZINC(data_path, subset=True, split="train")
    char_to_int, int_to_char, vocab_size = get_vocab(train_dataset)

    # --- Load Model ---
    # Use the same arguments as in training for model architecture
    train_args = argparse.Namespace(
        latent_size=200,
        unit_size=512,
        n_rnn_layer=3,
        num_prop=1,
        batch_size=128,  # Not used in generation but part of model init
    )
    model = ConditionalAutoencoder(vocab_size, train_args).to(device)

    # Find the latest model
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

    # --- Generate Molecule ---
    model.eval()

    print(f"Generating a molecule with target logP: {args.logP}...")

    mol = None
    attempts = 0
    max_attempts = 100  # Limit the number of attempts to avoid infinite loops

    generated_smiles = ""
    while not mol or generated_smiles == "" and attempts < max_attempts:
        attempts += 1
        # Create a random latent vector for each attempt
        z = torch.randn(1, train_args.latent_size).to(device)

        # Create the condition vector with the desired logP
        c = torch.FloatTensor([[args.logP]]).to(device)

        print(f"\nAttempt {attempts}/{max_attempts}...")
        generated_smiles = sample(
            model, z, c, args.seq_length, device, char_to_int, int_to_char
        )

        print(f"Generated SMILES: {generated_smiles}")

        # Try to create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(generated_smiles)

        if not mol or mol.GetNumAtoms() == 0 or generated_smiles == "" :
            print("Invalid SMILES, trying again...")

    print("\n----------------------------------------")
    if mol:
        print("Valid molecule generated!")
        print(f"Final SMILES: {generated_smiles}")
        logp = Descriptors.MolLogP(mol)
        print(f"Calculated logP: {logp:.2f}")
    else:
        print(f"Failed to generate a valid molecule after {max_attempts} attempts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logP", help="Target logP value for generation", type=float, default=3.91
    )
    parser.add_argument("--seq_length", help="max_seq_length", type=int, default=35)
    parser.add_argument(
        "--save_dir", help="directory with saved models", type=str, default="save/"
    )
    args = parser.parse_args()
    main(args)
