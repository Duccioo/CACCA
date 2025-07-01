# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import ConditionalAutoencoder
from utils import smiles_from_zinc_graph
import numpy as np
import os
import time
import argparse
import collections
from tqdm import tqdm

from torch_geometric.datasets import ZINC


def process_data(dataset, char_to_int, int_to_char, seq_length):
    """Process ZINC dataset to get SMILES, properties, and tokenized sequences."""
    smiles_list = []
    properties = []

    print("Converting graphs to SMILES...")
    for data in tqdm(dataset):
        smiles = smiles_from_zinc_graph(data)
        if smiles and len(smiles) < seq_length - 2:
            smiles_list.append(smiles)
            properties.append(data.y.item())

    # Create input and output sequences from SMILES
    smiles_input = [("X" + s).ljust(seq_length, "E") for s in smiles_list]
    smiles_output = [(s + "E").ljust(seq_length, "E") for s in smiles_list]

    # Tokenize
    smiles_input_tokenized = np.array(
        [np.array(list(map(char_to_int.get, s))) for s in smiles_input]
    )
    smiles_output_tokenized = np.array(
        [np.array(list(map(char_to_int.get, s))) for s in smiles_output]
    )

    lengths = np.array([len(s) + 1 for s in smiles_list])  # +1 for 'X' or 'E'
    properties = np.array(properties, dtype=np.float32).reshape(-1, 1)

    return smiles_input_tokenized, smiles_output_tokenized, properties, lengths


def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ZINC dataset
    print("Loading ZINC dataset...")
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "ZINC"
    )
    train_dataset = ZINC(path, subset=True, split="train")
    test_dataset = ZINC(
        path, subset=True, split="val"
    )  # Using validation set as test set

    # --- Create Vocabulary ---
    print("Creating vocabulary...")
    all_smiles = []
    for data in tqdm(train_dataset):
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
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    # --- Process Data ---
    train_x, train_y, train_c, train_l = process_data(
        train_dataset, char_to_int, int_to_char, args.seq_length
    )
    test_x, test_y, test_c, test_l = process_data(
        test_dataset, char_to_int, int_to_char, args.seq_length
    )

    # --- Create DataLoaders ---
    train_data = TensorDataset(
        torch.LongTensor(train_x),
        torch.LongTensor(train_y),
        torch.FloatTensor(train_c),
        torch.LongTensor(train_l),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(
        torch.LongTensor(test_x),
        torch.LongTensor(test_y),
        torch.FloatTensor(test_c),
        torch.LongTensor(test_l),
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # --- Model, Optimizer, Loss ---
    model = ConditionalAutoencoder(vocab_size, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Use CrossEntropyLoss, but ignore padding index 'E'
    pad_idx = char_to_int["E"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # --- Training Loop ---
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    progress_bar = tqdm(range(args.num_epochs), desc="Training Progress")

    for epoch in progress_bar:
        st = time.time()

        # Training
        model.train()
        train_loss_list = []
        for x, y, c, l in train_loader:
            x, y, c, l = x.to(device), y.to(device), c.to(device), l.to(device)

            optimizer.zero_grad()
            _, logits, _ = model(x, c, l)

            # Flatten logits and targets for loss calculation
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        # Testing
        model.eval()
        test_loss_list = []
        with torch.no_grad():
            for x, y, c, l in test_loader:
                x, y, c, l = x.to(device), y.to(device), c.to(device), l.to(device)
                _, logits, _ = model(x, c, l)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                test_loss_list.append(loss.item())

        train_loss = np.mean(train_loss_list)
        test_loss = np.mean(test_loss_list)
        end = time.time()

        if epoch == 0:
            print("epoch\ttrain_loss\ttest_loss\ttime (s)")
        print(f"{epoch}\t{train_loss:.3f}\t\t{test_loss:.3f}\t\t{end - st:.3f}")

        # Save model
        model_path = os.path.join(args.save_dir, f"model_{epoch}.pt")
        model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
    parser.add_argument("--latent_size", help="latent_size", type=int, default=200)
    parser.add_argument(
        "--unit_size", help="unit_size of rnn cell", type=int, default=512
    )
    parser.add_argument(
        "--n_rnn_layer", help="number of rnn layer", type=int, default=3
    )
    parser.add_argument("--seq_length", help="max_seq_length", type=int, default=120)
    # num_prop is now 1 for ZINC's constrained solubility
    parser.add_argument("--num_prop", help="number of properties", type=int, default=1)
    parser.add_argument("--num_epochs", help="epochs", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
    parser.add_argument("--save_dir", help="save dir", type=str, default="save/")
    args = parser.parse_args()

    main(args)
