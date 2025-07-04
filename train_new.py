import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import os
import time
import argparse

from model_new import ConditionalAutoencoder
from utils_final import load_data

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


def main(args):
    device = torch.device("cuda" )#torch.cuda.is_available())
    print(f"Using device: {device}")

    # --- Load data from SMILES+properties file ---
    molecules_input, molecules_output, char, vocab, labels, length = load_data(args.prop_file, args.seq_length)
    vocab_size = len(char)
    print(f"Vocabulary size: {vocab_size}")

    # Save vocab for generation
    vocab_data = {"char": char, "vocab": vocab}
    with open(os.path.join(args.save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab_data, f)


    # --- Split into training and test sets (75/25) ---
    num_total = len(molecules_input)
    num_train = int(num_total * 0.75)

    train_x = torch.LongTensor(molecules_input[:num_train])
    train_y = torch.LongTensor(molecules_output[:num_train])
    train_c = torch.FloatTensor(labels[:num_train])
    train_l = torch.LongTensor(length[:num_train])

    test_x = torch.LongTensor(molecules_input[num_train:])
    test_y = torch.LongTensor(molecules_output[num_train:])
    test_c = torch.FloatTensor(labels[num_train:])
    test_l = torch.LongTensor(length[num_train:])

    train_loader = DataLoader(TensorDataset(train_x, train_y, train_c, train_l),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y, test_c, test_l),
                             batch_size=args.batch_size, shuffle=False)


    # --- Model setup ---
    model = ConditionalAutoencoder(vocab_size, args).to(device)

    # NEW ▶︎ se sono presenti >1 GPU, usa DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pad_idx = vocab['_']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.num_epochs):
        start = time.time()

        # Training
        model.train()
        train_losses = []
        for x, y, c, l in train_loader:
            x, y, c, l = x.to(device), y.to(device), c.to(device), l.to(device)

            optimizer.zero_grad()
            _, logits, _ = model(x, c, l)

            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            # loss_prop = F.mse_loss(prop_pred, c)  
            # loss = loss_rec + loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y, c, l in test_loader:
                x, y, c, l = x.to(device), y.to(device), c.to(device), l.to(device)
                _, logits, _ = model(x, c, l)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        end = time.time()

        if epoch == 0:
            print("epoch\ttrain_loss\ttest_loss\ttime (s)")
        print(f"{epoch}\t{train_loss:.4f}\t{test_loss:.4f}\t{end - start:.2f}")

        # Save model checkpoint
        model_path = os.path.join(args.save_dir, f"model_{epoch}.pt")
        model.module.save(model_path)

        #model.save(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=256, help="Emb size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--latent_size", type=int, default=200, help="Latent vector size")
    parser.add_argument("--unit_size", type=int, default=512, help="Size of RNN units")
    parser.add_argument("--n_rnn_layer", type=int, default=3, help="Number of RNN layers")
    parser.add_argument("--seq_length", type=int, default=120, help="Maximum SMILES sequence length")
    parser.add_argument("--num_prop", type=int, default=5, help="Number of conditional properties")
    parser.add_argument("--prop_file", type=str, required=True, help="Path to SMILES+properties .txt file")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="save", help="Directory to save checkpoints")
    args = parser.parse_args()

    main(args)
