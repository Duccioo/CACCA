import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import os
import time
import argparse
import random
import tqdm

# ---
from model.CAE import ConditionalAutoencoder
from utils.utils_final import load_data


# Seed
def set_seed(seed=69, hard=False):
    os.environ["PYTHONHASHSEED"] = str(seed)  # facoltativo

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hard:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # o :4096:8 prima di avviare Python
        # Fa fallire il run se usi un'operazione non deterministica
        torch.use_deterministic_algorithms(True)


def train_one_epoch(model, train_loader, optimizer, criterion, vocab_size):
    model.train()
    train_losses = []
    device = next(model.parameters()).device  # Get device from model parameters
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

    return train_losses


def evaluate(model, test_loader, criterion, vocab_size):
    model.eval()
    device = next(model.parameters()).device  # Get device from model parameters
    test_losses = []

    with torch.no_grad():
        for x, y, c, l in test_loader:
            x, y, c, l = x.to(device), y.to(device), c.to(device), l.to(device)
            _, logits, _ = model(x, c, l)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            test_losses.append(loss.item())

    return test_losses


def main(args):

    set_seed(69)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    training_percent = 0.75  # 75% training, 25% test
    dataset_folder = "dataset"

    # --- Load data from SMILES+properties file ---
    molecules_input, molecules_output, char, vocab, labels, length = load_data(
        args.prop_file, args.seq_length
    )
    vocab_size = len(char)
    print(f"Vocabulary size: {vocab_size}")

    exit()

    # Save vocab for generation
    vocab_data = {"char": char, "vocab": vocab}
    pad_idx = vocab["_"]

    with open(os.path.join(args.save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab_data, f)

    # --- Split into training and test sets (75/25) ---
    num_total = len(molecules_input)
    num_train = int(num_total * training_percent)

    train_x = torch.LongTensor(molecules_input[:num_train])
    train_y = torch.LongTensor(molecules_output[:num_train])
    train_c = torch.FloatTensor(labels[:num_train])
    train_l = torch.LongTensor(length[:num_train])

    test_x = torch.LongTensor(molecules_input[num_train:])
    test_y = torch.LongTensor(molecules_output[num_train:])
    test_c = torch.FloatTensor(labels[num_train:])
    test_l = torch.LongTensor(length[num_train:])

    train_loader = DataLoader(
        TensorDataset(train_x, train_y, train_c, train_l), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y, test_c, test_l), batch_size=args.batch_size, shuffle=False
    )

    # --- Model setup ---
    model = ConditionalAutoencoder(vocab_size, args).to(device)
    # torch.compile(model)  # Compilazione JIT (opzionale)

    # NEW ▶︎ se sono presenti >1 GPU, usa DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = model.configure_optimizers(weight_decay=1e-5, lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    progress_bar = tqdm.tqdm(range(args.num_epochs), desc="Training", unit="epoch", leave=True, position=0)
    for epoch in progress_bar:
        start = time.time()

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, vocab_size)

        # Evaluation
        test_losses = evaluate(model, test_loader, criterion, vocab_size)

        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_losses)
        end = time.time()

        if epoch == 0:
            print("epoch\ttrain_loss\ttest_loss\ttime (s)")
        print(f"{epoch}\t{train_loss:.4f}\t{test_loss:.4f}\t{end - start:.2f}")

        # Save model checkpoint
        model_path = os.path.join(args.save_dir, f"model_{epoch}.pt")
        model.module.save(model_path)

        # model.save(model_path)


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
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="save", help="Directory to save checkpoints")
    args = parser.parse_args()

    main(args)
