import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import os
import time
import argparse
import random
import tqdm
import pathlib

# ---
from model.CAE import ConditionalAutoencoder
from utils.data_utils import load_preprocessed_data, SmilesDataset
import json


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
    for data in train_loader:

        optimizer.zero_grad()
        _, logits, _ = model(
            data["input"].to(device), data["properties"].to(device), data["length"].to(device)
        )

        loss = criterion(logits.view(-1, vocab_size), data["output"].to(device).view(-1))
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
        for data in test_loader:
            data = {key: value.to(device) for key, value in data.items()}

            _, logits, _ = model(data["input"], data["properties"], data["length"])
            loss = criterion(logits.view(-1, vocab_size), data["output"].view(-1))
            test_losses.append(loss.item())

    return test_losses


def main(args):
    # --- Hyperparameters ---
    seed = args.seed
    training_percent = args.training_percent
    prop_file = args.prop_file
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    max_seq_length = args.max_seq_length
    model_path = args.model_dir

    # --- Model parameters ---
    emb_size = args.emb_size
    latent_size = args.latent_size
    hidden_size = args.hidden_size
    n_rnn_layer = args.n_rnn_layer

    # --- Setup ---
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    model_path = pathlib.Path(model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    # Save all args parameters to a JSON file
    args_dict = vars(args)
    # Convert any pathlib.Path objects to strings for JSON serialization
    args_dict = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in args_dict.items()}
    with open(os.path.join(model_path, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    # --- Load data from SMILES+properties file ---
    processed_data = load_preprocessed_data(prop_file, max_seq_length=max_seq_length)
    pytorch_dataset = SmilesDataset(processed_data)

    # split the dataset into train and test sets
    num_total = len(pytorch_dataset)
    num_train = int(num_total * training_percent)

    train_dataset, test_dataset = train_test_split(
        pytorch_dataset, train_size=num_train, shuffle=True, random_state=seed
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Save vocab for generation
    vocab_data = {"char": processed_data.chars, "vocab": processed_data.vocab}
    pad_idx = processed_data.vocab["_"]

    with open(os.path.join(model_path, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab_data, f)

    # --- Model setup ---
    model = ConditionalAutoencoder(
        len(processed_data.vocab),
        num_prop=processed_data.properties.shape[1],
        latent_size=latent_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
        n_rnn_layer=n_rnn_layer,
    ).to(device)
    torch.compile(model)  # Compilazione JIT (opzionale)

    # NEW ▶︎ se sono presenti >1 GPU, usa DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    #     model = nn.DataParallel(model)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialize lists to store losses
    train_losses_history = []
    test_losses_history = []

    progress_bar = tqdm.tqdm(range(num_epochs), desc="Training", unit="epoch", position=1)
    for epoch in progress_bar:
        start = time.time()

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, len(processed_data.vocab))

        # Evaluation
        test_losses = evaluate(model, test_loader, criterion, len(processed_data.vocab))

        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_losses)
        end = time.time()

        # Store losses
        train_losses_history.append(float(train_loss))
        test_losses_history.append(float(test_loss))

        if epoch == 0:
            tqdm.tqdm.write("epoch\ttrain_loss\ttest_loss\ttime (s)")
        tqdm.tqdm.write(f"{epoch+1}\t{train_loss:.4f}\t{test_loss:.4f}\t{end - start:.2f}")

        # Save model checkpoint
        model_path_checkpoint = os.path.join(model_path, f"model_{epoch}.pt")
        model.save(model_path_checkpoint)

        # Save losses to JSON after each epoch
        losses_dict = {"train_loss": train_losses_history, "test_loss": test_losses_history}
        with open(os.path.join(model_path, "losses.json"), "w") as f:
            json.dump(losses_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=256, help="Emb size")
    parser.add_argument("--latent_size", type=int, default=256, help="Latent vector size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of RNN units")
    parser.add_argument("--n_rnn_layer", type=int, default=3, help="Number of RNN layers")
    # ------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=120, help="Maximum SMILES sequence length")
    parser.add_argument("--training_percent", type=float, default=0.75, help="Training set percentage")
    parser.add_argument("--seed", type=int, default=69, help="Random seed for reproducibility")
    # ------------------------------------------------------
    parser.add_argument(
        "--prop_file",
        type=str,
        default="dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore.csv",
        help="Path to SMILES+properties file",
    )
    # ------------------------------------------------------
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    # ------------------------------------------------------
    parser.add_argument(
        "--model_dir",
        type=str,
        default=pathlib.Path("saved_models", "model_nodec-bidir_v2"),
        help="Directory to save model info",
    )

    args = parser.parse_args()

    main(args)
