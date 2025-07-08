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
        


def train_one_epoch(model, train_loader, optimizer, vocab_size, criterion_recon, criterion_prop, pad_idx, scheduler=None, lambda_prop=0.5):
    model.train()
    
    epoch_ce_losses = []
    epoch_mse_losses = []
    epoch_mae_losses = []
    epoch_accuracies = []

    device = next(model.parameters()).device
    
    for data in train_loader:
        optimizer.zero_grad()
        
        _, logits, c_pred = model(
            data["input"].to(device), data["properties"].to(device), data["length"].to(device)
        )
        
        target_tokens = data["output"].to(device)
        target_props = data["properties"].to(device)

        # Calcolo delle loss
        loss_recon = criterion_recon(logits.view(-1, vocab_size), target_tokens.view(-1))
        loss_prop_mse = criterion_prop(c_pred, target_props)
        
        # Loss totale
        total_loss = loss_recon + lambda_prop * loss_prop_mse
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calcolo metriche addizionali
        with torch.no_grad():
            # MAE
            loss_prop_mae = torch.nn.functional.l1_loss(c_pred, target_props)
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != pad_idx)
            correct_preds = (preds[mask] == target_tokens[mask]).sum().item()
            total_tokens = mask.sum().item()
            accuracy = correct_preds / total_tokens if total_tokens > 0 else 0
            
            epoch_ce_losses.append(loss_recon.item())
            epoch_mse_losses.append(loss_prop_mse.item())
            epoch_mae_losses.append(loss_prop_mae.item())
            epoch_accuracies.append(accuracy)

    return {
        "ce": np.mean(epoch_ce_losses),
        "mse": np.mean(epoch_mse_losses),
        "mae": np.mean(epoch_mae_losses),
        "acc": np.mean(epoch_accuracies)
    }


def evaluate(model, test_loader, vocab_size, criterion_recon, criterion_prop, pad_idx):
    model.eval()
    
    epoch_ce_losses = []
    epoch_mse_losses = []
    epoch_mae_losses = []
    epoch_accuracies = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for data in test_loader:
            _, logits, c_pred = model(
                data["input"].to(device), data["properties"].to(device), data["length"].to(device)
            )
            
            target_tokens = data["output"].to(device)
            target_props = data["properties"].to(device)

            # Calcolo delle loss
            loss_recon = criterion_recon(logits.view(-1, vocab_size), target_tokens.view(-1))
            loss_prop_mse = criterion_prop(c_pred, target_props)
            
            # Calcolo metriche addizionali
            loss_prop_mae = torch.nn.functional.l1_loss(c_pred, target_props)
            
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != pad_idx)
            correct_preds = (preds[mask] == target_tokens[mask]).sum().item()
            total_tokens = mask.sum().item()
            accuracy = correct_preds / total_tokens if total_tokens > 0 else 0
            
            epoch_ce_losses.append(loss_recon.item())
            epoch_mse_losses.append(loss_prop_mse.item())
            epoch_mae_losses.append(loss_prop_mae.item())
            epoch_accuracies.append(accuracy)

    return {
        "ce": np.mean(epoch_ce_losses),
        "mse": np.mean(epoch_mse_losses),
        "mae": np.mean(epoch_mae_losses),
        "acc": np.mean(epoch_accuracies)
    }


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

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
    # torch.compile(model)  # Compilazione JIT (opzionale)

    # NEW ▶︎ se sono presenti >1 GPU, usa DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    #     model = nn.DataParallel(model)

    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=lr)
    criterion_recon = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_prop = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialize lists to store losses
    metrics_history = []

    progress_bar = tqdm.tqdm(range(num_epochs), desc="Training", unit="epoch", position=1)
    for epoch in progress_bar:
        
        lambda_prop = min(1.0, epoch/20) * 0.1
        start = time.time()

        # Training
        train_metrics = train_one_epoch(model, train_loader, optimizer, len(processed_data.vocab), criterion_recon, criterion_prop, pad_idx, scheduler, lambda_prop)

        # Evaluation
        test_metrics = evaluate(model, test_loader, len(processed_data.vocab), criterion_recon, criterion_prop, pad_idx)
        end = time.time()

        # Log metrics
        epoch_log = {
            "epoch": epoch + 1,
            "ce_train": train_metrics["ce"],
            "mse_train": train_metrics["mse"],
            "mae_train": train_metrics["mae"],
            "acc_train": train_metrics["acc"],
            "ce_test": test_metrics["ce"],
            "mse_test": test_metrics["mse"],
            "mae_test": test_metrics["mae"],
            "acc_test": test_metrics["acc"],
            "time_s": end - start
        }
        metrics_history.append(epoch_log)

        if epoch == 0:
            header = " | ".join(f"{k:<10}" for k in epoch_log.keys())
            tqdm.tqdm.write(header)
            tqdm.tqdm.write("-" * len(header))
        
        log_line = " | ".join(f"{v:<10.4f}" if isinstance(v, float) else f"{v:<10}" for v in epoch_log.values())
        tqdm.tqdm.write(log_line)


        # Save model checkpoint
        model_path_checkpoint = os.path.join(model_path, f"model_{epoch}.pt")
        model.save(model_path_checkpoint)

        # Save losses to JSON after each epoch
        with open(os.path.join(model_path, "losses.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=256, help="Emb size")
    parser.add_argument("--latent_size", type=int, default=128, help="Latent vector size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Size of RNN units")
    parser.add_argument("--n_rnn_layer", type=int, default=3, help="Number of RNN layers")
    # ------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    # ------------------------------------------------------
    parser.add_argument(
        "--model_dir",
        type=str,
        default=pathlib.Path("saved_models", "model_v3.5"),
        help="Directory to save model info",
    )

    args = parser.parse_args()

    main(args)
