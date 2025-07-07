import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class CSVDataset(Dataset):
    """A PyTorch dataset that loads tabular data from a CSV file."""

    def __init__(self, csv_path: str, train_fraction: float = 0.8) -> None:
        self.dataset = pd.read_csv(csv_path, sep=';', usecols=['ExactMW','logP','HBD','HBA','TPSA']).to_numpy()
        idx = np.arange(len(self.dataset))
        np.random.shuffle(idx)
        tr_idx = int(len(self.dataset) * train_fraction)
        train_idx = idx[:tr_idx]
        test_idx = idx[tr_idx:]
        self.train_dataset = self.dataset[train_idx]
        self.test_dataset = self.dataset[test_idx]

        # Convert to NumPy float32 and scale features
        self.scaler = StandardScaler()
        self.train_dataset = self.scaler.fit_transform(self.train_dataset.astype(np.float32))
        self.test_dataset = self.scaler.transform(self.test_dataset.astype(np.float32))
        self.train_dataset = torch.from_numpy(self.train_dataset)
        self.test_dataset = torch.from_numpy(self.test_dataset)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class Autoencoder(nn.Module):
    """Simple fully connected Autoencoder."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 2,
        hidden_dims: tuple[int, ...] = ([3]),
    ) -> None:
        super().__init__()

        # Encoder
        enc_layers = []
        dims = [input_dim, *hidden_dims, encoding_dim]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            enc_layers.append(nn.Linear(in_dim, out_dim))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers = []
        rev_dims = [encoding_dim, *reversed(hidden_dims), input_dim]
        for in_dim, out_dim in zip(rev_dims[:-1], rev_dims[1:]):
            dec_layers.append(nn.Linear(in_dim, out_dim))
            # No activation after final layer
            if out_dim != input_dim:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)



def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:03d}/{epochs} | MSE Loss: {epoch_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Train an MLP autoencoder on CSV data")
    parser.add_argument("--csv", default='ZINC_smiles_prop.csv', help="Path to the input CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini‑batch size (default: 64)")
    parser.add_argument("--encoding_dim", type=int, default=2, help="Size of the latent encoding (default: 32)")
    parser.add_argument("--save_model", default="autoencoder.pt", help="Path to save the trained model weights")
    parser.add_argument("--save_scaler", default="scaler.pkl", help="Path to save the fitted scaler object")
    parser.add_argument("--device", default="cpu", help="Training device: 'cpu', 'cuda', or 'mps'")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    dataset = CSVDataset(csv_path)

    train_dataloader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=args.batch_size, shuffle=False)
    model = Autoencoder(input_dim=dataset.train_dataset.shape[1], encoding_dim=args.encoding_dim)

    train(model, train_dataloader, epochs=args.epochs, device=args.device)
    print("\nTraining complete! Evaluating on test set...")
    # Evaluate on test set
    model.eval()    
    with torch.no_grad():
        total_loss = 0.0
        criterion = nn.MSELoss()
        for batch in test_dataloader:
            batch = batch.to(args.device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(test_dataloader.dataset)
        print(f"Test MSE Loss: {avg_loss:.6f}")
    
    # Persist artefacts
    torch.save(model.state_dict(), args.save_model)
    joblib.dump(dataset.scaler, args.save_scaler)
    print(f"\nTraining complete! Model saved to '{args.save_model}', scaler saved to '{args.save_scaler}'.")


if __name__ == "__main__":
    main()
