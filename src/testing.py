import torch
from torch.utils.data import DataLoader
from your_dataset import SmilesDataset, pad_batch   # implementali o adatta i tuoi
from model import ConditionalAutoencoder            # modello già addestrato
from tqdm import tqdm

device = 'cuda'
model = ConditionalAutoencoder(...).to(device)
model.restore('best_model.pt', map_location=device)
model.eval()

def get_latents(smiles_list, prop_tensor):
    """
    smiles_list : list[str]       (es. ['CCO', 'c1ccccc1', ...])
    prop_tensor : torch.Tensor    (B, num_prop) – stesse unità usate a training
    """
    ds  = SmilesDataset(smiles_list)          # ritorna (tokens, length)
    dl  = DataLoader(ds, batch_size=256,
                     collate_fn=pad_batch, shuffle=False)

    Z, all_idx = [], []
    with torch.no_grad():
        for x, lengths, idx in tqdm(dl):
            x, lengths = x.to(device), lengths.to(device)
            c          = prop_tensor[idx].to(device)
            z, _, _    = model(x, c, lengths)
            Z.append(z.cpu())
            all_idx.extend(idx.tolist())
    # -- riconcatena nello stesso ordine dell’input
    Z = torch.cat(Z)[torch.tensor(all_idx).argsort()]
    return Z.numpy()        # (N, latent_size)