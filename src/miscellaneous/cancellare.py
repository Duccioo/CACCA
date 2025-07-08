import pickle
import os
import torch
from utils.utils_final import load_data
import numpy as np

# ---
from model.CAE import ConditionalAutoencoder

save_dir = "/mnt/beegfs/home/giulio/calamita/save"
vocab_path = os.path.join(save_dir, "vocab.pkl")
with open(vocab_path, "rb") as f:
    vocab_data = pickle.load(f)
char = vocab_data["char"]
vocab = vocab_data["vocab"]
vocab_size = len(char)
char_to_int = vocab
int_to_char = {i: c for c, i in vocab.items()}

print(char)
print(vocab)


# save_dir = '/mnt/beegfs/home/giulio/calamita/save'
# vocab_path = os.path.join(save_dir, "vocab_pre.pkl")
# with open(vocab_path, "rb") as f:
#     vocab_data2 = pickle.load(f)
# char2 = vocab_data2["char"]
# vocab2 = vocab_data2["vocab"]
# vocab_size2 = len(char2)
# char_to_int = vocab2
# int_to_char2 = {i: c for c, i in vocab2.items()}

# print(char2)
# print(vocab2)


prop_file = "/mnt/beegfs/home/giulio/calamita/smiles_prop.txt"
molecules_input, molecules_output, char, vocab, labels, length = load_data(prop_file, 120)
vocab_size = len(char)
print(f"Vocabulary size: {vocab_size}")


# --- Split into training and test sets (75/25) ---
num_total = len(molecules_input)
num_train = int(num_total * 0.75)


labels = labels.astype(np.float32)
train_x = torch.LongTensor(molecules_input[:num_train])
train_y = torch.LongTensor(molecules_output[:num_train])
train_c = torch.FloatTensor(labels[:num_train])
train_l = torch.LongTensor(length[:num_train])

test_x = torch.LongTensor(molecules_input[num_train:])
test_y = torch.LongTensor(molecules_output[num_train:])
test_c = torch.FloatTensor(labels[num_train:])
test_l = torch.LongTensor(length[num_train:])


from torch.utils.data import DataLoader, TensorDataset

batch_size = 128
train_loader = DataLoader(
    TensorDataset(train_x, train_y, train_c, train_l), batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(TensorDataset(test_x, test_y, test_c, test_l), batch_size=batch_size, shuffle=False)

import argparse

args = argparse.Namespace(
    emb_size=256,
    batch_size=128,
    latent_size=512,
    unit_size=512,
    n_rnn_layer=3,
    seq_length=120,
    num_prop=5,
)

# Ora puoi usarlo come nel training script:
model = ConditionalAutoencoder(vocab_size, args).to("cuda")


dato = next(iter(train_loader))
model(dato[0].to("cuda"), dato[2].to("cuda"), dato[3].to("cuda"))


# with open("molecole.smi", "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         smiles = parts[0]
#         name = parts[1] if len(parts) > 1 else None
#         print(smiles, name)
