import numpy as np
from rdkit import Chem
import collections
import h5py


def convert_to_smiles(vector, char):
    list_char = list(char)
    # list_char = char.tolist()
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


def stochastic_convert_to_smiles(vector, char):
    list_char = char.tolist()
    s = ""
    for i in range(len(vector)):
        prob = vector[i].tolist()
        norm0 = sum(prob)
        prob = [i / norm0 for i in prob]
        index = np.random.choice(len(list_char), 1, p=prob)
        s += list_char[index[0]]
    return s


def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))


def one_hot_index(vec, charset):
    return list(map(charset.index, vec))


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0,):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


def load_dataset(filename, split=True):
    h5f = h5py.File(filename, "r")
    if split:
        data_train = h5f["data_train"][:]
    else:
        data_train = None
    data_test = h5f["data_test"][:]
    charset = h5f["charset"][:]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset


def encode_smiles(smiles, model, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array(
        [list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))]
    )
    latent = model.encoder.predict(preprocessed)
    return latent


def smiles_to_onehot(smiles, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array(
        [list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))]
    )
    return preprocessed


def smiles_to_vector(smiles, vocab, max_length):
    while len(smiles) < max_length:
        smiles += " "
    return [vocab.index(str(x)) for x in smiles]


def decode_latent_molecule(latent, model, charset, latent_dim):
    decoded = model.decoder.predict(latent.reshape(1, latent_dim)).argmax(axis=2)[0]
    smiles = decode_smiles_from_indexes(decoded, charset)
    return smiles


def interpolate(source_smiles, dest_smiles, steps, charset, model, latent_dim):
    source_latent = encode_smiles(source_smiles, model, charset)
    dest_latent = encode_smiles(dest_smiles, model, charset)
    step = (dest_latent - source_latent) / float(steps)
    results = []
    for i in range(steps):
        item = source_latent + (step * i)
        decoded = decode_latent_molecule(item, model, charset, latent_dim)
        results.append(decoded)
    return results


def get_unique_mols(mol_list):
    inchi_keys = [Chem.InchiToInchiKey(Chem.MolToInchi(m)) for m in mol_list]
    u, indices = np.unique(inchi_keys, return_index=True)
    unique_mols = [[mol_list[i], inchi_keys[i]] for i in indices]
    return unique_mols


def accuracy(arr1, arr2, length):
    total = len(arr1)
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(arr1)):
        if np.array_equal(arr1[i, : length[i]], arr2[i, : length[i]]):
            count1 += 1
    for i in range(len(arr1)):
        for j in range(length[i]):
            if arr1[i][j] == arr2[i][j]:
                count2 += 1
            count3 += 1

    return float(count1 / float(total)), float(count2 / count3)


def load_data(path, seq_length):
    """
    Loads and processes SMILES data from a file, preparing it for sequence modeling tasks.
    Args:
        path (str): Path to the input file containing SMILES strings and associated properties.
        seq_length (int): The fixed sequence length for input/output sequences (including special tokens).
    Returns:
        smiles_input (np.ndarray): Array of shape (N, seq_length) with input SMILES sequences as indices,
            padded and prepended with SOS token.
        smiles_output (np.ndarray): Array of shape (N, seq_length) with output SMILES sequences as indices,
            padded and appended with EOS token.
        chars (tuple): Tuple of all characters in the vocabulary, starting with PAD, EOS, SOS.
        vocab (dict): Dictionary mapping each character in the vocabulary to its integer index.
        prop (np.ndarray): Array of shape (N, 5) with molecular properties as float32.
        length (np.ndarray): Array of shape (N,) with the true lengths (including SOS) of each sequence.
    Notes:
        - The input file is expected to have one SMILES string and five properties per line, separated by spaces.
        - Sequences longer than (seq_length - 2) are filtered out.
        - Special tokens: PAD ('_'), EOS ('E'), SOS ('X').
        - The function prints "diolupo" if a line does not have exactly six elements.
    """

    PAD = "_"  # token di padding
    EOS = "E"  # end-of-sequence
    SOS = "X"  # start-of-sequence

    # ── Leggi file ────────────────────────────────────────────────
    with open(path) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip()]

    # Filtra le sequenze troppo lunghe
    lines = [l for l in lines if len(l[0]) < seq_length - 2]  # -SOS -EOS
    smiles = [l[0] for l in lines]

    # ── Costruisci il vocabolario ────────────────────────────────
    counter = collections.Counter("".join(smiles))
    chars_in_smiles, _ = zip(*sorted(counter.items(), key=lambda x: -x[1]))

    # Ordine: PAD (id 0)  →  EOS  →  SOS  →  caratteri SMILES
    chars = (PAD, EOS, SOS) + tuple(ch for ch in chars_in_smiles if ch not in (PAD, EOS, SOS))
    vocab = {ch: idx for idx, ch in enumerate(chars)}

    # ── Converte SMILES in indici + padding ──────────────────────
    length = np.array([len(s) + 1 for s in smiles], dtype=np.int64)  # +1 per SOS

    smiles_input = [(SOS + s).ljust(seq_length, PAD) for s in smiles]
    smiles_output = [(s + EOS).ljust(seq_length, PAD) for s in smiles]

    smiles_input = np.array([[vocab[ch] for ch in s] for s in smiles_input], dtype=np.int64)
    smiles_output = np.array([[vocab[ch] for ch in s] for s in smiles_output], dtype=np.int64)

    # ── Carica le proprietà (float32) ────────────────────────────
    prop = np.array([l[1:] for l in lines], dtype=np.float32)

    return smiles_input, smiles_output, chars, vocab, prop, length
