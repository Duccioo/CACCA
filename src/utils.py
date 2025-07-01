# utils.py
# NESSUNA MODIFICA NECESSARIA

import numpy as np
from rdkit import Chem
import collections
from tqdm import tqdm


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


def convert_to_smiles(vector, char):
    list_char = list(char)
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


def load_data(n, seq_length):
    f = open(n)
    lines = f.read().split("\n")[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l[0]) < seq_length - 2]
    smiles = [l[0] for l in lines]

    total_string = ""
    for s in smiles:
        total_string += s
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    # Add special characters
    chars = list(chars)
    chars += ("E",)  # End of smiles
    chars += ("X",)  # Start of smiles
    vocab["E"] = len(vocab)
    vocab["X"] = len(vocab)

    length = np.array([len(s) + 1 for s in smiles])
    smiles_input = [("X" + s).ljust(seq_length, "E") for s in smiles]
    smiles_output = [s.ljust(seq_length, "E") for s in smiles]
    smiles_input = np.array([np.array(list(map(vocab.get, s))) for s in smiles_input])
    smiles_output = np.array([np.array(list(map(vocab.get, s))) for s in smiles_output])
    prop = np.array([l[1:] for l in lines], dtype=float)
    return smiles_input, smiles_output, chars, vocab, prop, length


def smiles_from_zinc_graph(data):
    # Estrai gli atomi (nodi)
    atom_types = [
        "C",
        "O",
        "N",
        "F",
        "H",
        "S",
        "Cl",
        "Br",
        "I",
        "P",
    ]  # Tipi di atomi in ZINC
    atoms = [atom_types[x.argmax()] for x in data.x]

    # Crea una molecola modificabile
    mol = Chem.RWMol()

    # Aggiungi atomi
    node_to_idx = {}
    for i, atom_symbol in enumerate(atoms):
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        node_to_idx[i] = idx

    # Aggiungi legami
    edge_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[:, i]
        if u.item() < v.item():  # Evita di aggiungere legami duplicati
            bond_type_idx = data.edge_attr[i].argmax()
            bond_type = edge_types[bond_type_idx]
            mol.AddBond(node_to_idx[u.item()], node_to_idx[v.item()], bond_type)

    # Finalizza la molecola e ottieni lo SMILES
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None  # Restituisce None se la molecola non è valida
