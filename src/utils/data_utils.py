import collections
import pathlib
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# --- 1. Struttura Dati per l'Output Intermedio ---

SPECIAL_TOKENS = {
    "PAD": "_",
    "SOS": "X",
    "EOS": "E",
}


@dataclass
class ProcessedData:
    """
    Un contenitore per i dati processati pronti per essere messi in un a PyTorch Dataset.
    """

    inputs: np.ndarray
    outputs: np.ndarray
    properties: Optional[np.ndarray]
    lengths: np.ndarray  # <<< MODIFICA: Aggiunto campo per le lunghezze
    vocab: Dict[str, int]
    chars: Tuple[str, ...]

    def __len__(self):
        return len(self.inputs)


# --- 2. Funzione Principale di Caricamento e Processamento ---
def load_preprocessed_data(processed_csv_path: str, max_seq_length: int = 120) -> ProcessedData:
    """
    Carica i dati da un file CSV pre-processato, costruisce il vocabolario
    e vettorizza gli SMILES in sequenze di indici.
    """
    path = pathlib.Path(processed_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File pre-processato non trovato in '{path}'. " "Esegui prima lo script 'preprocess_data.py'."
        )

    # --- Lettura del CSV ---
    print(f"📖  Caricamento dati da: {path.name}")
    smiles_list = []
    props_list = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        num_prop_cols = len(header) - 1

        for row in reader:
            if len(row[0]) < max_seq_length - 1:
                smiles_list.append(row[0])
                if num_prop_cols > 0:
                    props_list.append(row[1:])

    if not smiles_list:
        raise ValueError("Nessuno SMILES valido trovato nel file CSV.")

    print(f"    -> Trovate {len(smiles_list)} molecole valide.")

    # --- Costruzione Vocabolario ---
    PAD = SPECIAL_TOKENS["PAD"]
    SOS = SPECIAL_TOKENS["SOS"]
    EOS = SPECIAL_TOKENS["EOS"]

    counter = collections.Counter("".join(smiles_list))
    unique_smiles_chars = sorted(counter.keys())

    chars = (PAD, SOS, EOS) + tuple(c for c in unique_smiles_chars if c not in SPECIAL_TOKENS.values())
    vocab = {ch: idx for idx, ch in enumerate(chars)}
    print(f"    -> Creato vocabolario con {len(vocab)} caratteri.")

    # --- Vettorizzazione e Calcolo Lunghezze ---
    # La lunghezza dell'input è len(SMILES) + 1 (per il token SOS).
    # La lunghezza dell'output è len(SMILES) + 1 (per il token EOS).
    # Usiamo una sola misura di lunghezza, che può essere usata per entrambi.
    lengths_np = np.array([len(s) + 1 for s in smiles_list], dtype=np.int16)

    smiles_input_padded = [(SOS + s).ljust(max_seq_length, PAD) for s in smiles_list]
    smiles_output_padded = [(s + EOS).ljust(max_seq_length, PAD) for s in smiles_list]

    inputs_np = np.array([[vocab[ch] for ch in s] for s in smiles_input_padded], dtype=np.int32)
    outputs_np = np.array([[vocab[ch] for ch in s] for s in smiles_output_padded], dtype=np.int64)

    properties_np = None
    if props_list:
        properties_np = np.array(props_list, dtype=np.float32)

    return ProcessedData(
        inputs=inputs_np,
        outputs=outputs_np,
        properties=properties_np,
        lengths=lengths_np,
        vocab=vocab,
        chars=chars,
    )


# --- 3. Classe PyTorch Dataset ---
class SmilesDataset(Dataset):
    """
    Dataset PyTorch per le sequenze SMILES.
    Restituisce un dizionario per ogni campione, inclusa la lunghezza reale.
    """

    def __init__(self, processed_data: ProcessedData):
        self.inputs = torch.from_numpy(processed_data.inputs)
        self.outputs = torch.from_numpy(processed_data.outputs)
        self.lengths = torch.from_numpy(processed_data.lengths)

        if processed_data.properties is not None:
            self.properties = torch.from_numpy(processed_data.properties)
        else:
            self.properties = None

        self.vocab = processed_data.vocab
        self.chars = processed_data.chars
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab[SPECIAL_TOKENS["PAD"]]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "input": self.inputs[idx],
            "output": self.outputs[idx],
            "length": self.lengths[idx],  # <<< MODIFICA: Aggiungiamo la lunghezza al campione
        }
        if self.properties is not None:
            sample["properties"] = self.properties[idx]

        return sample


# --- 4. Esempio di Utilizzo (Aggiornato) ---

if __name__ == "__main__":
    PREPROCESSED_FILE = "dataset/ZINC/smiles_preprocessed.csv"  # Assicurati che esista

    # Crea un file fittizio se non esiste per il test
    path_to_check = pathlib.Path(PREPROCESSED_FILE)
    if not path_to_check.exists():
        print(f"File di test '{PREPROCESSED_FILE}' non trovato. Ne creo uno fittizio.")
        path_to_check.parent.mkdir(parents=True, exist_ok=True)
        with open(PREPROCESSED_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "Prop1", "Prop2"])
            writer.writerow(["C", "1.0", "2.0"])
            writer.writerow(["CC(=O)N", "3.0", "4.0"])
            writer.writerow(["c1ccccc1", "5.0", "6.0"])

    processed_data = load_preprocessed_data(PREPROCESSED_FILE, max_seq_length=120)
    pytorch_dataset = SmilesDataset(processed_data)

    print("\n--- Informazioni sul Dataset PyTorch ---")
    print(f"Numero di campioni: {len(pytorch_dataset)}")

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset=pytorch_dataset, batch_size=4, shuffle=True)

    print("\n--- Esempio di un Batch dal DataLoader (con lunghezze) ---")
    first_batch = next(iter(data_loader))

    print("Chiavi del batch:", first_batch.keys())
    print("Shape del tensore 'input':", first_batch["input"].shape)
    print("Shape del tensore 'output':", first_batch["output"].shape)
    print("Shape del tensore 'length':", first_batch["length"].shape)  # <<< MODIFICA: Verifica
    print("Valori del tensore 'length':", first_batch["length"])  # <<< MODIFICA: Verifica
    if "properties" in first_batch:
        print("Shape del tensore 'properties':", first_batch["properties"].shape)

    print("Esempio di input:", first_batch["input"][0])
    print("Esempio di output:", first_batch["output"][0])
    print("Esempio di lunghezza:", first_batch["length"][0])  # <<< MODIFICA: Verifica
