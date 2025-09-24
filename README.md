# 💩 C.A.C.C.A 💩

 **C**-alamita's **A**-lexia with **C**-ool **C**-onditional **A**-utoencoder

---

## 🚧 Important disclaimer! 🚧
This repo is still in active development, so some things might change over time, including the project name! 💡✨ Just a heads up: there may be some adjustments, improvements, or modifications, but the spirit remains the same. We're excited to see how this creation evolves! 🚀😉

Welcome to the CACCA 💩 project!

🧪✨ This isn't just any autoencoder. It's a **Conditional Autoencoder** specifically designed to learn powerful embeddings of chemical structures (SMILES) from the ZINC dataset.

The main goal 🎯 is to create a rich, latent space where molecules are organized not only by their structure but also by their key chemical properties 🧠

## 🚀 Installation

Getting started is easy! You'll need Python 3.8+ and a virtual environment.

1.  **Create a virtual environment:** 📦

    ```bash
    python -m venv venv
    ```

2.  **Activate it:**

    -   On Windows 🪟:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux 🐧:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the dependencies:** ⚙️

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

Follow these steps to get your embeddings!

### 📊 Step 1: Preprocess Your Data

First, you need to process your raw SMILES data. The `preprocess_data.py` script will:
1.  Canonicalize SMILES strings.
2.  Calculate key chemical properties (like Molecular Weight, logP, etc.).
3.  Scale these properties so the model can use them effectively.

You can specify the scaling method (`zscore` or `minmax`). Using a scaler is highly recommended! The script will also save the scaling parameters (like mean and standard deviation for `zscore`) for later use.

▶️ **Example command:**

```bash
python preprocess_data.py --input-file dataset/ZINC_base/smiles.txt --scale zscore
```

This will create a new preprocessed file, ready for the model! 📄➡️📈

#### Note: per aggiungere altri smiles (per esempio quelli dei romani) basta aggiungerli al file `dataset/ZINC/smiles.txt` e poi lanciare il comando di preprocessamento.
(Gli smiles dei romani sono in `assets/mol_test.csv`)


### 🧠 Step 2: Train the Model
...

---

Happy Hacking, and have fun with CACCA! 💩
