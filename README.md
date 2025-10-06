# 💩✨ C.A.C.C.A ✨💩

**C**alamita's **A**lexia with **C**ool **C**onditional **A**utoencoder

---

## 🚧⚠️ Disclaimer ⚠️🚧
This repository is under active development! 🔨👷‍♀️ Some features might change over time, including the project name! 💡✨🎨 Stay tuned for updates! 🚀🎉

---

## 📖🔬 Description 

CACCA is a **Conditional Autoencoder (CAE)** 🤖 and **Conditional Variational Autoencoder (CVAE)** 🧠 project designed to learn powerful latent representations (embeddings) of chemical structures 🧪 represented as SMILES strings from the ZINC dataset! 💊📊

🎯🌟 **Main Goal**: Create a rich and structured latent space where molecules 🧬 are organized not only by their chemical structure but also by their key molecular properties (molecular weight ⚖️, logP 💧, H-bond donors/acceptors 🔗, TPSA 🎈)!

### 🔬✨ Key Features 🌈🚀

- 🎨 **Automatic Preprocessing**: SMILES canonicalization and calculation of 5 molecular properties 📐
- 🤖 **Supported Models**: CAE and CVAE with BiLSTM architecture 🧠💪
- 🏋️ **Flexible Training**: Multi-GPU support, automatic checkpoints, and detailed metrics 📊✅
- 🎨 **Conditional Generation**: Generate new molecules by specifying target properties! 🎯🧪
- 📈 **Advanced Analysis**: Evaluate embeddings with PCA, UMAP, and clustering metrics 🔍📉

---

## 🚀💫 Installation 

### 📋✅ Requirements
- 🐍 Python 3.8+
- 🎮 CUDA (optional, for GPU training) ⚡

### 🛠️⚙️ Setup

1. 📥 **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CACCA
   ```

2. 🎁 **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. ✨ **Activate it**:
   - On Windows 🪟💻:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux 🐧🍎:
     ```bash
     source venv/bin/activate
     ```

4. 📦🔧 **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻🎯 Usage 

### 📊🧹 Step 1: Data Preprocessing 

The preprocessing step canonicalizes SMILES 🧪, calculates molecular properties 🔬, and applies scaling (Z-score or Min-Max) 📐!

**🎬 Basic command**:
```bash
python src/preprocess_data.py \
    --input-file dataset/ZINC_base/smiles.txt \
    --scale zscore
```

**⚙️📝 Available parameters**:
- 📂 `--input-file`: Path to file containing SMILES (`.txt` or `.csv`)
- 📋 `--smiles-column`: SMILES column name (for CSV files only)
- 📏 `--scale`: Scaling method (`zscore` or `minmax` or `None`)
- 💾 `--output-file`: Output file path (optional)

**✨📤 Output**: CSV file with canonical SMILES and 5 scaled properties:
- ⚖️ Exact Molecular Weight (ExactMolWt)
- 💧 logP (partition coefficient)
- 🔗 H-bond Donors (HBD)
- 🎯 H-bond Acceptors (HBA)
- 🎈 Topological Polar Surface Area (TPSA)

**🎨 Example with CSV**:
```bash
python src/preprocess_data.py \
    --input-file assets/mol_test.csv \
    --smiles-column SMILES \
    --scale zscore
```

---

### 🧠💪 Step 2: Model Training 

Train a Conditional Autoencoder or CVAE on preprocessed SMILES! 🚀🎓

**🎬⚡ Basic command**:
```bash
python src/trainining_cae.py \
    --model_type CVAE \
    --prop_file dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore.csv \
    --model_dir saved_models/my_model \
    --num_epochs 100 \
    --batch_size 256 \
    --lr 3e-4
```

**⚙️🎛️ Main parameters**:

*🏗️🤖 Model Architecture*:
- 🎭 `--model_type`: Model type (`CAE` or `CVAE`)
- 📦 `--emb_size`: Embedding dimension (default: 256)
- 🌌 `--latent_size`: Latent space dimension (default: 200)
- 🧩 `--hidden_size`: RNN unit size (default: 512)
- 🔢 `--n_rnn_layer`: Number of RNN layers (default: 3)

*🏋️📚 Training*:
- 📂 `--prop_file`: Path to preprocessed file
- 🔄 `--num_epochs`: Number of epochs
- 📦 `--batch_size`: Batch size
- 📈 `--lr`: Learning rate
- ⚖️ `--weight_decay`: Weight decay for optimizer
- 📊 `--training_percent`: Training data percentage (default: 0.75)
- 📏 `--max_seq_length`: Maximum SMILES sequence length (default: 120)
- 🎲 `--seed`: Random seed for reproducibility

*💾📁 Output*:
- 🗂️ `--model_dir`: Directory to save models and checkpoints

**✨📤 Training outputs**:
- 📖 `vocab.pkl`: SMILES character vocabulary
- ⚙️ `args.json`: Training parameters
- 💾 `model_X.pt`: Model checkpoint at epoch X
- 🏆 `model_X_best.pt`: Best model based on validation loss

**🎯🔥 Complete training example**:
```bash
python src/trainining_cae.py \
    --model_type CVAE \
    --prop_file dataset/ZINC_with_drugs/smiles_preprocessed_scale-zscore.csv \
    --model_dir saved_models/model_CVAE_v5.1 \
    --num_epochs 150 \
    --batch_size 128 \
    --lr 3e-4 \
    --latent_size 200 \
    --emb_size 256 \
    --hidden_size 512 \
    --seed 69
```

---

### 🎨✨ Step 3: Generate New Molecules 

Generate brand new molecules 🧪💫 by specifying target molecular properties! 🎯🔬

**🎬🚀 Basic command**:
```bash
python src/generate_new.py \
    --save_dir saved_models/model_CVAE_v5.1 \
    --num_samples 50 \
    --logP_min 0 \
    --logP_max 3
```

**⚙️📝 Available parameters**:
- 🗂️ `--save_dir`: Directory of the trained model
- 🔢 `--num_samples`: Number of molecules to generate
- 🌌 `--latent_size`: Latent space dimension (must match the model)
- 📏 `--seq_length`: Maximum generated sequence length (default: 120)
- 💾 `--output_file`: File to save generated SMILES (optional)

*🎯🔬 Molecular property ranges*:
- 💧 `--logP_min`, `--logP_max`: Range for logP
- ⚖️ `--MolWt_min`, `--MolWt_max`: Range for molecular weight
- 🔗 `--HBD_min`, `--HBD_max`: Range for H donors
- 🎯 `--HBA_min`, `--HBA_max`: Range for H acceptors
- 🎈 `--TPSA_min`, `--TPSA_max`: Range for TPSA

**🎨🔥 Targeted generation example**:
```bash
python src/generate_new.py \
    --save_dir saved_models/model_CVAE_v5.1 \
    --num_samples 100 \
    --logP_min 2.0 \
    --logP_max 4.0 \
    --MolWt_min 200 \
    --MolWt_max 500 \
    --output_file assets/generated.smi
```

---

### 📈🔍 Step 4: Embeddings Analysis 

Evaluate the quality of embeddings 🧠✨ generated by the model using various dimensionality reduction techniques and metrics! 📊🎯

**🎬💫 Basic command**:
```bash
python src/metric_final.py
```

This script does the following magic 🪄:
1. 📥 Loads a trained model and a test molecule set
2. 🌈 Generates embeddings using different methods:
   - 🤖 **CAE**: Embeddings from autoencoder latent space
   - 📉 **PCA**: Dimensionality reduction on molecular descriptors
   - 🗺️ **UMAP**: Dimensionality reduction on descriptors, fingerprints, or CAE embeddings
3. 📊 Calculates evaluation metrics:
   - 🎯 **Enrichment Factor (EF@k)**: Ability to retrieve similar molecules
   - 💯 **PP-hit@k**: Proportion of positive molecules among k neighbors
   - 🌟 **Silhouette Score**: Clustering quality

**✨📤 Output**:
- 📁 Directory `analysis_results_nX/`:
  - 🗂️ `embeddings_METHOD.csv`: Embeddings for each method
  - 📏 `distances_METHOD.csv`: Distances from reference polymer
  - 📋 `metrics_summary_report.csv`: Summary report of all metrics

**📝✏️ Note**: Modify the file to specify:
- 🗂️ Path to trained model
- 📂 CSV file with test molecules
- 🎯 Reference polymer name
- 💾 Output directory

---

### 🔧🛠️ Additional Scripts 

#### 🎨 `generatore2.py`
Alternative molecule generator 🧪✨ with similar interface to `generate_new.py`!

#### 📊 `prop_smiles.py`
Utility to calculate and visualize 🔬👀 molecular properties from SMILES!

```bash
python src/prop_smiles.py
```

---

## 📁🗂️ Project Structure 

```
CACCA/ 🏠
├── src/ 💻
│   ├── preprocess_data.py      # 🧹 SMILES preprocessing
│   ├── trainining_cae.py       # 🏋️ Model training
│   ├── generate_new.py         # 🎨 Molecule generation
│   ├── generatore2.py          # 🎪 Alternative generator
│   ├── metric_final.py         # 📊 Analysis and metrics
│   ├── prop_smiles.py          # 🔬 Property calculation
│   ├── model/ 🤖
│   │   ├── CAE.py             # 🧠 CAE and CVAE architectures
│   │   └── ... ✨
│   └── utils/ 🛠️
│       ├── data_utils.py      # 📦 Data management utilities
│       ├── utils.py           # 🔧 Helper functions
│       └── ... 💫
├── dataset/ 📚
│   ├── ZINC_base/             # 💊 Base ZINC dataset
│   └── ZINC_with_drugs/       # 💉 ZINC + drugs dataset
├── saved_models/ 💾            # 🏆 Trained models
├── assets/ 🎁                  # 📄 Test files and resources
├── analysis_results_nX/ 📈    # 📊 Analysis results
├── requirements.txt 📋         # 🐍 Python dependencies
└── README.md 📖                # 📚 This documentation
```

---

## 🔬🧪 Technical Details 

### 🏗️🤖 Model Architecture 

**🎯 Conditional Autoencoder (CAE)**:
- 🧠 **Encoder**: BiLSTM that processes SMILES + molecular properties 🔗
- 🌌 **Latent Space**: Compact representation of the molecule ✨
- 🎨 **Decoder**: Autoregressive LSTM that generates SMILES token-by-token 📝
- 🔮 **Property Predictor**: MLP that predicts properties from latent vector 🎯

**✨ Conditional Variational Autoencoder (CVAE)**:
- 🎲 Like CAE, but with sampling from latent space (mean + log_sigma) 🌟
- 📉 Additional loss: KL divergence for regularization ⚖️

### 🧬💊 Molecular Properties Used 

1. ⚖️ **ExactMolWt**: Exact molecular weight
2. 💧 **MolLogP**: Octanol/water partition coefficient (lipophilicity)
3. 🔗 **CalcNumHBD**: Number of hydrogen bond donors
4. 🎯 **CalcNumHBA**: Number of hydrogen bond acceptors
5. 🎈 **CalcTPSA**: Topological polar surface area

---

## 📊💊 Dataset 

The project uses the ZINC dataset 🏛️, a free library of commercially available chemical compounds! 🧪✨ The file is automatically downloaded 📥 during preprocessing if not present! 🎉

**✨🎨 To add custom molecules**:
1. 📝 Add SMILES to the file `dataset/ZINC_base/smiles.txt` (one per line)
2. 📋 Or create a CSV with a SMILES column
3. 🚀 Run preprocessing as indicated above!

---

## 🎯🔥 Complete Workflow 

```bash
# 1. 🧹✨ Preprocessing
python src/preprocess_data.py \
    --input-file dataset/ZINC_base/smiles.txt \
    --scale zscore

# 2. 🏋️💪 Training
python src/trainining_cae.py \
    --model_type CVAE \
    --prop_file dataset/ZINC_base/smiles_preprocessed_scale-zscore.csv \
    --model_dir saved_models/my_cvae \
    --num_epochs 100

# 3. 🎨🧪 Generation
python src/generate_new.py \
    --save_dir saved_models/my_cvae \
    --num_samples 50 \
    --logP_min 1.0 \
    --logP_max 3.0 \
    --output_file generated_molecules.smi

# 4. 📊🔍 Analysis (after configuring metric_final.py)
python src/metric_final.py
```

---

## 🤝💖 Contributions 

Contributions, suggestions, and feedback are super welcome! 🎉✨ Feel free to open issues 🐛 or pull requests 🔀! We love collaboration! 💪🌟

---

## 📝⚖️ License 

*Specify license here if applicable* 📜✨

---

## 🙏🌟 Acknowledgments 

- 🧪 **RDKit**: For molecular property calculations! 🔬
- 💊 **ZINC Database**: For the amazing molecule dataset! 📊
- 🔥 **PyTorch**: Deep learning framework extraordinaire! 🧠

---

## 🎮🚀 Get Started! 

Ready to dive into molecular magic? 🧙‍♂️✨ Follow the steps above and start creating amazing molecules! 🧬💫

Got questions? 🤔 Need help? 💬 Don't hesitate to reach out! 📧

---

**Happy Molecular Hacking!** 💩🧬🎉✨🔬🎯🚀💫🌟🎨🔥💪🧪🏆

*Let's make some awesome molecules together!* 🤝💖🧬
```

---

## � Contributi

Contributi, suggerimenti e feedback sono benvenuti! Sentiti libero di aprire issue o pull request.

---


## 🙏 Riconoscimenti

- **RDKit**: Per il calcolo delle proprietà molecolari
- **ZINC Database**: Per il dataset di molecole
- **PyTorch**: Framework per il deep learning

---

Happy Molecular Hacking! 💩🧬
