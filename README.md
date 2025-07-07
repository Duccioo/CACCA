# 💩 CACCA 💩 

Calamita 
Alexia 
Con 
Conditional 
Autoencoder

Benvenuti nel fantastico progetto CACCA💩! 🧪✨ Questo non è un autoencoder qualsiasi, è un **Conditional Autoencoder** addestrato per generare nuove molecole 🧬 partendo da stringhe SMILES. Il modello impara la struttura delle molecole e può generarne di nuove con proprietà specifiche!

Questo progetto è stato costruito con amore ❤️, PyTorch 🔥 e PyTorch Geometric 🕸️.

## 🚀 Come Iniziare

Segui questi semplici passaggi per far partire la magia!

### 1. 📦 Prerequisiti e Installazione

Assicurati di avere Python 3.x installato. Poi, apri il tuo terminale e installa tutte le dipendenze necessarie con questo singolo comando:

```bash
pip install torch torch-geometric rdkit-pypi tqdm
```

> 📝 **Nota:** `rdkit` può essere un po' complicato da installare. Se il comando `pip` non funziona, potresti doverlo installare tramite `conda`:
> `conda install -c conda-forge rdkit`

### 2. 🏃‍♀️ Eseguire l'Addestramento

Una volta installato tutto, sei pronto per addestrare il modello! 🧠 Lancia lo script `train.py` dal tuo terminale:

```bash
python train.py
```

Lo script inizierà a scaricare il dataset ZINC 📉, pre-elaborerà i dati e avvierà il ciclo di addestramento. Vedrai l'output della loss di training e di test per ogni epoca. 🤩

I modelli addestrati verranno salvati nella directory `save/` 💾.

#### ⚙️ Argomenti Personalizzati

Puoi personalizzare l'addestramento passando alcuni argomenti. Ecco i più importanti:

-   `--batch_size`: Dimensione del batch (default: 128)
-   `--latent_size`: Dimensione dello spazio latente (default: 200)
-   `--num_epochs`: Numero di epoche di addestramento (default: 100)
-   `--lr`: Learning rate (default: 0.0001)

Esempio con argomenti personalizzati:
```bash
python train.py --batch_size 256 --num_epochs 50 --lr 0.001
```

## 📂 Struttura del Progetto

Ecco una rapida occhiata ai file principali:

-   `model.py` 🧠: Contiene l'architettura del nostro `ConditionalAutoencoder` scritta in PyTorch. Qui vive il cervello 🧠 del progetto!
-   `train.py` 🏋️‍♀️: Gestisce il caricamento dei dati (usando PyTorch Geometric per il dataset ZINC), il pre-processing, il ciclo di addestramento e il salvataggio dei modelli.
-   `utils.py` 🛠️: Contiene funzioni di utilità, come quella per convertire i dati dei grafi di ZINC in stringhe SMILES leggibili.

---

Buona CACCA 💩!
