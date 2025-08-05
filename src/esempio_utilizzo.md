# Utilizzo dello Script PCA Migliorato

## Descrizione

Lo script `pca.py` è stato migliorato per offrire flessibilità nell'analisi chimica. Ora puoi scegliere se utilizzare solo i descrittori chimici o includere anche i fingerprint Morgan.

## Opzioni Disponibili

### 1. Solo Descrittori Chimici (Default)
```bash
python pca.py
```
Questo comando utilizza solo i 5 descrittori chimici principali:
- LogP (Partition coefficient)
- TPSA (Topological Polar Surface Area)
- NumHDonors (Numero di donatori di idrogeno)
- NumHAcceptors (Numero di accettori di idrogeno)
- ExactMolWt (Peso molecolare esatto)

### 2. Descrittori + Fingerprint Morgan
```bash
python pca.py --use-fingerprints
```
Questo comando include sia i descrittori che i fingerprint Morgan compressi.

### 3. Personalizzazione Componenti PCA per Fingerprint
```bash
python pca.py --use-fingerprints --n-fp-components 15
```
Permette di specificare quante componenti PCA utilizzare per comprimere i fingerprint (default: 10).

## Output

Lo script genera:
- Un grafico UMAP salvato come PNG
- Informazioni dettagliate sul processo di elaborazione
- Statistiche finali sull'analisi

## Miglioramenti Implementati

1. **Argomenti da linea di comando**: Possibilità di scegliere le feature da utilizzare
2. **Plot migliorati con seaborn**: Stile più professionale e leggibile
3. **Annotazioni migliorate**: Etichette con sfondo per maggiore leggibilità
4. **Gestione modulare**: Codice più organizzato e mantenibile
5. **Feedback dettagliato**: Informazioni chiare sui parametri utilizzati

## Esempio di Output

```
Modalità: Solo Descrittori
Elaborazione delle molecole...
Processate 25 molecole con successo.
Matrice finale: 5 feature (Solo Descrittori)
Applicazione di UMAP...

Grafico salvato come 'umap_plot_descriptors_only.png'

Analisi completata utilizzando: Solo Descrittori
Numero totale di feature utilizzate: 5
Numero di molecole analizzate: 25
```
