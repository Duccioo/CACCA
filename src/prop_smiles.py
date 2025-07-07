from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen  import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA
from rdkit import Chem

def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return (
        Chem.MolToSmiles(m),          # SMILES canonico
        ExactMolWt(m),                # Peso molecolare esatto
        MolLogP(m),                   # logP (Crippen)
        CalcNumHBD(m),                # H-bond donors
        CalcNumHBA(m),                # H-bond acceptors
        CalcTPSA(m)                   # TPSA
    )

# ---------- ESEMPIO SU UN UNICO SMILES ----------
smiles = "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"          
res = cal_prop(smiles)

if res is not None:
    (smi, mw, logp, hbd, hba, tpsa) = res
    print(f"SMILES: {smi}")
    print(f"MolWt : {mw:.2f}")
    print(f"logP  : {logp:.2f}")
    print(f"HBD   : {hbd}")
    print(f"HBA   : {hba}")
    print(f"TPSA  : {tpsa:.2f}")
else:
    print("SMILES non valido")


from rdkit import Chem
from rdkit.Chem import Draw

smi1 = '"c1ccccc1C(c2ccccc2)C3NCCO3"'
mol = Chem.MolFromSmiles("c1ccccc1C(c2ccccc2)C3NCCO3")
Draw.MolToImage(mol)
img = Draw.MolToImage(mol, size=(400, 300))
#img.save("polyphox_monomer.png")

res = cal_prop(smiles)

if res is not None:
    (smi1, mw, logp, hbd, hba, tpsa) = res
    print(f"SMILES: {smi1}")
    print(f"MolWt : {mw:.2f}")
    print(f"logP  : {logp:.2f}")
    print(f"HBD   : {hbd}")
    print(f"HBA   : {hba}")
    print(f"TPSA  : {tpsa:.2f}")
else:
    print("SMILES non valido")



mol2 = Chem.MolFromSmiles("c1ccc(C(c2ccccc2)C2NCCO2)cc1")
Draw.MolToImage(mol2)
img = Draw.MolToImage(mol2, size=(400, 300))
#img.save("polyphox_monomer2.png")

mol3 = Chem.MolFromSmiles('c1ccc(cc1)C(c2ccccc2)C3NCCO3')
Draw.MolToImage(mol3)
img = Draw.MolToImage(mol3, size=(400, 300))
img.save("polyphox_monomer3.png")

mol4 = Chem.MolFromSmiles('c1ccccc1C(C2=NCCO2)c3ccccc3')
Draw.MolToImage(mol4)
img = Draw.MolToImage(mol4, size=(400, 300))
img.save("polyphox_monomer4.png")

mol5 = Chem.MolFromSmiles('c1ccccc1C(C2=NCCO2)c3ccccc3')
Draw.MolToImage(mol5)
img = Draw.MolToImage(mol5, size=(400, 300))
img.save("polyphox_monomer5.png")

smi5 = 'c1ccccc1C(C2=NCCO2)c3ccccc3'
if res is not None:
    (smi5, mw, logp, hbd, hba, tpsa) = res
    print(f"SMILES: {smi5}")
    print(f"MolWt : {mw:.2f}")
    print(f"logP  : {logp:.2f}")
    print(f"HBD   : {hbd}")
    print(f"HBA   : {hba}")
    print(f"TPSA  : {tpsa:.2f}")
else:
    print("SMILES non valido")