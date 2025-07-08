# https://chemistry.stackexchange.com/questions/168293/accessing-the-zinc15-and-zinc20-databases
"""
B. Fingerprint → rete di proiezione	Generi un Morgan FP (o RDKit descriptors) del polimero,
poi addestri un piccolo MLP per proiettarlo nello stesso spazio latente del CAE (vedi § 4)
Usa info chimica reale del polimero	Serve un minimissimo fine-tuning (self-supervised)
"""


from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem


def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    # return s, ExactMolWt(m), MolLogP(m), CalcNumHBD(m), CalcNumHBA(m), CalcTPSA(m)
    # return s, ExactMolWt(m), MolLogP(m), CalcTPSA(m)
    # return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcNumHBD(m), CalcNumHBA(m), CalcTPSA(m)


aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
Metoclopramide = "CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl"
Pyramidone = "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)C"
Oestradiol = "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"
Indomethacin = "CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O"


print(f"Aspirin", cal_prop(aspirin))
print(f"ibuprofen", cal_prop(ibuprofen))
print(f"metoclopramide", cal_prop(Metoclopramide))
print(f"pyramidone", cal_prop(Pyramidone))
print(f"oestradiol", cal_prop(Oestradiol))
print(f"indomethacin", cal_prop(Indomethacin))


pol = "OCCN(C(=O)C(c1ccccc1)c1ccccc1)"
print(cal_prop(pol))

# smiles2chemdraw.py


# def smiles_to_molfile(
#     smiles: str,
#     out_dir: str | Path = ".",
#     base_name: str = "molecule",
#     open_chemdraw: bool = False,
#     chemdraw_path: Optional[str] = None,   # es. r"C:\Program Files\ChemDraw\ChemDraw.exe"
# ) -> Path:
#     """
#     • Converte SMILES → MOL + PNG (2-D) in out_dir
#     • Se open_chemdraw=True prova ad aprire il file MOL in ChemDraw
#     Ritorna il Path al file .mol salvato.
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError("SMILES non valido")

#     # Coord 2-D
#     mol = Chem.AddHs(mol)
#     AllChem.Compute2DCoords(mol)
#     mol = Chem.RemoveHs(mol)

#     # Salvataggio MOL
#     mol_path = out_dir / f"{base_name}.mol"
#     Chem.MolToMolFile(mol, str(mol_path))

#     # Salvataggio PNG (anteprima)
#     img_path = out_dir / f"{base_name}.png"
#     img = Draw.MolToImage(mol, size=(400, 300))
#     img.save(img_path)

#     print(f"✔️  Salvato: {mol_path.name} e {img_path.name}")


# if __name__ == "__main__":
#     smiles_to_molfile(
#         pol
#     )
