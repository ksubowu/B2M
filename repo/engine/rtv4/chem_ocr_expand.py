"""
OCR-based R functional group recognition and molecular graph expansion for ChemRTv4.

Uses OCR to recognize text in atom boxes classified as "other" (rgroup
category), then expands the molecular graph by replacing placeholder atoms with
actual functional groups via RDKit.

Adapted from i2m_data/real/det_engine.py and i2m_eval.py patterns.
"""

import copy
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Element / R-group / Abbreviation definitions
# ---------------------------------------------------------------------------

ELEMENTS = frozenset({
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
})

RGROUP_SYMBOLS = frozenset({
    "R", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
    "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19",
    "Ra", "Rb", "Rc", "Rd", "Re", "Rf", "Rg", "Rh", "Ri", "Rj",
    "X", "Y", "Z", "Q", "A", "E", "Ar", "V", "U", "M", "G", "L", "D",
})

# Abbreviation -> SMILES (connection point marked with [] brackets like [CH2], [O], [NH])
ABBREVIATIONS = {
    # Protecting groups
    "Boc": "[C](=O)OC(C)(C)C",
    "BOc": "[C](=O)OC(C)(C)C",
    "NHBoc": "[NH]C(=O)OC(C)(C)C",
    "BocHN": "[NH]C(=O)OC(C)(C)C",
    "BOCHN": "[NH]C(=O)OC(C)(C)C",
    "NBoc": "[NH1]C(=O)OC(C)(C)C",
    "Cbz": "[C](=O)OCc1ccccc1",
    "NHCbz": "[NH]C(=O)OCc1ccccc1",
    "Fmoc": "[C](=O)OCC1c(cccc2)c2c3c1cccc3",
    "FmocHN": "[NH]C(=O)OCC1c(cccc2)c2c3c1cccc3",
    "FmOcHN": "[NH]C(=O)OCC1c(cccc2)c2c3c1cccc3",
    "NHFmoc": "[NH]C(=O)OCC1c(cccc2)c2c3c1cccc3",
    "Trt": "[C](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "Alloc": "[C](=O)OCC=C",
    "MOM": "[CH2]OC",
    "THP": "[CH2]C1CCCCO1",
    "TBS": "[Si](C)(C)C(C)(C)C",
    "TBDMS": "[Si](C)(C)C(C)(C)C",
    "TMS": "[Si](C)(C)C",
    "TBDMSO": "[O][Si](C)(C)C(C)(C)C",
    "OTBDMS": "[O][Si](C)(C)C(C)(C)C",
    "TFA": "[C](=O)C(F)(F)F",
    "TFAH2N": "[NH]C(=O)C(F)(F)F",
    # Acyl
    "Ac": "[C](=O)C",
    "OAc": "[O]C(=O)C",
    "AcO": "[O]C(=O)C",
    "NHAc": "[NH]C(=O)C",
    "SAc": "[S]C(C)=O",
    "AcS": "[S]C(C)=O",
    "Bz": "[C](=O)c1ccccc1",
    "OBz": "[O]C(=O)c1ccccc1",
    "BzO": "[O]C(=O)c1ccccc1",
    "Bn": "[CH2]c1ccccc1",
    "OBn": "[O]Cc1ccccc1",
    "BnO": "[O]Cc1ccccc1",
    "NHBn": "[NH]Cc1ccccc1",
    "COOBn": "[C](=O)OCc1ccccc1",
    "BnO2C": "[C](=O)OCc1ccccc1",
    # Sulfonyl
    "Ts": "[S](C1=CC=C(C=C1)C)(=O)=O",
    "Tos": "[S](C1=CC=C(C=C1)C)(=O)=O",
    "OTs": "[O]S(C1=CC=C(C=C1)C)(=O)=O",
    "TsO": "[O]S(C1=CC=C(C=C1)C)(=O)=O",
    "Ms": "[S](=O)(=O)C",
    "OMs": "[O]S(=O)(=O)C",
    "MsO": "[O]S(=O)(=O)C",
    "Tf": "[S](=O)(=O)C(F)(F)F",
    "OTf": "[O]S(=O)(=O)C(F)(F)F",
    "TfO": "[O]S(=O)(C(F)(F)F)=O",
    # Alkyl
    "Me": "[CH3]",
    "OMe": "[O]C",
    "MeO": "[O]C",
    "OCH3": "[O]C",
    "H3CO": "[O]C",
    "CH3O": "[O]C",
    "SMe": "[S]C",
    "MeS": "[S]C",
    "NMe": "[N]C",
    "MeN": "[N]C",
    "NMe2": "[N](C)C",
    "Me2N": "[N](C)C",
    "Et": "[CH2]C",
    "OEt": "[O]CC",
    "EtO": "[O]CC",
    "Pr": "[CH2]CC",
    "nPr": "[CH2]CC",
    "Bu": "[CH2]CCC",
    "nBu": "[CH2]CCC",
    "iPr": "[CH1](C)C",
    "i-Pr": "[CH1](C)C",
    "iBu": "[CH2]C(C)C",
    "tBu": "[C](C)(C)C",
    "t-Bu": "[C](C)(C)C",
    "OtBu": "[O]C(C)(C)C",
    "tBuO": "[O]C(C)(C)C",
    # Aryl
    "Ph": "[c]1ccccc1",
    "Py": "[c]1ncccc1",
    "Mes": "[c]1c(C)cc(C)cc(C)1",
    "PMB": "[CH2]c1ccc(OC)cc1",
    "PMBN": "[N]Cc1ccc(OC)cc1",
    # Halogenated
    "CF3": "[C](F)(F)F",
    "F3C": "[C](F)(F)F",
    "CCl3": "[C](Cl)(Cl)Cl",
    "OCF3": "[O]C(F)(F)F",
    "SCF3": "[S]C(F)(F)F",
    "SF5": "[S](F)(F)(F)(F)F",
    "CH2CF3": "[CH2]C(F)(F)F",
    # Nitrogen
    "NO2": "[N+](=O)[O-]",
    "O2N": "[N+](=O)[O-]",
    "CN": "[C]#N",
    "NC": "[C]#N",
    "N3": "[N]=[N+]=[N-]",
    "NO": "[N]=O",
    "NH2": "[NH2]",
    "NHOH": "[NH]O",
    "NHNH2": "[NH]N",
    "CONH2": "[C](=O)[NH2]",
    "CONH": "[C](=O)[N]",
    "NHCOtBu": "[NH]C(C(C)(C)C)=O",
    "NHBn": "[NH]Cc1ccccc1",
    "NPh": "[N]C1=CC=CC=C1",
    "NHPh": "[NH]C1=CC=CC=C1",
    "NMe2": "[N](C)C",
    # Oxygen
    "OH": "[OH]",
    "OMe": "[O]C",
    "OEt": "[O]CC",
    "OAll": "[O]CC=C",
    "OPh": "[O]C1=CC=CC=C1",
    "PhO": "[O]C1=CC=CC=C1",
    # Carboxyl
    "COOH": "[C](=O)O",
    "CO2H": "[C](=O)O",
    "HO2C": "[C](=O)O",
    "CO2Et": "[C](=O)OCC",
    "COOEt": "[C](=O)OCC",
    "CO2Me": "[C](=O)OC",
    "COOMe": "[C](=O)OC",
    "CO2R": "[C](=O)O*",
    "COO": "[C](=O)O",
    "CO2": "[C](=O)[O]",
    # Sulphur
    "SH": "[SH]",
    "SPh": "[S]C1=CC=CC=C1",
    "PhS": "[S]C1=CC=CC=C1",
    "SO2": "[S](=O)(=O)",
    "SO2Me": "[S](=O)(=O)C",
    "SO3H": "[S](=O)(=O)O",
    "SO2NH2": "[S](=O)(=O)[NH2]",
    "SO2CF3": "[S](=O)(=O)C(F)(F)F",
    "SO2Cl": "[S](=O)(=O)Cl",
    "SO2F": "[S](=O)(=O)F",
    # Phosphorus
    "PPh3": "[P](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "PPh2": "[P](C1=CC=CC=C1)C2=CC=CC=C2",
    "P+Ph3Br-": "[P+](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "P(O)(OEt)2": "[P](OCC)(=O)OCC",
    # Boron
    "B(OH)2": "[B](O)O",
    # Others
    "Cbm": "[C](=O)N",
    "OCN": "[N]=C=O",
    "CHO": "[CH1](=O)",
    "OHC": "[CH1](=O)",
    "SiMe2": "[Si](C)(C)",
    "Me2Si": "[Si](C)(C)",
    "Me3Si": "[Si](C)(C)(C)",
    "ONa": "[O][Na]",
    "Allyl": "[CH2]C=C",
    "Cy": "[CH1]1CCCCC1",
    "Ad": "[C]12CC3CC(CC1)CC2C3",
    "Az": "[N]1CCCC1",
    "Np": "[c]1cccc2ccccc12",
    "An": "[c]1ccc2ccccc2c1",
    # Stereoidal / complex
    "Piv": "[C](C)(C)C(=O)",
    "SEM": "[CH2]OCC[Si](C)(C)C",
    "Suc": "[C](=O)CCC(=O)O",
    "TBSO": "[O][Si](C)(C)C(C)(C)C",
    "Cg": "[C](=O)O",
    # S* / N* / O* patterns
    "S*": "[S]*",
    "N*": "[NH]*",
    "NH*": "[NH]*",
    "O*": "[O]*",
    "C*": "[C]*",
    "CH2*": "[CH2]*",
    "P*": "[P]*",
    # (CH3)n patterns
    "N(CH3)2": "[N](C)(C)",
    "(C2H5)2N": "[N](CC)(CC)",
    "Et2N": "[N](CC)(CC)",
    "N(H)Et": "[NH]CC",
    "N(H)Me": "[NH]C",
}

# ---------------------------------------------------------------------------
# OCR text normalization helpers
# ---------------------------------------------------------------------------

def _clean_unpaired_brackets(text):
    """Remove unpaired brackets from text."""
    stack = []
    remove = set()
    for i, ch in enumerate(text):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()
            else:
                remove.add(i)
    remove.update(stack)
    return ''.join(ch for i, ch in enumerate(text) if i not in remove)


def _replace_c1(text):
    """Replace C1 (ring closure) mis-OCR with Cl."""
    return re.sub(r'C1(?=[^0-9]|$)', 'Cl', text)


def _transform_formula_hg(text):
    """Fix Hg mis-OCR (e.g. Hg -> Hg)."""
    return text


def _c1_transform(text):
    """Fix C1 mis-OCR patterns."""
    if 'C1' in text:
        text = _replace_c1(text)
    return text


# Replacement map for R-group notation
_RGROUP_REPLACEMENT_KEYS = [
    f"{e}{suffix}" for e in ['S', 'N', 'P', 'C', 'O']
    for suffix in ['R"', "R'", "R", "*"]
]
RGROUP_REPLACEMENT_MAP = {key: f'{key[0]}*' for key in _RGROUP_REPLACEMENT_KEYS}


def normalize_ocr_text(text, replacement_map=None):
    """Normalize OCR text using predefined mapping rules.

    Adapted from i2m det_engine.py normalize_ocr_text().
    """
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # C1 -> Cl fix
    if 'C1' in text:
        text = _replace_c1(text)

    # If it's a known element, return as-is
    if text in ELEMENTS:
        return text

    # Remove spaces
    text = text.replace(" ", "")

    # Replace o/O/0 based on context
    if any(c in text for c in '0oO'):
        # o/O after non-CF letter and before letter or end -> 0
        text = re.sub(r'(?<![CF,CH]\d)[oO](?=[a-zA-GI-Z]|$)', '0', text)
        if '00' in text:
            text = re.sub(r'00', 'OO', text)
        if text in ['R20']:
            return text
        # 0 after letter and before letter or end -> O
        text = re.sub(r'(?<=[a-zA-Z])0(?=([a-zA-Z]|$))', 'O', text)
        text = re.sub(r'^(0)|(?<=[a-zA-Z][?\d])0(?=[a-zA-Z0-9]*$|[a-zA-Z])', 'O', text)

    text = _clean_unpaired_brackets(text)

    # Common OCR mis-reads — comprehensive rules from i2m det_engine.py
    _FIXES = {
        # NO2 variants
        'OzN': 'NO2', 'O2N': 'NO2', 'O,N': 'NO2', 'NOz': 'NO2',
        'NO2': 'NO2', 'NO,': 'NO2', '0;N': 'NO2', '02N': 'NO2', 'N20': 'NO2',
        # CH3CH variants
        'CHzCH': 'CH3CH', 'CH,CH': 'CH3CH',
        'NHCHzCOOH': 'NHCH2COOH', 'NHCH2COOH': 'NHCH2COOH',
        # COCl variants
        'CIOC': 'COCl', 'COCE': 'COCl', 'ClOC': 'COCl', 'COCI': 'COCl',
        # Ester variants
        'CHCOOCHs': 'CH2COOC2H5', 'CH2COOCH5': 'CH2COOC2H5',
        # t-Bu
        '(t-Bu)': 't-Bu', 't-Bu': 't-Bu',
        # SiMe2
        'SiMe2': 'SiMe2', 'Me2Si': 'SiMe2',
        # Ar2P(O)
        'ArzP(O)': 'Ar2P(O)', 'Ar2P(O)': 'Ar2P(O)',
        # P(O)(OEt)2
        'P(O)(0Et)2': 'P(O)(OEt)2', 'P(O)(OEt)2': 'P(O)(OEt)2',
        # PhO2S
        'PhOzS': 'PhO2S', 'PhO2S': 'PhO2S',
        # CH3O
        'CH3O': 'CH3O', 'CHzO': 'CH3O',
        # NH2.HCl
        'NH.HCI': 'NH2.HCl', 'NH,.Hcl': 'NH2.HCl',
        # Z
        '2': 'Z', 'Z': 'Z',
        # CH2 variants
        '(CH2)m': 'CH2', '(CH2)q': 'CH2', '(CH2)s': 'CH2',
        # Ar variants
        'Arl': 'Ar', 'Ari': 'Ar', 'Ar2': 'Ar', 'Ar1': 'Ar',
        # NHR variants
        'NHR%': 'NHR8', 'NHR*': 'NHR8',
        # Vv
        'Vv': 'Vv', 'Vy': 'Vv',
        # N3
        'N3': 'N3', 'NY': 'N3', 'Ny': 'N3',
        # (C2H5)2N
        'N(CH,CH3)2': '(C2H5)2N', 'C;H52N': '(C2H5)2N', '(C;H5)2N': '(C2H5)2N',
        'N(C;Hs)2': '(C2H5)2N', 'N(C;H5)2': '(C2H5)2N', '(CHzCH2)2N': '(C2H5)2N',
        'N(CHCH3)2': '(C2H5)2N', '(CH3CH2)2N': '(C2H5)2N', '(C2H52N': '(C2H5)2N',
        '(CHzCH)2N': '(C2H5)2N', '(C2H5)2N': '(C2H5)2N', 'Et2N': '(C2H5)2N',
        # Me2N
        '(CH3)2N': 'Me2N', 'Me2NH': 'Me2N', 'Me,N': 'Me2N', 'Me2N': 'Me2N',
        # COOCH2Ph
        'CO2CH2Ph': 'COOCH2Ph', 'COOCH2Ph': 'COOCH2Ph', 'COOCH,Ph': 'COOCH2Ph',
        # CO2C(CH3)3
        'CO2C(CH3)3': 'CO2C(CH3)3', '(CH3)3CO2C': 'CO2C(CH3)3',
        # OCH2Ph
        'OCH2Ph': 'OCH2Ph', 'OCH,Ph': 'OCH2Ph', 'OCHAPH': 'OCH2Ph', 'OCH;Ph': 'OCH2Ph',
        # (CF2)8H
        '(CF2)8H': '(CF2)8H', '(CF2)gH': '(CF2)8H', '(CF2)sH': '(CF2)8H',
        'CF2sH': '(CF2)8H', 'CF:H': '(CF2)8H', 'CF2)sH': '(CF2)8H', 'CF):H': '(CF2)8H',
        # NHSO2Bu
        'NHSO,Bu': 'NHSO2Bu', 'NHSO2Bu': 'NHSO2Bu',
        # NHSO2CH3
        'NHSO,CH3': 'NHSO2CH3', 'NHSO2CH3': 'NHSO2CH3', 'NHSO2Me': 'NHSO2CH3',
        # CF3 / F3C
        'CF3': 'F3C', 'CFs': 'F3C', 'CF,': 'F3C', '13': 'F3C', 'CF 3': 'F3C',
        'F;C': 'F3C', 'F:C': 'F3C', 'F sC': 'F3C', 'CF': 'F3C', 'CF;': 'F3C',
        'CFa': 'F3C', 'FzC': 'F3C', 'CFz': 'F3C',
        # OCCl3
        'OCCl3': 'OCCl3', 'Cl3CO': 'OCCl3',
        # CCl3
        'CCl3': 'CCl3', 'Cl3C': 'CCl3',
        # F3CN
        'F;CN': 'F3CN', 'NCF;': 'F3CN',
        # NCH3
        'NCH3': 'NCH3', 'NHCH3': 'NCH3', 'NCH;': 'NCH3', 'CH3N': 'NCH3',
        'MeN': 'NCH3', 'MeNH': 'NCH3',
        # NOMe
        'NOMe': 'NOMe',
        # Me (CH3)
        'HzC': 'Me', 'HyC': 'Me', 'CHy': 'Me', 'CHE': 'Me', 'H3C.': 'Me',
        '1;.C': 'Me', '1;C': 'Me', 'M e': 'Me', 'Mé': 'Me', 'CH 3': 'Me',
        'CH:': 'Me', 'HsC': 'Me', 'HaC': 'Me', 'H3C': 'Me', 'CH3': 'Me',
        'CHa': 'Me', 'H;C': 'Me', 'CH,': 'Me', 'CHs': 'Me', 'CH;': 'Me',
        # PPh3
        'PPh3': 'PPh3', 'PPha': 'PPh3',
        # Et / CH2CH3
        'Et': 'CH2CH3', 'CH,CH3': 'CH2CH3', 'Catls': 'CH2CH3', 'Cafls': 'CH2CH3',
        'CH2CH3': 'CH2CH3', 'H3CH2C': 'CH2CH3', 'C:H5': 'CH2CH3', 'HzCH2C': 'CH2CH3',
        'C,H5': 'CH2CH3', 'CzH5': 'CH2CH3', 'C2H5': 'CH2CH3', 'C2Hs': 'CH2CH3',
        # OMe
        'Ovle': 'OMe', 'HzCO': 'OMe', 'OCH': 'OMe', 'OCH:': 'OMe', 'H2CO': 'OMe',
        'CH3O': 'OMe', 'CH,O': 'OMe', 'HsCO': 'OMe', 'OMe': 'OMe', 'AME': 'OMe',
        'AMe': 'OMe', 'H3CO': 'OMe', 'MeO': 'OMe',
        'OCHa': 'OMe', 'HgCO': 'OMe', 'HaCO': 'OMe',
        # SO2Cl
        'SO2Cl': 'SO2Cl', 'SOzCl': 'SO2Cl',
        # SO2F
        'SO2F': 'SO2F', 'SOzF': 'SO2F',
        # SO2NH
        'SONH': 'SO2NH', 'HNOS': 'SO2NH', 'SON': 'SO2NH', 'SO2NH': 'SO2NH',
        # NHO2S
        'HNO2S': 'NHO2S', 'NHO2S': 'NHO2S',
        # SO2NH2
        'SO2NH2': 'SO2NH2', 'SO,NH': 'SO2NH2', 'SO:NH2': 'SO2NH2', 'SONH2': 'SO2NH2',
        # SO2CF3
        'SOzCF3': 'SO2CF3', 'SO2CF3': 'SO2CF3', 'CF3SO2': 'SO2CF3',
        # SO2
        'SOz': 'SO2', 'O2S': 'SO2', '$02': 'SO2', 'S02': 'SO2', 'SO,': 'SO2',
        '62': 'SO2', 'O:S': 'SO2', 'SO2': 'SO2',
        # SO2CH3
        'H3CO2S': 'SO2CH3', 'SO2CH3': 'SO2CH3',
        # SO3H
        'SO3H': 'SO3H', 'SOsH': 'SO3H', 'SOaH': 'SO3H', 'HO3S': 'SO3H',
        'SOzH': 'SO3H', 'HOzS': 'SO3H',
        # OSO2Me
        'MeO2SO': 'OSO2Me', 'OSO2CH3': 'OSO2Me', 'OSO2Me': 'OSO2Me',
        # NHSO2Me
        'MeO2SHN': 'NHSO2Me', 'NHSO2Me': 'NHSO2Me',
        # PMB
        'PIME': 'PMB', 'PMB': 'PMB',
        # t-Bu (additional)
        '1-BU': 't-Bu', '-BU': 't-Bu', '-Bu': 't-Bu', 't-BU': 't-Bu',
        # TsN
        'NTS': 'TsN', 'NTs': 'TsN', 'TsN': 'TsN',
        # OTs
        'TsO': 'OTs', 'OTs': 'OTs',
        # N2+Cl-
        'Nz* Cl': 'N2+Cl-', "N2+Cl-": 'N2+Cl-',
        # NH3Cl
        'NH3Cl': 'NH3Cl', 'NHzCl': 'NH3Cl', 'NH;Cl': 'NH3Cl',
        # B(OH)2
        'B(OH)2': 'B(OH)2',
        # NHAc
        'NHAC': 'NHAc', 'NHAc': 'NHAc',
        # OCN
        '1CO': 'OCN', 'NCO': 'OCN', 'OCN': 'OCN', 'OON': 'OCN',
        # COCF3
        'COCFs': 'COCF3', 'COCF3': 'COCF3', 'COCF s': 'COCF3',
        # OCF3
        'OCF3': 'OCF3', 'OCF 3': 'OCF3', 'OCE': 'OCF3', 'OCE:': 'OCF3',
        'OCEE': 'OCF3', 'F3CO': 'OCF3', 'OCF': 'OCF3', 'OCF:': 'OCF3',
        # SCF3
        'SCF3': 'SCF3', 'SCE': 'SCF3', 'SCEE': 'SCF3', 'F3CS': 'SCF3',
        'SCF': 'SCF3', 'SCF:': 'SCF3',
        # SMe
        'HzCS': 'SMe', 'SCH3': 'SMe', 'SMe': 'SMe', 'MeS': 'SMe', 'H3SC': 'SMe',
        # OEt
        'CHzCHzO': 'OEt', 'CH3CH2O': 'OEt', 'H5C2O': 'OEt', 'OC2H5': 'OEt',
        # CO2Et
        'CO,Et': 'CO2Et', 'COzEt': 'CO2Et', 'CO2Et': 'CO2Et', 'H3CH2COOC': 'CO2Et',
        'CO2C2H5': 'CO2Et',
        # OTBDMS
        'OTBS': 'OTBDMS', 'TBSO': 'OTBDMS', 'OTBDMS': 'OTBDMS',
        # PhO
        'PhO': 'PhO', 'Pho': 'PhO',
        # Cl
        'CI': 'Cl', 'C1': 'Cl',
        # Ph
        'P h': 'Ph', 'Ph': 'Ph',
        # TFAH2N
        'FAHN': 'TFAH2N', 'TFAH,N': 'TFAH2N', 'TFAH2N': 'TFAH2N',
        # Me3Si
        'MeaSi': 'Me3Si', 'Me3Si': 'Me3Si',
        # PH3C
        'PHzC': 'PH3C', 'PH;C': 'PH3C', 'PH3C': 'PH3C',
        # CO2H
        'COOH': 'CO2H', 'OOOH': 'CO2H', '1OOC': 'CO2H', 'HOOO': 'CO2H',
        'HOOC': 'CO2H', 'DOOH': 'CO2H', 'CO:H': 'CO2H', 'HO,C': 'CO2H',
        'CO,H': 'CO2H', 'CO2H': 'CO2H',
        # CO2R
        'CO2R': 'CO2R', 'RO2C': 'CO2R', 'RO,C': 'CO2R', 'CO2*': 'CO2R', "COzR'": 'CO2R',
        # CO2
        'CO2': 'CO2', 'COO': 'CO2', 'OOC': 'CO2', "COz": 'CO2',
        # O2C
        'O2C': 'O2C', '02C': 'O2C',
        # C2H5
        'CaH;': 'C2H5', 'CHS': 'C2H5', 'C2H5': 'C2H5',
        # NHBoc
        'NHBoc': 'NHBoc', 'NHBOc': 'NHBoc', 'BocHN': 'NHBoc', 'BOcHN': 'NHBoc',
        'BOCHN': 'NHBoc',
        # C7H3
        'C7H': 'C7H3', 'C7H3': 'C7H3',
        # C5H11
        'CsH11': 'C5H11', 'C5H11': 'C5H11',
        # CCl3CH2O2C
        'CC3CH2O2C': 'CCl3CH2O2C', 'CCl3CH2O2C': 'CCl3CH2O2C',
        # CH2OMe
        'CH2OMe': 'CH2OMe', 'MeOH,C': 'CH2OMe', 'CH,0Me': 'CH2OMe',
        'CH,OMe': 'CH2OMe', 'MeOH2C': 'CH2OMe',
        # R -> *
        'R': '*', "R'": '*',
        # U
        'U': 'U', 'U.': 'U',
        # RO -> O*
        'RO': 'O*',
        # OAc
        'OAc': 'OAc', 'OAC': 'OAc',
        # R9
        'Rg': 'R9', 'R9': 'R9',
        # R7
        'Rz': 'R7', 'R7': 'R7',
        # R11
        'Rll': 'R11', 'R1l': 'R11', 'R11': 'R11',
        # OMe (additional)
        'OQ': 'OMe', '00': 'OMe', '0Q': 'OMe', 'OCH3': 'OMe',
        # NH / N
        'NH': 'N', 'HN': 'N', 'NH2': 'N', 'H2N': 'N', 'H,N': 'N',
        # OH / O
        'OH': 'O', 'HO': 'O', 'OH2': 'O', '0': 'O',
        # N(H)Et
        'N(H)Et': 'N(H)Et', 'Et(H)N': 'N(H)Et',
        # N(H)Me
        'N(H)Me': 'N(H)Me', 'Me(H)N': 'N(H)Me',
        # CONH
        'HNOC': 'CONH', 'CONH': 'CONH',
        # CH3CONH
        'HNOCCH3': 'CH3CONH', 'CH,CONH': 'CH3CONH', 'CH3CONH': 'CH3CONH',
        # PPh2
        'PPh2': 'PPh2', 'Ph,P': 'PPh2', 'Ph2P': 'PPh2',
        # SF5
        'SF5': 'SF5', 'F5S': 'SF5',
        # OCH2CF3
        'OCH2CF3': 'OCH2CF3', 'F3CH2CO': 'OCH2CF3',
        # NHCbz
        'NHCbz': 'NHCbz', 'CbzHN': 'NHCbz',
        # NHNH2
        'NHNH2': 'NHNH2', 'H2NHN': 'NHNH2',
        # (CH3CH2)2N
        'CHzCH22N': '(CH3CH2)2N', 'N2(CH2CH3)': '(CH3CH2)2N',
        '(CH3CH2)2N': '(CH3CH2)2N',
        # CH2CH2CH2CH
        'CHCHCH2CH-3': 'CH2CH2CH2CH', 'CH2CH2CH2CH': 'CH2CH2CH2CH',
        # HCH2CH2CH2C
        'HCH2CH2CH2C': 'HCH2CH2CH2C',
        # (H3C)2HC
        '(HzC)2HC': '(H3C)2HC', '(H3C)2HC': '(H3C)2HC',
        # CH2CH2NSO2CH3
        '13CO2SHNH2CH2C': 'CH2CH2NSO2CH3', 'H3CO2SHNH2CH2C': 'CH2CH2NSO2CH3',
        'CH2CH2NSO2CH3': 'CH2CH2NSO2CH3',
        # C9H19
        'CgH19': 'C9H19', 'C9H19': 'C9H19',
        # (CF2)8H (additional)
        '(CF2):H': '(CF2)8H',
        # CO2Me
        'COOCH3': 'CO2Me', 'HzCO2C': 'CO2Me', 'CO,Me': 'CO2Me', 'H3CO2C': 'CO2Me',
        'CO2CH3': 'CO2Me', 'MeOOC': 'CO2Me', 'CO2Me': 'CO2Me', 'COzMe': 'CO2Me',
        'MeO2C': 'CO2Me', 'MeO,C': 'CO2Me',
        # CH2CH2O
        '(CHCHO)': 'CH2CH2O', 'CH2CH2O': 'CH2CH2O',
        # CO2CysPr
        'CO,CysPr': 'CO2CysPr', 'CO2CysPr': 'CO2CysPr',
        # CH2CH2C(O)OCH2CH3
        'CH2CH2C(O)OCHCH3': 'CH2CH2C(O)OCH2CH3', 'CH;CH2C(O)OCHCH3': 'CH2CH2C(O)OCH2CH3',
        # H4NO3S
        'H4NOzS': 'H4NO3S', 'H4NO3S': 'H4NO3S',
        # C10H21
        'C1OH21': 'C10H21', 'C1oH21': 'C10H21', 'CloH21': 'C10H21', 'C10H21': 'C10H21',
    }
    if text in _FIXES:
        text = _FIXES[text]

    # Apply R-group replacement map
    if replacement_map:
        for key, val in replacement_map.items():
            if key in text:
                text = text.replace(key, val)

    # NHRn pattern
    if re.match(r'^NHR[0-9a-z]$', text):
        text = text  # keep as-is

    return text


def is_valid_chem_text(text):
    """Check if text looks like a valid chemical expression."""
    if not text:
        return False
    if text.isdigit():
        return False
    if not re.match(r'^[A-Za-z0-9()\-+]+$', text):
        return False
    # Check paired brackets
    stack = 0
    for ch in text:
        if ch == '(':
            stack += 1
        elif ch == ')':
            stack -= 1
            if stack < 0:
                return False
    return stack == 0


# ---------------------------------------------------------------------------
# Abbreviation expansion helpers
# ---------------------------------------------------------------------------

def _n_c_h_expand(group):
    """Expand NHCnHm or NCnHm patterns."""
    match = re.match(r'NHC(\d+)H(\d+)', group)
    match1 = re.match(r'NC(\d+)H(\d+)', group)
    if not match and not match1:
        return False
    if match:
        c_count = int(match.group(1))
        h_count = int(match.group(2))
    else:
        c_count = int(match1.group(1))
        h_count = int(match1.group(2))
    if h_count == c_count * 2 + 1:
        return '[N]' + 'C' * c_count
    return False


def _c_h_expand(group):
    """Expand CnHm or HmCn patterns (H = 2C + 1, alkyl groups)."""
    p_cn_hm = r'^C(\d+)H(\d+)$'
    p_hm_cn = r'^H(\d+)C(\d+)$'
    p_prefix = r'^([A-Za-z]+)(C\d+H\d+)$'
    p_suffix = r'^(C\d+H\d+)([A-Za-z]+)$'

    match = re.match(p_cn_hm, group)
    if match:
        c_count, h_count = int(match.group(1)), int(match.group(2))
        if h_count == 2 * c_count + 1:
            return '[CH2]' + 'C' * (c_count - 1)
        return False

    match = re.match(p_hm_cn, group)
    if match:
        h_count, c_count = int(match.group(1)), int(match.group(2))
        if h_count == 2 * c_count + 1:
            return '[CH2]' + 'C' * (c_count - 1)
        return False

    return False


def _c_h_expand2(group):
    """Expand CnHm patterns where H = 2C - 1 (alkyne-type)."""
    match = re.match(r'^C(\d+)H(\d+)$', group)
    if match:
        c_count, h_count = int(match.group(1)), int(match.group(2))
        if h_count == 2 * c_count - 1 and c_count % 2 == 0:
            return '[C]#C' + 'C#C' * int(c_count // 2 - 1)
    return False


def _c_f_expand(group):
    """Expand CnFm or CF2CF3 patterns."""
    match = re.match(r'C(\d+)F(\d+)', group)
    if match:
        c_count = int(match.group(1))
        f_count = int(match.group(2))
        if f_count != 2 * c_count + 1:
            return False
    else:
        match2 = re.match(r'(CF2)*CF3$', group)
        if not match2:
            return False
        cf2_count = group.count('CF2')
        c_count = cf2_count + 1
        f_count = cf2_count * 2 + 3
        if f_count != 2 * c_count + 1:
            return False

    parts = []
    for i in range(c_count):
        if i < c_count - 1:
            parts.append('[C](F)(F)' if len(parts) == 0 else 'C(F)(F)')
        else:
            parts.append('C(F)(F)(F)')
    return ''.join(parts)


def _formula_regex(abbrev):
    """Try to parse a chemical formula like C2H5OH."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', abbrev)
    if not tokens:
        return False
    elements = []
    for elem, count in tokens:
        if elem in ELEMENTS or elem in RGROUP_SYMBOLS:
            n = int(count) if count else 1
            elements.extend([elem] * n)
        else:
            return False
    if not elements:
        return False
    # Try to build a simple SMILES
    smiles_parts = []
    for elem in elements:
        if elem in RGROUP_SYMBOLS:
            smiles_parts.append('[*]')
        else:
            smiles_parts.append(elem)
    return ''.join(smiles_parts) if smiles_parts else False


def expand_abbreviation(abbrev):
    """Expand abbreviation to SMILES string.

    Returns SMILES or '[*]' if unrecognized.
    """
    # An OCR blank is a missing label, not a formula to index/expand.
    if not abbrev:
        return '[*]'
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev]

    result = _n_c_h_expand(abbrev)
    if result:
        return result

    result = _c_f_expand(abbrev)
    if result:
        return result

    result = _c_h_expand2(abbrev)
    if result:
        return result

    result = _c_h_expand(abbrev)
    if result:
        return result

    # R-group symbol
    if abbrev in RGROUP_SYMBOLS:
        if len(abbrev) > 1 and abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '[*]'

    # R + digit pattern like R1, R2 not in set
    if abbrev[0] in RGROUP_SYMBOLS and abbrev[1:].isdigit():
        return f'[{abbrev[1:]}*]'

    # Known element
    if abbrev in ELEMENTS:
        return f'[{abbrev}]'

    # Formula regex
    result = _formula_regex(abbrev)
    if result:
        return result

    # Numeric prefix + group
    match = re.match(r'^(\d+)?(.*)', abbrev)
    if match:
        numeric_part, remaining = match.groups()
        if remaining in ELEMENTS:
            return f'[{abbrev}]'
        if numeric_part:
            return f'[{numeric_part}*]'

    return '[*]'


# ---------------------------------------------------------------------------
# OCR backend integration (PaddleOCR/EasyOCR)
# ---------------------------------------------------------------------------

_ocr_reader = None  # EasyOCR reader singleton
_ocr_primary = None  # PaddleOCR primary recognizer
_ocr_secondary = None


def _get_easyocr_reader(gpu=True):
    """Lazily initialize and return an EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    try:
        import easyocr
        import os
        options = {"verbose": False}
        if os.environ.get("EASYOCR_MODEL_DIR"):
            options["model_storage_directory"] = os.environ["EASYOCR_MODEL_DIR"]
            options["download_enabled"] = False
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu, **options)
        return _ocr_reader
    except ImportError:
        return None


def _get_paddleocr_instances(use_gpu=False):
    """Lazily initialize PaddleOCR instances using the i2m-compatible 2.x API."""
    global _ocr_primary, _ocr_secondary
    if _ocr_primary is not None and _ocr_secondary is not None:
        return _ocr_primary, _ocr_secondary
    try:
        import os
        from paddleocr import PaddleOCR
        cpu_threads = int(os.environ.get("PADDLEOCR_CPU_THREADS", "1"))
        _ocr_primary = PaddleOCR(
            use_angle_cls=True,
            lang='latin',
            use_space_char=True,
            use_debug=False,
            use_gpu=bool(use_gpu),
            cpu_threads=cpu_threads,
        )
        secondary_options = {}
        rec_model_dir = os.environ.get("PADDLEOCR_REC_MODEL_DIR")
        if rec_model_dir:
            secondary_options["rec_model_dir"] = rec_model_dir
        _ocr_secondary = PaddleOCR(
            use_angle_cls=True,
            use_gpu=False,
            use_debug=False,
            cpu_threads=cpu_threads,
            rec_algorithm='SVTR_LCNet',
            lang='en',
            **secondary_options,
        )
        return _ocr_primary, _ocr_secondary
    except Exception as e:
        print(f"[chem_ocr_expand] PaddleOCR init error: {e}")
        return None, None


def _easyocr_recognize(reader, image_np):
    """Run EasyOCR on a cropped numpy image and return (text, score)."""
    if image_np is None or image_np.size == 0:
        return "", 0.0
    if image_np.shape[0] < 2 or image_np.shape[1] < 2:
        return "", 0.0
    try:
        results = reader.readtext(image_np)
        if not results:
            return "", 0.0
        # Pick highest confidence result
        best_text, best_score = "", 0.0
        for (bbox, text, conf) in results:
            if float(conf) > best_score:
                best_text = str(text)
                best_score = float(conf)
        return best_text, best_score
    except Exception as e:
        print(f"[chem_ocr_expand] EasyOCR error: {e}")
        return "", 0.0


def _extract_paddleocr_2x_text_score(result):
    """Parse PaddleOCR 2.x det=False output into (text, score)."""
    if not result:
        return "", 0.0
    try:
        first = result[0]
    except Exception:
        return "", 0.0
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], str):
        try:
            return str(first[0]), float(first[1])
        except (TypeError, ValueError):
            return str(first[0]), 0.0
    if isinstance(first, (list, tuple)) and first:
        first = first[0]
    if isinstance(first, (list, tuple)):
        text = first[0] if len(first) > 0 else ""
        score = first[1] if len(first) > 1 else 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return str(text), score
    if isinstance(first, str):
        return first, 0.0
    return "", 0.0


def _paddleocr_recognize(ocr, image_np):
    """Run PaddleOCR on a cropped numpy image and return (text, score).

    PaddleOCR 2.x supports the i2m path: ``ocr.ocr(image_np, det=False)``.
    PaddleOCR 3.x redirects ``ocr()`` to ``predict()`` and no longer accepts
    ``det=False``; keep a best-effort 3.x parser so older cached environments do
    not fail immediately, but the supported rtv4 OCR stack is 2.x.
    """
    if image_np is None or image_np.size == 0:
        return "", 0.0
    if image_np.shape[0] < 2 or image_np.shape[1] < 2:
        return "", 0.0
    try:
        try:
            return _extract_paddleocr_2x_text_score(ocr.ocr(image_np, det=False))
        except TypeError:
            pass

        result_list = ocr.predict(image_np)
        if not result_list:
            return "", 0.0
        res = result_list[0]
        rec_texts = res.get('rec_texts', []) if hasattr(res, 'get') else getattr(res, 'rec_texts', [])
        rec_scores = res.get('rec_scores', []) if hasattr(res, 'get') else getattr(res, 'rec_scores', [])
        if not rec_texts:
            return "", 0.0
        best_idx = 0
        best_score = 0.0
        for i, (txt, sc) in enumerate(zip(rec_texts, rec_scores)):
            try:
                sc = float(sc)
            except (TypeError, ValueError):
                sc = 0.0
            if sc > best_score:
                best_score = sc
                best_idx = i
        return str(rec_texts[best_idx]), best_score
    except Exception as e:
        print(f"[chem_ocr_expand] PaddleOCR error: {e}")
        return "", 0.0


def select_chem_expression(orig_text, orig_score, scaled_text, scaled_score):
    """Select the more reasonable chemical expression between two OCR results.

    Adapted from i2m det_engine.py select_chem_expression().
    """
    try:
        orig_score = float(orig_score)
    except (TypeError, ValueError):
        orig_score = 0.0
    try:
        scaled_score = float(scaled_score)
    except (TypeError, ValueError):
        scaled_score = 0.0

    # If one is in ABBREVIATIONS and contains the other, prefer it
    if scaled_text and orig_text in ABBREVIATIONS and scaled_text in orig_text:
        return orig_text, orig_score
    if orig_text and scaled_text in ABBREVIATIONS and orig_text in scaled_text:
        return scaled_text, scaled_score

    orig_valid = is_valid_chem_text(orig_text)
    scaled_valid = is_valid_chem_text(scaled_text)
    score_diff = abs(orig_score - scaled_score)

    if score_diff > 0.1:
        if orig_valid and scaled_valid:
            if orig_score >= scaled_score and orig_text:
                return orig_text, orig_score
            elif scaled_text:
                return scaled_text, scaled_score
        elif orig_valid:
            return orig_text, orig_score
        elif scaled_valid:
            return scaled_text, scaled_score
        else:
            return (orig_text, orig_score) if orig_score >= scaled_score else (scaled_text, scaled_score)
    else:
        if orig_valid and not scaled_valid:
            return orig_text, orig_score
        elif scaled_valid and not orig_valid:
            return scaled_text, scaled_score
        elif orig_valid and scaled_valid:
            if orig_text in ABBREVIATIONS and scaled_text not in ABBREVIATIONS:
                return orig_text, orig_score
            elif scaled_text in ABBREVIATIONS and orig_text not in ABBREVIATIONS:
                return scaled_text, scaled_score
            # Both valid and similar score: prefer longer text (more complete)
            if len(orig_text) >= len(scaled_text):
                return orig_text, orig_score
            else:
                return scaled_text, scaled_score
        else:
            return (orig_text, orig_score) if len(orig_text) >= len(scaled_text) else (scaled_text, scaled_score)


# ---------------------------------------------------------------------------
# Main: OCR recognition on "other" atom boxes + graph expansion
# ---------------------------------------------------------------------------

def recognize_rgroup_labels(graph, image, ocr_reader=None, margin=2, use_gpu=True, backend="auto"):
    """Run OCR on atom boxes with label_name='other' to identify R-groups.

    Uses PaddleOCR as the default backend when available. Dual-scale
    recognition uses original + 2x scaled crops, same as i2m_eval.py.

    Args:
        graph: output of assemble_molecular_graph()
        image: PIL.Image of the original chemical structure
        ocr_reader: EasyOCR Reader instance (lazy init if None)
        margin: pixel margin to expand the crop box
        use_gpu: whether to use GPU for EasyOCR
        backend: one of {"auto","easyocr","paddleocr"}.
            - "auto": prefer PaddleOCR; if unavailable, fall back to EasyOCR
            - "easyocr": force EasyOCR only (no fallback)
            - "paddleocr": force PaddleOCR only (no fallback)

    Returns:
        dict mapping atom_index -> {'text': str, 'score': float, 'smiles': str}
    """
    if image is None:
        return {}

    img_w, img_h = image.size
    # Prepare scaled image (2x)
    scale_x, scale_y = 2.0, 2.0
    img_scaled = image.resize((int(img_w * scale_x), int(img_h * scale_y)), Image.LANCZOS)

    backend = str(backend or "auto").strip().lower()
    if backend not in ("auto", "easyocr", "paddleocr"):
        backend = "auto"

    # Initialize OCR backend
    _recognize = None
    if backend in ("auto", "paddleocr"):
        ocr_primary, _ = _get_paddleocr_instances(use_gpu=use_gpu)
        if ocr_primary is not None:
            _recognize = lambda img_np: _paddleocr_recognize(ocr_primary, img_np)
        elif backend == "paddleocr":
            return {}

    if _recognize is None and backend in ("auto", "easyocr"):
        if ocr_reader is None:
            ocr_reader = _get_easyocr_reader(gpu=use_gpu)
        if ocr_reader is not None:
            _recognize = lambda img_np: _easyocr_recognize(ocr_reader, img_np)
        elif backend == "easyocr":
            # Forced EasyOCR but unavailable.
            return {}

    if _recognize is None:
        print("[chem_ocr_expand] No OCR backend available (need paddleocr or easyocr)")
        return {}

    ocr_labels = {}

    for atom in graph['atoms']:
        if atom['label_name'] != 'other':
            continue

        # Skip rgroup category atoms that already have known labels
        symbol = atom.get('symbol', '*')
        if symbol != '*':
            continue

        box = atom['box']
        atom_idx = atom['index']

        # Original scale crop
        x1 = max(0, box[0] - margin)
        y1 = max(0, box[1] - margin)
        x2 = min(img_w, box[2] + margin)
        y2 = min(img_h, box[3] + margin)
        crop_orig = np.array(image.crop((x1, y1, x2, y2)))

        # Scaled crop
        sx1 = max(0, (box[0] - margin) * scale_x)
        sy1 = max(0, (box[1] - margin) * scale_y)
        sx2 = min(int(img_w * scale_x), (box[2] + margin) * scale_x)
        sy2 = min(int(img_h * scale_y), (box[3] + margin) * scale_y)
        crop_scaled = np.array(img_scaled.crop((sx1, sy1, sx2, sy2)))

        # OCR at both scales (same reader for both, dual-scale strategy)
        orig_raw_text, orig_score = _recognize(crop_orig)
        scaled_raw_text, scaled_score = _recognize(crop_scaled)

        # Normalize
        orig_text = normalize_ocr_text(orig_raw_text, RGROUP_REPLACEMENT_MAP)
        scaled_text = normalize_ocr_text(scaled_raw_text, RGROUP_REPLACEMENT_MAP)

        # Special overrides (NO2, SO2 are commonly misread)
        if orig_text == 'NO2' or scaled_text == 'NO2':
            final_text = 'NO2'
            final_score = max(orig_score if orig_text == 'NO2' else 0,
                              scaled_score if scaled_text == 'NO2' else 0)
        elif orig_text == 'SO2' or scaled_text == 'SO2':
            final_text = 'SO2'
            final_score = max(orig_score if orig_text == 'SO2' else 0,
                              scaled_score if scaled_text == 'SO2' else 0)
        else:
            final_text, final_score = select_chem_expression(
                orig_text, orig_score, scaled_text, scaled_score
            )

        if final_text:
            smiles = expand_abbreviation(final_text)
            ocr_labels[atom_idx] = {
                'text': final_text,
                'score': float(final_score),
                'smiles': smiles,
                # Raw OCR outputs at both scales (before normalize_ocr_text).
                'orig_raw_text': str(orig_raw_text) if orig_raw_text is not None else "",
                'scaled_raw_text': str(scaled_raw_text) if scaled_raw_text is not None else "",
                'orig_text': orig_text,
                'orig_score': float(orig_score),
                'scaled_text': scaled_text,
                'scaled_score': float(scaled_score),
            }

    return ocr_labels


# ---------------------------------------------------------------------------
# RDKit-based molecular graph expansion
# ---------------------------------------------------------------------------

def expand_molecule_graph(graph, ocr_labels):
    """Expand the molecular graph using OCR-recognized labels.

    For each atom that got an OCR label, replace its '*' symbol with the
    recognized text, then build an RDKit molecule with abbreviations expanded.

    Args:
        graph: output of assemble_molecular_graph()
        ocr_labels: output of recognize_rgroup_labels()

    Returns:
        dict with keys:
          'skeleton_smiles': SMILES of the skeleton (with * placeholders)
          'expanded_smiles': SMILES with abbreviations expanded (if RDKit available)
          'ocr_atoms': list of atom info dicts
    """
    result = {
        'skeleton_smiles': None,
        'expanded_smiles': None,
        'ocr_atoms': [],
    }

    # Build skeleton SMILES with atom labels
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("[chem_ocr_expand] rdkit not available, skipping expansion")
        return result

    # Mark atoms that have OCR labels
    ocr_atom_map = {}  # atom_index -> ocr_labels[atom_index]
    for atom in graph['atoms']:
        if atom['index'] in ocr_labels:
            ocr_atom_map[atom['index']] = ocr_labels[atom['index']]

    # Build RDKit molecule with placeholder atoms
    rwmol = Chem.RWMol()
    atom_index_map = {}  # graph atom index -> rdkit atom index
    placeholder_atoms = {}  # rdkit atom index -> abbreviation text

    for atom in graph['atoms']:
        symbol = atom['symbol']
        charge = int(atom.get('formal_charge', 0))

        if atom['index'] in ocr_atom_map:
            label = ocr_atom_map[atom['index']]['text']
            smiles_exp = ocr_atom_map[atom['index']]['smiles']
            # Use wildcard atom as placeholder
            rd_atom = Chem.Atom("*")
            rd_idx = rwmol.AddAtom(rd_atom)
            atom_index_map[atom['index']] = rd_idx
            rwmol.GetAtomWithIdx(rd_idx).SetProp("atomLabel", label)
            placeholder_atoms[rd_idx] = label
        elif symbol == '*':
            rd_atom = Chem.Atom("*")
            rd_idx = rwmol.AddAtom(rd_atom)
            atom_index_map[atom['index']] = rd_idx
        else:
            # Standard atom
            try:
                rd_atom = Chem.Atom(symbol)
            except Exception:
                rd_atom = Chem.Atom("*")
                placeholder_atoms[rwmol.GetNumAtoms()] = symbol
            rd_atom.SetFormalCharge(charge)
            rd_idx = rwmol.AddAtom(rd_atom)
            atom_index_map[atom['index']] = rd_idx

    # Add bonds
    for bond in graph['bonds']:
        begin = atom_index_map.get(bond['begin_atom_idx'])
        end = atom_index_map.get(bond['end_atom_idx'])
        if begin is None or end is None:
            continue
        if begin == end:
            continue
        kind = bond['graph_bond']['kind']
        if kind == 'double':
            bt = Chem.BondType.DOUBLE
        elif kind == 'triple':
            bt = Chem.BondType.TRIPLE
        elif kind == 'aromatic':
            bt = Chem.BondType.AROMATIC
        else:
            bt = Chem.BondType.SINGLE
        if rwmol.GetBondBetweenAtoms(begin, end) is None:
            rwmol.AddBond(begin, end, bt)

    # Skeleton SMILES
    mol_skel = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol_skel)
        result['skeleton_smiles'] = Chem.MolToSmiles(mol_skel)
    except Exception as e:
        print(f"[chem_ocr_expand] SanitizeMol failed for skeleton: {e}")
        result['skeleton_smiles'] = Chem.MolToSmiles(rwmol)

    # Expand abbreviations
    if placeholder_atoms:
        expanded_mol = _expand_abbreviations(rwmol, placeholder_atoms)
        if expanded_mol is not None:
            try:
                AllChem.Compute2DCoords(expanded_mol)
                Chem.DetectBondStereochemistry(expanded_mol)
                Chem.AssignChiralTypesFromBondDirs(expanded_mol)
                Chem.AssignStereochemistry(expanded_mol)
                result['expanded_smiles'] = Chem.MolToSmiles(expanded_mol)
            except Exception as e:
                print(f"[chem_ocr_expand] expansion stereochemistry failed: {e}")
                try:
                    result['expanded_smiles'] = Chem.MolToSmiles(expanded_mol)
                except Exception:
                    pass

    # Collect OCR atom info
    for atom in graph['atoms']:
        if atom['index'] in ocr_labels:
            info = dict(ocr_labels[atom['index']])
            info['atom_index'] = atom['index']
            info['box'] = atom['box']
            result['ocr_atoms'].append(info)

    return result


def _expand_abbreviations(rwmol, placeholder_atoms):
    """Expand placeholder atoms in an RDKit RWMol using ABBREVIATIONS.

    Adapted from i2m det_engine.py expandABB().
    Processes atoms in reverse index order to preserve indices.
    """
    from rdkit import Chem

    mol = Chem.RWMol(rwmol)

    for idx in sorted(placeholder_atoms.keys(), reverse=True):
        group = placeholder_atoms[idx]
        group_smiles = expand_abbreviation(group)

        try:
            submol = Chem.MolFromSmiles(group_smiles)
            if submol is None:
                submol = Chem.MolFromSmiles('[*]')
            submol_rw = Chem.RWMol(submol)
        except Exception as e:
            print(f"[chem_ocr_expand] Failed to parse SMILES '{group_smiles}' for '{group}': {e}")
            submol = Chem.MolFromSmiles('[*]')
            submol_rw = Chem.RWMol(submol)

        # Find anchor atoms in submol (radical electrons or first atom)
        anchor_atoms = [0]
        for atom in submol_rw.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0 and atom.GetIdx() not in anchor_atoms:
                anchor_atoms.append(atom.GetIdx())

        # Copy main molecule
        new_mol = Chem.RWMol(mol)
        placeholder_idx = idx

        # Record bonds to the placeholder atom
        bonds_info = []
        for bond in new_mol.GetBonds():
            if bond.GetBeginAtomIdx() == placeholder_idx:
                bonds_info.append({
                    "neighbor": bond.GetEndAtomIdx(),
                    "bond_type": bond.GetBondType()
                })
            elif bond.GetEndAtomIdx() == placeholder_idx:
                bonds_info.append({
                    "neighbor": bond.GetBeginAtomIdx(),
                    "bond_type": bond.GetBondType()
                })

        # Remove placeholder's bonds
        for bond_info in bonds_info:
            new_mol.RemoveBond(placeholder_idx, bond_info["neighbor"])

        # Remove placeholder atom
        new_mol.RemoveAtom(placeholder_idx)

        # Adjust neighbor indices (after atom removal)
        adjusted_bonds = []
        for bond_info in bonds_info:
            neighbor = bond_info["neighbor"]
            adjusted = neighbor if neighbor < placeholder_idx else neighbor - 1
            adjusted_bonds.append({
                "neighbor": adjusted,
                "bond_type": bond_info["bond_type"]
            })

        # Merge submol
        new_mol = Chem.RWMol(Chem.CombineMols(new_mol, submol_rw))

        # Compute anchor indices in merged mol
        submol_offset = new_mol.GetNumAtoms() - submol_rw.GetNumAtoms()
        new_anchors = [submol_offset + a for a in anchor_atoms]

        # Reconnect bonds
        if len(new_anchors) == 1:
            anchor_idx = new_anchors[0]
            for bond_info in adjusted_bonds:
                neighbor = bond_info["neighbor"]
                bond_type = bond_info["bond_type"]
                if new_mol.GetBondBetweenAtoms(neighbor, anchor_idx) is None:
                    new_mol.AddBond(neighbor, anchor_idx, bond_type)
                a1 = new_mol.GetAtomWithIdx(neighbor)
                a2 = new_mol.GetAtomWithIdx(anchor_idx)
                a1.SetNumRadicalElectrons(0)
                a2.SetNumRadicalElectrons(0)
        elif len(new_anchors) > 1:
            # Multi-anchor: connect first bond to first anchor, rest to others
            for i, bond_info in enumerate(adjusted_bonds):
                anchor_idx = new_anchors[i] if i < len(new_anchors) else new_anchors[-1]
                neighbor = bond_info["neighbor"]
                bond_type = bond_info["bond_type"]
                if new_mol.GetBondBetweenAtoms(neighbor, anchor_idx) is None:
                    new_mol.AddBond(neighbor, anchor_idx, bond_type)
                new_mol.GetAtomWithIdx(neighbor).SetNumRadicalElectrons(0)
                new_mol.GetAtomWithIdx(anchor_idx).SetNumRadicalElectrons(0)

        mol = new_mol

    return mol.GetMol()


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def ocr_expand_graph(graph, image, use_gpu=True):
    """Run OCR recognition + graph expansion in one call.

    Args:
        graph: output of assemble_molecular_graph()
        image: PIL.Image of the original chemical structure
        use_gpu: whether to use GPU for OCR

    Returns:
        dict with keys from expand_molecule_graph() plus 'ocr_labels'
    """
    ocr_labels = recognize_rgroup_labels(graph, image, use_gpu=use_gpu)
    expansion = expand_molecule_graph(graph, ocr_labels)
    expansion['ocr_labels'] = ocr_labels
    return expansion
