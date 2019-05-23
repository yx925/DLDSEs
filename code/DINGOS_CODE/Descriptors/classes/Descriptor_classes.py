from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sys
import os

class Descriptors:
    def mol_MACCSkey(self,input_mol):
        descriptor_name = 'MACCSkey'
        descriptor_value = np.asarray(list(MACCSkeys.GenMACCSKeys(input_mol).ToBitString()), dtype=int)
        return (descriptor_value,descriptor_name)

    def mol_rdkit_FP(self,input_mol,**kwargs):
        minPath = kwargs.get('minPath', 1)
        maxPath = kwargs.get('maxPath', 7)
        descriptor_name = 'RDKIT_FP'
        descriptor_value = np.asarray(list(Chem.RDKFingerprint(input_mol,minPath=minPath,maxPath=maxPath).ToBitString()), dtype=int)
        return (descriptor_value, descriptor_name)

    def mol_morgan_FP(self,input_mol,**kwargs):
        radius = kwargs.get('radius', 3)
        descriptor_name = 'MORGAN_FP'
        descriptor_value = np.asarray([bit for bit in Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(input_mol,radius)], dtype=int)
        return (descriptor_value, descriptor_name)