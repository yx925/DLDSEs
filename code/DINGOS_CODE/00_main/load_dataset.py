import sys
import numpy as np
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'
sys.path.insert(0, root_path + '/DINGOS_database/classes')
from Make_db import Database_functions

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

class Generate_databases:
    def read_in_molecule_set(self,mol_filename):
        mol_file = open(mol_filename,"r")
        header = mol_file.readline()

        mol_db = []
        mol_id = 1
        for line in mol_file:
            SMILES_string = line.strip("\n")
            try:
                mol = MolFromSmiles(SMILES_string)
                SMILES_canonical = MolToSmiles(mol)
                MW = ExactMolWt(mol)
                mol_id += 1
                mol_db.append([mol_id,SMILES_canonical,MW])
            except:
                print("failed to load",SMILES_string)

        return np.asarray(mol_db)

    def read_in_reaction_set(self,reaction_filename):
        rxn_file = open(reaction_filename, "r")
        header = rxn_file.readline()

        rxn_db = []
        rxn_id = 0
        for line in rxn_file:
            rxn_entries = line.strip("\n").split("|")
            rxn_id += 1
            rxn_db.append([rxn_id]+rxn_entries)

        return np.asarray(rxn_db)

    def create_descriptor_set(self,mol_set,descriptor_function):
        mol_id = mol_set[:,0]
        SMILES_set = mol_set[:,1]

        descriptor_db = []
        for index,SMILES_string in enumerate(SMILES_set):
            try:
                desc_value = descriptor_function(SMILES_string)
                ave_ones = np.sum(np.asarray(desc_value,dtype=float))/len(desc_value)
                desc_id = mol_id[index]
                descriptor_db.append([desc_id]+desc_value+[ave_ones])
            except:
                print("failed for",SMILES_string)

        return np.asarray(descriptor_db)

    def create_mol_rxn_junction(self,mol_set,rxn_set):
        DF = Database_functions()
        return DF.mol_rxn_check(mol_set[:,:2], rxn_set)

from rdkit.Chem import MACCSkeys
def descriptor_function(input_SMILES):
    input_mol = MolFromSmiles(input_SMILES)
    return list(MACCSkeys.GenMACCSKeys(input_mol).ToBitString())

mol_filename = "../../../data/Datasets/TEST_SET/mol_set.txt"
reaction_filename = "../../../data/Datasets/TEST_SET/rxn_set.txt"

from rdkit.Chem import MACCSkeys
def descriptor_function(input_SMILES):
    input_mol = MolFromSmiles(input_SMILES)
    return list(MACCSkeys.GenMACCSKeys(input_mol).ToBitString())

GD = Generate_databases()
custom_name = str(sys.argv[1])
database_dir = str(sys.argv[2])

mol_set = GD.read_in_molecule_set(mol_filename)
np.save(database_dir+custom_name+"-MOL_DB_DATA.npy",mol_set)

rxn_set = GD.read_in_reaction_set(reaction_filename)
np.save(database_dir+custom_name+"-RXN_DB_DATA.npy",rxn_set)

mol_rxn_junction = GD.create_mol_rxn_junction(mol_set,rxn_set)
np.save(database_dir+custom_name+"-mol_rxn_DB_DATA.npy",mol_rxn_junction)

desc_db = GD.create_descriptor_set(mol_set,descriptor_function)
np.save(database_dir+custom_name+"-DESCRIPTOR_DB_DATA_MACCSkey.npy",desc_db)