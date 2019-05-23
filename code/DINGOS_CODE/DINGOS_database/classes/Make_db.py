import numpy as np
from rdkit import Chem
import sys
import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'
sys.path.insert(0, root_path+'/Perform_reactions/classes')
import reaction_class_auto

class Database_functions:
    def mol_rxn_check(self, mol_set, rxn_set):
        mol_rxn_junction_data = []
        for mol_entry in mol_set:
            mol_id = int(mol_entry[0])
            # Accepts both rdkit mol objects and smiles strings
            try:
                mol_input = Chem.MolFromSmiles(mol_entry[1])
            except:
                mol_input = mol_entry[1]
            for rxn_entry in rxn_set:
                rxn = reaction_class_auto.reaction(rxn_entry[1], rxn_entry[2], rxn_entry[3])
                # Calculates and stores all potential reactant positions
                Reactant_positions = rxn.Reactant_position(mol_input)
                for position in Reactant_positions:
                    mol_rxn_junction_data.append([mol_id, rxn_entry[0], position])

        return np.asarray(mol_rxn_junction_data, dtype=int)