from rdkit import Chem
import numpy as np
import sys
import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'

sys.path.insert(0, root_path+'/Run_reaction/classes')
import reaction_class_auto
sys.path.insert(0, root_path+'/Descriptors/classes')
from Descriptor_classes import *
sys.path.insert(0, root_path+'/01_code/DINGOS_database/classes')
import Make_db
from Make_db import *

class Reaction_functions:
    def __init__(self,rxn_db):
        self.Database_functions = Database_functions()
        self.rxn_db = rxn_db

        # The temporary databases are initially set to be empty
        self.temporary_product_db = []
        self.temporary_mol_rxn_junction_db = []

    def Perform_1_component_reaction(self,mol_input,rxn_input,*args, **kwargs):
        start_mol_id = int(mol_input[0])
        start_mol_smiles = mol_input[1]
        start_mol = Chem.MolFromSmiles(start_mol_smiles)

        rxn_id = int(rxn_input[0])
        rxn_name = rxn_input[1]
        rxn_SMARTS = rxn_input[2]
        rxn_label = rxn_input[3]

        # Sets the reaction class
        reaction = reaction_class_auto.reaction(rxn_name,rxn_SMARTS,rxn_label)

        # Performs the reaction
        product_list = reaction.PerformReaction([start_mol])
        product_list = [prod_mol for prod_mol in product_list if prod_mol != None]
        [Chem.SanitizeMol(prod_mol) for prod_mol in product_list]
        product_smiles = [Chem.MolToSmiles(mol) for mol in product_list]

        # Sets the product ids
        product_output = []
        for index, product_smile_input in enumerate(product_smiles):
            product_id = len(self.temporary_mol_rxn_junction_db)
            product_output.append([product_id,product_list[index]])

        # Inserts the product values into the temporary database
        self.create_temp_junction_table(product_output=product_output)

        return product_output

    def Perform_2_component_reaction(self,mol_input,building_block_input,rxn_input,*args, **kwargs):
        start_mol = mol_input[0]
        start_mol_id = int(mol_input[1])

        building_block_id = int(building_block_input[0])
        building_block_smiles = building_block_input[1]
        building_block_mol = Chem.MolFromSmiles(building_block_smiles)

        rxn_id = int(rxn_input[0])
        rxn_name = rxn_input[1]
        rxn_SMARTS = rxn_input[2]
        rxn_label = rxn_input[3]

        # Sets the reaction class
        reaction = reaction_class_auto.reaction(rxn_name,rxn_SMARTS,rxn_label)

        # Performs the reaction
        product_list = reaction.PerformReaction([start_mol,building_block_mol])
        product_list = [prod_mol for prod_mol in product_list if prod_mol != None]
        [Chem.SanitizeMol(prod_mol) for prod_mol in product_list]
        product_smiles = [Chem.MolToSmiles(mol) for mol in product_list]

        # Sets the product ids
        product_output = []
        for index, product_smile_input in enumerate(product_smiles):
            product_id = len(self.temporary_mol_rxn_junction_db)
            product_output.append([product_id, product_list[index]])

        # Inserts the product values into the temporary database
        self.create_temp_junction_table(product_output=product_output)
        return np.array(product_output)

    def create_temp_junction_table(self, *args, **kwargs):
        product_output = kwargs.get('product_output', None)
        product_mol_rxn_junciton = self.Database_functions.mol_rxn_check(product_output, self.rxn_db)
        if len(self.temporary_mol_rxn_junction_db) == 0:
            self.temporary_mol_rxn_junction_db = product_mol_rxn_junciton
        # Accounts for end point molecules that can not take part in any other reactions
        elif len(product_mol_rxn_junciton) != 0:
            self.temporary_mol_rxn_junction_db = np.vstack(
                (self.temporary_mol_rxn_junction_db, product_mol_rxn_junciton))

    def reinitalize_temporary_databases(self):
        self.temporary_product_db = []
        self.temporary_mol_rxn_junction_db = []