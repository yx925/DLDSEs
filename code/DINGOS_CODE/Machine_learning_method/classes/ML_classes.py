from rdkit import Chem
import sys

import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'
sys.path.insert(0,root_path+'/Distance_metric/classes')
from bit_dist import bit_comparison_count
sys.path.insert(0, root_path+'/Perform_reactions/classes')
import reaction_class_auto
sys.path.insert(0, root_path+'/Descriptors/classes')
from Descriptor_classes import *
sys.path.insert(0, root_path+'/DINGOS_database/classes')
from Make_db import *
sys.path.insert(0,root_path+'/Perform_reactions/classes')
from perform_reaction_class import Reaction_functions

import numpy as np
import sys

class Make_prediction:
    def __init__(self,molecule_db,mol_rxn_junction_db,descriptor_db,rxn_db,descriptor_function):
        self.molecule_db = molecule_db
        self.mol_rxn_junction_db = mol_rxn_junction_db
        self.descriptor_db = descriptor_db
        self.rxn_db = rxn_db
        self.descriptor_function = descriptor_function
        test_descriptor, self.descriptor_tag = self.descriptor_function(Chem.MolFromSmiles('CCC'))
        self.descriptor_dim = len(test_descriptor)

        self.viable_reaction_with_reactant_position = None
        self.RF = Reaction_functions(self.rxn_db)

    def produce_viable_list(self,*args,**kwargs):
        start_input = kwargs.get('start_input', None)
        mol_rxn_database = kwargs.get('mol_rxn_database', self.mol_rxn_junction_db)

        start_mol, start_id = start_input
        start_mol_rxns = mol_rxn_database[np.where(mol_rxn_database[:,0]==int(start_id))]
        
        # Converts the entires of the starting molecule mol-rxn junction table into query strings
        start_reaction_id_list = np.array([start_mol_rxns[:,1]],dtype='str')
        start_reactant_inverted_position = np.array([1-start_mol_rxns[:,2]],dtype='str')
        reaction_search_list = np.core.defchararray.add(start_reaction_id_list, "|")
        reaction_search_list = np.core.defchararray.add(reaction_search_list, start_reactant_inverted_position)

        # Converts the entires of the building block mol-rxn junction table into query strings
        mol_rxn_id_list = np.array([self.mol_rxn_junction_db[:,1]],dtype='str')
        mol_rxn_reactant_position = np.array(self.mol_rxn_junction_db[:,2],dtype='str')
        mol_rxn_search_list = np.core.defchararray.add(mol_rxn_id_list, "|")
        mol_rxn_search_list = np.core.defchararray.add(mol_rxn_search_list, mol_rxn_reactant_position)

        # Finds the intersecting queries between the starting molecule and building block mol-rxn junction tables
        reaction_list_query_match = np.in1d(mol_rxn_search_list,reaction_search_list)
        viable_mol_rxn_entries = self.mol_rxn_junction_db[reaction_list_query_match]

        # Locates the corresponding molecule entries in the mol set
        viable_molecules = self.molecule_db[np.in1d(self.molecule_db[:,0], viable_mol_rxn_entries[:,0])]
        viable_reaction_with_reactant_position = viable_mol_rxn_entries
        
        # Saves the descriptor and mol ids 
        desc_input = np.asarray(self.descriptor_db[:,0],dtype=int)
        vi_mol_input = np.asarray(viable_molecules[:,0],dtype=int)
        
        # Locates the corresponding descriptor entries in the descriptor set
        viable_mol_descriptor_values = np.asarray(self.descriptor_db[np.in1d(desc_input,vi_mol_input)],dtype=int)

        # Adds the NULL entry to molecule db
        viable_molecules = np.vstack(([1,'Null'],viable_molecules))

        # Adds the NULL to descriptor db
        null_descriptor_value = np.zeros((1, self.descriptor_dim),dtype=int)
        null_entry = np.append(1,null_descriptor_value)
        viable_mol_descriptor_values = np.vstack((null_entry,viable_mol_descriptor_values))

        self.viable_reaction_with_reactant_position = viable_reaction_with_reactant_position

        return viable_molecules,viable_mol_descriptor_values

    def Predict_building_block_from_product_mol(self,*args,**kwargs):
        start_input = kwargs.get('start_input', None)
        product = kwargs.get('product', None)
        ML_method = kwargs.get('ML_method', None)
        round_threshold = kwargs.get('round_threshold', 0.5)
        substructure_list = kwargs.get('substructure_list', 'None')
        precalculated_product_fp = kwargs.get('product_fp', 'None')
        mol_rxn_database = kwargs.get('mol_rxn_database', self.mol_rxn_junction_db)

        # Loads the viable molecule and descriptor sets
        viable_molecules, viable_mol_descriptor_values = self.produce_viable_list(start_input=start_input,substructure_list=substructure_list,mol_rxn_database=mol_rxn_database)
        library_matrix = viable_mol_descriptor_values[:,1:]

        # Calculates the starting molecule fingerprint
        start_mol_fp = self.descriptor_function(start_input[0])[0]
        if precalculated_product_fp == 'None':
            # Calculates the template fingerprint
            product_fp = np.asarray(self.descriptor_function(product)[0],dtype=int)
        else:
            # Sets the precalculated template fingerprint
            product_fp = np.asarray(precalculated_product_fp,dtype=int)
        
        # Creates the input for the ML model
        query = np.concatenate([start_mol_fp, product_fp], axis=0)

        # Predicts the building block fingerprint
        predicted_output = ML_method(sample=np.asarray([query]))[0]
        predicted_output = np.asarray([int(p > round_threshold) for p in predicted_output])

        # Calculates the distance values of the viable molecule set to the predicted fingerprint 
        BC = bit_comparison_count(predicted_output,library_matrix)
        BC.hamming_loss()
        distance_list = BC.hamming_loss_vector
        
        # Sorts the distance values (lowest to highest)
        sorted_index = np.argsort(distance_list)

        # Sorts the viable molecule set according to the sorted distance values
        prediction_sorted_bb_id = viable_molecules[:,0][sorted_index]
        prediction_sorted_SMILES = viable_molecules[:,1][sorted_index]
        prediction_sorted_distances= distance_list[sorted_index]

        return prediction_sorted_bb_id,prediction_sorted_SMILES,prediction_sorted_distances

    def Query_molecule(self,*args,**kwargs):
        start_mol = kwargs.get('start_mol', None)
        predicted_building_blocks = kwargs.get('predicted_building_blocks', None)
        reaction_step = kwargs.get('reaction_step', None)
        start_fragment_limit = kwargs.get('start_fragment_limit', 1)
        flagged_reactions = kwargs.get('flagged_reactions', 'None')

        # If there are no possible reactions for the input mol
        if len(predicted_building_blocks) != 0:
            sorted_building_block_id = predicted_building_blocks[0][:start_fragment_limit]
            sorted_building_block_SMILES = predicted_building_blocks[1][:start_fragment_limit]
            sorted_building_block_list = np.transpose(np.array([sorted_building_block_id,sorted_building_block_SMILES]))
        else:
            return []

        total_product_output = []
        total_product_id_output = []
        total_rxns = []

        # Filters the reactions based on the flagged_reactions list
        filtered_index = np.in1d(self.viable_reaction_with_reactant_position[:,1],flagged_reactions,invert=True)
        filtered_rxn = self.viable_reaction_with_reactant_position[filtered_index]

        for suggested_bb in sorted_building_block_list:
            # Retieves the shared reactions between the start molecule and the suggested building block
            possible_reaction = filtered_rxn[np.where(filtered_rxn[:,0] == int(suggested_bb[0]))]
            for rxn_id in possible_reaction[:,1]:
                reaction_entry = self.rxn_db[np.where(np.asarray(self.rxn_db[:,0],dtype=int) == rxn_id)][0]
                if suggested_bb[0] == 1:
                    product_output = self.RF.Perform_1_component_reaction(start_mol,reaction_entry,reaction_step=reaction_step)
                elif suggested_bb[0] != 1:
                    product_output = self.RF.Perform_2_component_reaction(start_mol,suggested_bb,reaction_entry,reaction_step=reaction_step)

                if len(product_output) != 0:
                    total_product_id_output.append(product_output[:,0])
                    total_product_output.append(product_output[:,1])
                    # Saves the reaction pathway of the product
                    for i in range(0,len(product_output)):
                        total_rxns.append([suggested_bb[1]+">"+"["+reaction_entry[1].replace(" ","_")+"]"+">"])

        if len(total_product_output) != 0:
            total_product_id_output = np.concatenate(total_product_id_output)
            total_product_output = np.concatenate(total_product_output)

        return total_product_id_output,total_product_output,total_rxns
