import numpy as np
from rdkit import Chem
import sys
import os

root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'


sys.path.insert(0,root_path+'/Distance_metric/classes')
from bit_dist import bit_comparison_count
sys.path.insert(0,root_path+'/Machine_learning_method/classes')
from ML_classes import *
sys.path.insert(0,root_path+'/01_code/Descriptors/classes')
from Descriptor_classes import *
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DINGOS_run:
    def __init__(self,target_mol,bb_prediction,assembly_method,molecular_descriptor,distance_metric,start_mol_ML,ML_class):
        self.target_mol = target_mol
        self.bb_prediction = bb_prediction
        self.assembly_method = assembly_method
        self.molecular_descriptor = molecular_descriptor
        self.distance_metric = distance_metric
        self.start_mol_ML = start_mol_ML
        self.ML_class = ML_class
        
        self.set_template_fp()

    def select_start_molecule(self, start_library):
        mol_library, descriptor_library = start_library
        library_matrix = np.asarray(descriptor_library[:, 1:], dtype=int)

        # Calculates distance values of the building block set to the template 
        BC = bit_comparison_count(self.target_fp, library_matrix)
        BC.hamming_loss()
        distance_value = BC.hamming_loss_vector

        # Sorts the distance values (lowest to highest)
        sorted_index = np.argsort(distance_value)

        # Sorts the building block set according to the sorted distance values
        start_mol_dist = distance_value[sorted_index]
        start_mol_index_sorted = mol_library[:, 0][sorted_index]
        start_mol_smiles_sorted = mol_library[:, 1][sorted_index]

        return start_mol_smiles_sorted, start_mol_index_sorted, start_mol_dist

    def select_start_molecule_ML(self, start_library):
        mol_library, descriptor_library = start_library
        library_matrix = np.asarray(descriptor_library[:, 1:], dtype=int)

        # Creates the input for the ML model
        start_mol_query = np.concatenate([np.zeros(self.target_fp.shape), self.target_fp], axis=0)
        # Predicts the building block fingerprint
        start_mol_prediction = self.start_mol_ML.predict(np.asarray([start_mol_query]))[0]
        start_mol_prediction = np.asarray([int(p > 0.5) for p in start_mol_prediction])

        # Calculates distance values of the building block set to the fingerprint 
        BC = bit_comparison_count(start_mol_prediction, library_matrix)
        BC.hamming_loss()
        distance_value = BC.hamming_loss_vector

        # Sorts the distance values (lowest to highest)
        sorted_index = np.argsort(distance_value)

        # Sorts the building block set according to the sorted distance values
        start_mol_dist = distance_value[sorted_index]
        start_mol_index_sorted = mol_library[:, 0][sorted_index]
        start_mol_smiles_sorted = mol_library[:, 1][sorted_index]

        return start_mol_smiles_sorted, start_mol_index_sorted, start_mol_dist

    def set_template_fp(self):
        # Calculates the template fingerprint
        self.target_fp = np.asarray(self.molecular_descriptor(self.target_mol)[0],dtype=int)
        # Sets the template fingerprint in the predictive method
        self.bb_prediction = functools.partial(self.bb_prediction, product=self.target_mol)

    def single_assembly_step(self,input_mol,input_distance,reaction_step):
        output_molecules = None
        output_distance = None
        output_id = None

        # Checks if the temporary mol-rxn junction table is empty 
        if len(self.ML_class.RF.temporary_mol_rxn_junction_db) == 0:
            # Uses the mol-rxn numpy array loaded in the DINGOS.py script
            mol_rxn_db = self.ML_class.mol_rxn_junction_db
        else:
            mol_rxn_db = self.ML_class.RF.temporary_mol_rxn_junction_db
            # Wipes the temporary mol-rxn junction table clean
            self.ML_class.RF.reinitalize_temporary_databases()

        # Predicts building block pairs 
        bb_prediction = self.bb_prediction(start_input=input_mol,mol_rxn_database=mol_rxn_db)
        new_products_id,new_products,rxns_used = self.assembly_method(start_mol=input_mol, predicted_building_blocks=bb_prediction, distance_metric=self.distance_metric,reaction_step=reaction_step)

        # Logs the single step reacitons
        rxns_used = np.asarray(rxns_used)

        if len(new_products) != 0:
            # Calculates the distance values of the intermediate products 
            new_products_desc_matrix = np.asarray([self.molecular_descriptor(mol)[0] for mol in new_products],dtype=int)
            BC = bit_comparison_count(self.target_fp, new_products_desc_matrix)
            BC.hamming_loss()
            distance_calc = BC.hamming_loss_vector
            # Sorts the products according to their distance to the template
            sorted_distances_index = np.argsort(distance_calc)
            
            # Only saves the product if it has a shorter distance than the input molecule
            if distance_calc[sorted_distances_index][0] < input_distance:
                output_molecules = new_products[sorted_distances_index][0]
                output_id = new_products_id[sorted_distances_index][0]
                output_distance = distance_calc[sorted_distances_index][0]
                rxns_used = rxns_used[sorted_distances_index][0]
        return output_molecules, output_id, output_distance,rxns_used

    def full_assembly(self,*args,**kwargs):
        start_mol_library = kwargs.get('start_mol_library', None)
        start_molecule_set = kwargs.get('start_molecule_set', 'None')
        reaction_length_limit = kwargs.get('reaction_length_limit', 4)
        product_limit = kwargs.get('product_limit', 100)
        mass_threshold = kwargs.get('mass_threshold', float("inf"))

        # Checks if a custom start molecule set has been provided
        if 'None' in start_molecule_set:
            # Selects the start molecules based on their distance to the template
            if self.start_mol_ML == 'Null':
                mol_data = np.transpose(self.select_start_molecule(start_mol_library))
            # Selects the start molecules based on their distance to the predicted building block fingerprint
            elif self.start_mol_ML != 'Null':
                mol_data = np.transpose(self.select_start_molecule_ML(start_mol_library))
        else:
            # Sets the starting molecules based on the custom start molecule set
            mol_set,mol_id_set = start_molecule_set
            # Calculates the distance of the custom starting molecules to the template
            mol_dist_set = [self.distance_metric(self.molecular_descriptor(Chem.MolFromSmiles(mol))[0],self.target_fp) for mol in start_molecule_set]
            # Sorts the distance values
            sorted_index_list = np.argsort(mol_dist_set)
            mol_data = np.transpose(np.asarray([mol_set[sorted_index_list],mol_id_set[sorted_index_list],mol_dist_set[sorted_index_list]]))

        # Set the design cycle function
        design_cycle_func = functools.partial(self.design_cycle,reaction_step_limit=reaction_length_limit,threshold=mass_threshold)

        DINGOS_results = []
        index = 0
        for mol_entry in mol_data[:product_limit]:
            index = index + 1
            print "Design number ",index
            print "Start_mol :", mol_entry[0]

            DINGOS_design,DINGOS_pathway = design_cycle_func(mol_entry=mol_entry)
            DINGOS_results.append([DINGOS_design,DINGOS_pathway])
        return DINGOS_results

    def design_cycle(self,mol_entry,*args,**kwargs):
        reaction_step_limit = kwargs.get('reaction_step_limit', None)
        threshold = kwargs.get('threshold', None)

        mol = mol_entry[0]
        mol_id = int(mol_entry[1])
        mol_dist = float(mol_entry[2])
        mol = Chem.MolFromSmiles(mol)

        reaction_pathway = []

        for index in range(0,reaction_step_limit):
            product, product_id, product_distance,product_rxn = self.single_assembly_step([mol,mol_id], mol_dist,reaction_step=index)

            # Terminates the assembly if either no products are formed or if the product exceeds the MW threshold
            if product == None or Chem.rdMolDescriptors.CalcExactMolWt(product) > threshold:
                product = mol
                break

            print "Step =", index + 1
            print "Products :", Chem.MolToSmiles(product)

            # Saves the reaction pathway
            reaction_pathway.append(Chem.MolToSmiles(mol)+"."+product_rxn[0]+Chem.MolToSmiles(product))

            mol_dist = product_distance
            mol = product
            mol_id = product_id
        print "------------------------"
        reaction_pathway = "|".join(reaction_pathway)
        return Chem.MolToSmiles(product),reaction_pathway
