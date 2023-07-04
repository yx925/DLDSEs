import pickle
import numpy as np
try:
    data = np.load('C:/Users/GrandPharma/OneDrive/Desktop/2023-03/DLDSEs.zip1/data/Databases/TEST_database/custom_set-DESCRIPTOR_DB_DATA_MACCSkeys.npy')
except ValueError as e:
    if str(e).startswith("invalid literal for int() with base 10"):
        data = np.empty(0)  # Replace with your desired behavior for invalid data
    else:
        raise e

import sys
import rdkit
from rdkit import Chem
import numpy as np
from rdkit.Chem import MACCSkeys
import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'
sys.path.insert(0, root_path+'/Machine_learning_method/classes/')
import ML_classes
sys.path.insert(0, root_path+'/Run_reaction/classes')
from Descriptor_classes import *
sys.path.insert(0, root_path+'/Assembly/classes/')
from DINGOS_additive_class import DINGOS_ADD
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from time import gmtime, strftime
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import hamming_loss
import csv

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

class DINGOS:
    def __init__(self):
        self.Template_SMILES = None
        self.Building_block_label = None
        self.included_subgroups = None
        self.excluded_subgroups = None
        self.flagged_reactions = None
        self.start_fragment_limit = None
        self.mass_limit = None
        self.bb_mass_limit = None
        self.rxn_limit = None
        self.product_limit = None
        self.predict_start_mol = None
        self.Authour_name = None
        self.Template_name = None

        self.Mol_descriptors = None
        self.building_block_set = None
        self.descriptor_db = None
        self.selected_reactions = None
        self.rxn_array = None
        self.mol_rxn_array = None
        self.ML_method = None
        self.preselected_rxns = None

        self.building_block_label = None
        self.reaction_label = None
        self.descriptor_label = None
        self.ML_method_filename = None

        self.bb_index = None
        # self.database_path = "../../../data/Databases/"
        self.database_path = "C:/Users/GrandPharma/OneDrive/Desktop/2023-03/DLDSEs.zip1/data/Databases"
        self.unique_id = None

    def read_input_parameters(self,file_path):
        start_time = time.time()
        file = open(file_path,'r')
        input_file = csv.reader(file,delimiter=',')

        file_content = []

        for line in input_file:
            file_content.append(line)

        file_content = np.asarray(file_content)

        self.building_block_label, self.reaction_label, self.descriptor_label, self.ML_method_filename,self.bb_mass_limit,self.preselected_rxns = file_content[2][:6]

        self.set_descriptor(self.descriptor_label)
        self.load_building_block_set()
        self.load_building_block_descriptor_set()
        self.load_reaction_set()
        self.load_mol_reaction_junction()
        self.load_ML_method(self.ML_method_filename)

        for parameter_list in file_content[5:]:
            parameter_ranges = np.where(parameter_list == '')[0]
            self.Authour_name, self.Template_name, self.Template_SMILES = parameter_list[:parameter_ranges[0]]
            self.mass_limit, self.product_limit,self.start_fragment_limit,self.rxn_limit = np.asarray(parameter_list[parameter_ranges[0]+1:parameter_ranges[1]],dtype=int)
            self.flagged_reactions,self.included_subgroups,self.excluded_subgroups = parameter_list[parameter_ranges[1]+1:parameter_ranges[2]]
            self.predict_start_mol = parameter_list[-1]

            self.flagged_reactions = self.flagged_reactions.split(',')
            self.included_subgroups = self.included_subgroups.split(',')
            self.excluded_subgroups = self.excluded_subgroups.split(',')

            flagged_reaction_names = str(tuple(self.flagged_reactions)).replace(",)",")")
            # Converts the flagged reaction names to ids
            flagged_indices = np.intersect1d(self.rxn_array[:,1],flagged_reaction_names,return_indices=True)[1]
            self.flagged_reactions = np.asarray(flagged_indices)

            print ("template name ="),self.Template_name
            # Runs the DINGOS method
            output = self.run_DINGOS()
            print ("Run time = "), time.time() - start_time

    def run_DINGOS(self):
        product_mol = Chem.MolFromSmiles(self.Template_SMILES)
        # Sets the DINGOS_additive_class
        DINGOS = DINGOS_ADD(product_mol, self.mol_rxn_array, self.Mol_descriptors, hamming_loss)

        date = strftime("%Y-%m-%d", gmtime())
        DINGOS_run_label = self.Authour_name + "_" + self.Template_name + "_" + date

        # Sets the building block and descriptor databases
        DINGOS.building_block_db, DINGOS.descriptor_db = DINGOS.load_building_block_db(mol_db=self.building_block_set,
                                                                                 descriptor_db=self.descriptor_db,
                                                                                 excluded_substructure_list=self.excluded_subgroups)
        # Sets the predictive and assembly methods
        DINGOS.bb_prediction, DINGOS.assembly_method = DINGOS.set_assembly_method(design_label=DINGOS_run_label,
                                                                            rxn_db=self.rxn_array,
                                                                            ML_method=self.ML_method, start_fragment_limit=self.start_fragment_limit,
                                                                            flagged_reactions=self.flagged_reactions)
        
        # Runs DINGOS using the distance to the template for the selection of the starting molecule
        if self.predict_start_mol == 'FALSE':
            DINGOS_designs = DINGOS.run_DINGOS(rxn_limit=self.rxn_limit,
                                         number_of_products=self.product_limit,
                                         mass_threshold=self.mass_limit,
                                         included_substructure_list=self.included_subgroups)
            
        # Runs DINGOS using the distance to the predicted building block fingerprint for the selection of the starting molecule
        elif self.predict_start_mol == 'TRUE':
            DINGOS_designs = DINGOS.run_DINGOS(rxn_limit=self.rxn_limit,
                                         number_of_products=self.product_limit,
                                         mass_threshold=self.mass_limit,
                                         start_mol_ML=self.ML_method,
                                         included_substructure_list=self.included_subgroups)

        # Saves the outputs to the output csv file
        output_file = open("../../../results/"+self.Template_name+"_DINGOS_DESIGNS.txt","w+")
        output_file.write("DINGOS design number|SMILES|Reaction pathway\n")
        for index,design in enumerate(DINGOS_designs):
            DINGOS_design = design[0]
            reaction_pathway = design[1]
            if len(reaction_pathway) == 0:
                reaction_pathway = "Null"
            output_file.write("DINGOS design "+str(index+1)+"|"+DINGOS_design+"|"+reaction_pathway+"\n")
        output_file.close()


    def set_descriptor(self,descriptor_label):
        descriptor = Descriptors()
        if descriptor_label == 'MACCSkeys':
            Mol_desc = descriptor.mol_MACCSkey
            
        self.Mol_descriptors = Mol_desc

    def load_building_block_set(self):
        start_time = time.time()
        MW_limits = self.bb_mass_limit.split(",")
        # Locates and loads the MOL_DB numpy array
       # directory_contents = os.listdir(self.database_path+"/"self.building_block_label)
        directory_contents = os.listdir(self.database_path + "/" + self.building_block_label)

        # directory_contents = os.listdir('C:/Users/GrandPharma/OneDrive/Desktop/2023-03/DLDSEs.zip1/data/Databases/TEST_database')
        mol_file = [db_file for db_file in directory_contents if "MOL_DB_DATA" in db_file]
        # self.building_block_set = np.load(self.database_path+"/"self.building_block_label+"/"+mol_file[0])
        self.building_block_set = np.load(self.database_path + "/" + self.building_block_label + "/" + mol_file[0])
        # self.building_block_set = np.load('')

        # Removes all duplicates present within the set
        self.unique_id = np.unique(self.building_block_set[:,1],return_index=True)[1]
        self.building_block_set = self.building_block_set[self.unique_id]
        self.building_block_set = self.building_block_set[np.argsort(np.asarray(self.building_block_set[:,0],dtype=int))]
        
        # Filters MOL_DB based on the user defined molecular weight limits
        print ("BB MW limit is "),MW_limits
        mw_values = np.asarray(self.building_block_set[:, 2], dtype=float)
        self.bb_index = [np.where((mw_values > int(MW_limits[0]))*(mw_values < int(MW_limits[1])))]
        self.building_block_set = np.asarray(self.building_block_set[self.bb_index][0])
        self.building_block_set = self.building_block_set[:, :2]
        print ("Building block set loaded :"),time.time()-start_time, "seconds"

    def load_building_block_descriptor_set(self):
        start_time = time.time()
        # Locates and loads the DESCRIPTOR_DB numpy array
        directory_contents = os.listdir(self.database_path + "/" + self.building_block_label)
        desc_file = [db_file for db_file in directory_contents if "DESCRIPTOR_DB_DATA" in db_file and self.descriptor_label in db_file]
        import numpy as np
        self.descriptor_db = np.load(self.database_path+"/"+self.building_block_label+"/"+desc_file[0])

        # Retrieves the corresponding descirptor values of the building block set
        
        # bb_id_list = np.asarray(self.building_block_set[:,0],dtype=int)
        bb_id_list = np.asarray(self.building_block_set[:,0], dtype=str)

        self.descriptor_db = self.descriptor_db[np.in1d(self.descriptor_db[:,0],bb_id_list)]
        
        self.descriptor_db = self.descriptor_db[:,:-1]
        print ("Descriptor values calculated :"),time.time()-start_time, "seconds"

    def load_reaction_set(self):
        start_time = time.time()
        # Locates and loads the RXN_DB numpy array
        directory_contents = os.listdir(self.database_path + "/" + self.building_block_label)
        rxn_file = [db_file for db_file in directory_contents if "RXN_DB_DATA" in db_file]
        self.rxn_array = np.load(self.database_path+"/"+self.building_block_label+"/"+rxn_file[0])

        # Removes reactions that are not contained within a preselected list
        if self.preselected_rxns != 'None':
            self.rxn_array = self.rxn_array[np.in1d(self.rxn_array[:, 2], self.preselected_rxns)]
            
        print ("Reaction set loaded :"),time.time()-start_time, "seconds"

    def load_mol_reaction_junction(self):
        start_time= time.time()
        
        # Locates and loads the mol_rxn_DB numpy array
        directory_contents = os.listdir(self.database_path + "/" + self.building_block_label)
        mol_rxn_file = [db_file for db_file in directory_contents if "mol_rxn_DB_DATA" in db_file]
        self.mol_rxn_array = np.load(self.database_path+"/"+self.building_block_label+"/"+mol_rxn_file[0])
        
        # Extracts the building block and reaction ids
        bb_id = np.asarray(self.building_block_set[:,0],dtype=str)
        rxn_id = np.asarray(self.rxn_array[:,0], dtype=int)
        
        # Removes entries that are outside of the building block and reaction sets
        self.mol_rxn_array = self.mol_rxn_array[np.in1d(self.mol_rxn_array[:,0],bb_id)*np.in1d(self.mol_rxn_array[:,1],rxn_id)]
        print ("Molecule reaction junction table loaded :"),time.time()-start_time, "seconds"
        
    def load_ML_method(self,method_filename):
        start_time = time.time()
        self.ML_method = load_model('../../../data/'+method_filename)
        print ("Machine learning method loaded :"), time.time() - start_time, "seconds"

D = DINGOS()
D.read_input_parameters('C:/Users/GrandPharma/OneDrive/Desktop/2023-03/DLDSEs.zip1/data/'+ str(sys.argv[1]))
