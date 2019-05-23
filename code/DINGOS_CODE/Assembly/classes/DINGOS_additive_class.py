import pickle
import sys
from rdkit import Chem
import numpy as np

import os
root_path = os.getcwd().split('DINGOS_CODE')[0]+'DINGOS_CODE'
sys.path.insert(0,root_path+'/Machine_learning_method/classes/')
import ML_classes
sys.path.insert(0,root_path+'/Descriptors/classes')
from Descriptor_classes import *
sys.path.insert(0,root_path+'/Assembly')
from run_reaction import *

import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import functools
from keras.models import load_model
import time
start_time = time.time()
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

class DINGOS_ADD:
    def __init__(self,target,mol_rxn_junction,desc_func,distance_metric):
        self.mol_rxn_junction = mol_rxn_junction
        self.target = target
        self.desc_func = desc_func
        self.distance_metric = distance_metric

        self.building_block_db = None
        self.descriptor_db = None
        self.bb_prediction = None
        self.assembly_method = None
        self.ML_class = None

    def load_building_block_db(self,*args, **kwargs):
        mol_db = kwargs.get('mol_db', None)
        descriptor_db = kwargs.get('descriptor_db', None)
        substructure_list = kwargs.get('substructure_list', 'None')

        # Filters the results based on the substructure list
        if 'None' not in substructure_list:
            filtered_index = []
            for list_index, mol in enumerate(mol_db):
                if True not in [Chem.MolFromSmiles(mol[1]).HasSubstructMatch(Chem.MolFromSmiles(substrucut)) for substrucut in substructure_list]:
                    filtered_index.append(list_index)
            mol_db = mol_db[filtered_index]
            descriptor_db = descriptor_db[np.in1d(descriptor_db[:,0],mol_db[:,0])]

        return [mol_db, descriptor_db]

    def predictive_model(self,*args, **kwargs):
        # Sets the machine learning model
        model = kwargs.get('model', None)
        feature_extraction = kwargs.get('feature_extraction', None)
        sample = kwargs.get('sample', None)

        # Allows for models that utilize a representation with reduced features 
        if feature_extraction == None:
            return model.predict(sample)
        else:
            return model.predict(feature_extraction(sample))

    def set_assembly_method(self,**kwargs):
        rxn_db = kwargs.get('rxn_db', None)
        feature_method = kwargs.get('feature_method', None)
        ML_method = kwargs.get('ML_method', None)
        start_fragment_limit = kwargs.get('start_fragment_limit', None)
        substructure_list = kwargs.get('substructure_list', 'None')
        flagged_reactions = kwargs.get('flagged_reactions', 'None')

        ML = ML_classes.Make_prediction(self.building_block_db,self.mol_rxn_junction,self.descriptor_db,rxn_db, self.desc_func)
        self.ML_class = ML

        Machine_learning_method = functools.partial(self.predictive_model, model=ML_method,feature_extraction=feature_method)
        bb_prediction_method = functools.partial(ML.Predict_building_block_from_product_mol,ML_method=Machine_learning_method,substructure_list=substructure_list)
        Assembly_method = functools.partial(ML.Query_molecule,start_fragment_limit=start_fragment_limit,flagged_reactions=flagged_reactions,reaction_length_limit=1)
        return bb_prediction_method, Assembly_method

    def run_DINGOS(self,**kwargs):
        rxn_limit = kwargs.get('rxn_limit', None)
        number_of_products = kwargs.get('number_of_products', None)
        mass_threshold = kwargs.get('mass_threshold', None)
        start_molecule_set = kwargs.get('start_molecule_set', 'None')
        start_mol_ML = kwargs.get('start_mol_ML', 'Null')

        DR = DINGOS_run(self.target, self.bb_prediction, self.assembly_method, self.desc_func, self.distance_metric,start_mol_ML,self.ML_class)
        DINGOS_designs = DR.full_assembly(start_molecule_set=start_molecule_set,start_mol_library=[self.building_block_db,self.descriptor_db], reaction_length_limit =rxn_limit, product_limit = number_of_products,mass_threshold =mass_threshold)
        return DINGOS_designs