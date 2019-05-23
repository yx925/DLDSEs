from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.rdMolDescriptors import *

class reaction:
    def __init__(self,reaction_name,reaction,reaction_label):
        self.reaction_name = reaction_name
        self.reaction = AllChem.ReactionFromSmarts(reaction)
        rdChemReactions.ChemicalReaction.Initialize(self.reaction)
        self.reaction_label = reaction_label

    def Check_number_reactants(self):
        return self.reaction.GetNumReactantTemplates()

    def IsMoleculeReactant(self,input_mol):
        Is_building_block_a_Reactant = rdChemReactions.ChemicalReaction.IsMoleculeReactant(self.reaction, input_mol)
        condition = None
        if input_mol == condition:
            Is_building_block_a_Reactant = False
        return Is_building_block_a_Reactant

    def Reactant_position(self,mol):
        Reactants_generalized = self.reaction.GetReactants()
        Reactant_positions = []
        for position, position_check in enumerate([Chem.Mol.HasSubstructMatch(mol, reactant) for reactant in Reactants_generalized]):
            if position_check == True:
                Reactant_positions.append(position)
        return Reactant_positions

    def Reaction_components(self):
        Reactants_generalized = self.reaction.GetReactants()
        Reactants_smarts = [Chem.MolToSmiles(mol).replace('*','C') for mol in Reactants_generalized]
        Reactants_components = [Chem.MolFromSmiles(mol) for mol in Reactants_smarts]

        Product_generalized = self.reaction.GetProducts()
        Product_smarts = Chem.MolToSmiles(Product_generalized[0]).replace('*','C')
        Product_components = Chem.MolFromSmiles(Product_smarts)

        return Reactants_components+[Product_components]

    def PerformReaction(self,input_mol):
        product = []
        #input_mol = self.Reactant_condition(input_mol)
        
        # Performs the reaction for the reactants in all possible reactant positions 
        for shift in range(0,len(input_mol)):
            product = np.append(product,np.concatenate([self.reaction.RunReactants(list(np.roll(input_mol,shift)))]))

        #product = self.Product_condition(input_mol,product)
        
        # Removes the duplicates
        if len(product) != 0:
            product_smiles = [Chem.MolToSmiles(mol) for mol in product]
            product_smiles = list(set(product_smiles))
            product = [Chem.MolFromSmiles(mol) for mol in product_smiles]

        return product

    def Reactant_condition(self,input_mol):
        '''
        This is an optional function that allows the user to apply an 
        external condition to the reactants for a given reaction SMARTS
        '''
        reactant_list = []
        if self.reaction_name == "reaction name":
            # Some user defined condition
            condition = True
            for mol in input_mol:
                try:
                    if condition:
                        reactant_list.append(mol)
                except:
                    continue
        return reactant_list

    def Product_condition(self,input_mol,product):
        '''
        This is an optional function that allows the user to apply an 
        external condition to the products for a given reaction SMARTS
        '''
        product_list = []
        if self.reaction_name == "reaction name":
            # Some user defined condition
            condition = True
            for mol in input_mol:
                try:
                    if condition:
                        product_list.append(mol)
                except:
                    continue
        return product_list

