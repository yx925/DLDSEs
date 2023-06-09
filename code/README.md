# Automated de novo molecular design by hybrid machine intelligence and rule-driven chemical synthesis

#### Alexander Button, Daniel Merk, Jan A. Hiss & Gisbert Schneider

## Abstract

_Chemical creativity in the design of synthetic chemical entities (NCEs) with druglike properties has been the domain of medicinal chemists. Here, we explore the capability of a chemistry-savvy machine intelligence to generate synthetically accessible molecules. DINGOS (Design of Innovative NCEs Generated by Optimization Strategies) is a virtual assembly method that combines a rule-based approach with a machine learning model trained on successful synthetic routes described in chemical patent literature. This unique combination enables a balance between ligand-similarity based generation of innovative compounds by scaffold hopping and forward-synthetic feasibility of the designs. In a prospective proof-of-concept application, DINGOS successfully produced sets of de novo designs for four approved drugs that were in agreement with the desired structural and physicochemical properties. Target prediction indicated more than 50% of the designs as biologically active. Four selected computer-generated compounds were successfully synthesized in accordance with the synthetic route proposed by DINGOS. The results of this study demonstrate the capability of machine learning models to capture implicit chemical knowledge from chemical reaction data and suggest feasible syntheses of new chemical matter._

## Code

The code presented here represents the DINGOS _de novo_ design method. The main script is called **DINGOS.py** and can be found in **/code/DINGOS_CODE/00_main/**. Executing this script runs the entire procedure. All of the other components of the DINGOS method can be found in the various directories in the **DINGOS_CODE** folder. 

## Data

The datasets used by DINGOS are found in the directory **/data/Databases**. There are four datasets used by DINGOS:

- The MOL_DB which contains all the molecular structures.
- The RXN_DB containing all of the reaction SMARTS.
- The Descriptor_DB which contains all of the descriptor information of the MOL_DB entries.
- The mol_rxn_DB which relates the reaction entries of RXN_DB to the molecular entries of MOL_DB.

The parameters used for the DINGOS run are all set in the file **input_file.csv** which is located in the **/data** directory. This file is read in by DINGOS and contains all of the relevant run parameters, including the SMILES of the template molecule. These parameters can be changed by editing the csv file. 

The trained machine learning method **MLP_method** is also located in the **/data** directory. This model was trained to predict the MACCSKeys fingerprint of a building block molecule from its corresponding start and product molecules' fingerprints. This method can be extended to other fingerprints, however, this requires updating the machine learning method.

## Running DINGOS

To run the method, click the **Run** button in the top right-hand corner. The outputs can be found in the **/results** folder. 

The test example (Celecoxib) takes approximately 4 minutes to generate 300 molecules. 

## Loading Custom Sets

The DINGOS method, in principle, can be applied to any set of molecules and reactions; however, in order to use these sets, they first need to be preprocessed. This is done with the script **load_dataset.py**. This script loads in the molecule and reaction data, calculates the descriptor values, and generates the mol-rxn relational junction array.

**load_dataset.py** will generated the four **.npy** data files needed by DINGOS. The path to these files still needs to be set in the input file.

An example of a custom molecule and reaction dataset can be found in **data/Datasets/TEST_SET**. These include:

- mol_set.txt which contains a canonical SMILES as each line
- rxn_set.txt which contains the reaction name, reaction SMARTS, and reaction label separated by a '|' at each line.

The molecular descriptor is defined within **load_dataset.py**

## Publication Data 

We could not provide the original preprocessed datasets used for this publication; however, we have provided the reaction set **rxn_set.txt** and the CAS numbers of the molecules used  **publication_cas_number.txt** which can be found in **data/Datasets/PUBLICATION_SET**

