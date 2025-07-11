#||=========================(PROJECT DECOPOSITION)=============================
#|| #Get labelled acid and base data and reactivity parameters from csv files
#|| #Combinatorially generate training steps by following the arrow pushing code convension (source atom = 10; sink atom = 20)
#|| #Generate a rate constant [log(k_1)] value according to the function of rate constants:
#||                           log(k_1) = (9 + ConBase_pKa - Acid_pKa)
#|| #Store SMIRKS and rate constant data onto a CSV file
#|| #Print output stats to user upon completion of the reaction generation
#||============================================================================

#Import libraries that must be defined before anything else (responsible for function definitions)
from typing import List
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import sys
import os
import random
import argparse

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger


from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')


#==============================================================================

#Modified from http://rdkit.blogspot.com/2016/09/avoiding-unnecessary-work-and.html
#partialSanitize - paritally sanitize the molecule to not change any strings for aromaticity
def partialSanitize(mol):

  #We are setting all states to 1 with the Chem.SANITIZE_ALL flag, but we are using a XOR (^) to turn off all specified flags
  Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_CLEANUPCHIRALITY^Chem.SANITIZE_SYMMRINGS)
#===END partialSanitize===

#==============================================================================

#openWorkbookByURL - Open and return an instance of the workbook containing the training data
def getDataFromLocalCSV(path: str) -> pd.DataFrame:
  return pd.read_csv(path, header=0, engine='python', encoding='utf-8')
#===END getDataFrameByURL===


#==============================================================================

#getIndexFromAtomMapNum - Return the index value of a specific Atom Map Number in a given molecule
def getIndexFromAtomMapNum(molecule, atomMapNum):
  #Seach through each atom in the molecule
  for atom in molecule.GetAtoms():
    #return index number if the atom map number matches the given value
    if (atom.GetAtomMapNum() == atomMapNum):
      return atom.GetIdx()

  #Error case, not found
  return -1
#=====END of getIndexFromAtomMapNum()=====


#==============================================================================

#getSingleReactantSMILES - return a single reactant string from a list of molecules
def getSingleReactantSMILES(reactantMoleculeList):
  #Combine the all molecules into a molecule list
  reactantsMolecule = reactantMoleculeList[0]

  for i in range(1, len(reactantMoleculeList)):
    reactantsMolecule = Chem.CombineMols(reactantsMolecule, reactantMoleculeList[i])

  #Return the Smiles string for all combined reactant molecules
  return Chem.MolToSmiles(reactantsMolecule)
#=====END of getSingleReactantSMILES()=====

#==============================================================================

#getSingleArrowPushingCode - get the arrow pushing code from a reaction SMILES String
def getSingleArrowPushingCode(reactantsSMILES, sourceSiteOrbital, sinkSiteOrbital):
  #Check to ensure the reactants SMILES is a valid SMILES String
  reactantsMol = Chem.MolFromSmiles(reactantsSMILES, sanitize=False)
  partialSanitize(reactantsMol)

  #Set up Source and Sink atom label list
  sourceLabelList = []
  sinkLabelList = []

  #Parse through reactants
  for atom in reactantsMol.GetAtoms():
    mapNum = atom.GetAtomMapNum()
    #Source atom indicated by (10 <= map number < 20) and (220 <= map number < 230)
    if ((mapNum >= 10 and mapNum < 20) or (mapNum >= 220 and mapNum < 230)):
      sourceLabelList.append(mapNum)
    #Sink atom indicated by (20 <= map number < 30) and (230 <= map number < 240)
    elif ((mapNum >= 20 and mapNum < 30) or (mapNum >= 230 and mapNum < 240)):
      sinkLabelList.append(mapNum)

  #sort lists in numberical order for better parsing
  sourceLabelList.sort()
  sinkLabelList.sort()

  #List to store Arrow-Pushing Code
  arrowPushingList = []
 
  #Create Arrow-Pushing Code
  #===source site===
  #(14=13,14 [From lp: 14 used] ; 12,13=11,12 [Bond to bond: 13&12 used] ; 10,11=10,20 [Connecting bond: 11&10 used])
  #Odd pairing
  if (len(sourceLabelList) % 2 != 0 and max(sourceLabelList) >= 11):
    arrowPushingCode = str(sourceLabelList[-1]) + '=' + str(sourceLabelList[-2]) + ',' + str(sourceLabelList[-1])
    arrowPushingList.append(arrowPushingCode)
    del sourceLabelList[-1]
  #Even pairing
  while (len(sourceLabelList) != 0 and max(sourceLabelList) >= 12):
    arrowPushingCode = str(sourceLabelList[-2]) + ',' + str(sourceLabelList[-1]) + '=' + str(sourceLabelList[-3]) + ',' + str(sourceLabelList[-2])
    arrowPushingList.append(arrowPushingCode)
    del sourceLabelList[-1]
    del sourceLabelList[-1]

  #===sink site===
  #(24,25=25 [To empty orbital: 25 used] ; 22,23=23,24 [Bond to bond: 23,24 used] ; 20,21=21,22 [Bond to bond: 21,22 used] ; 10=10,20 [Connecting bond: 20 used])
  #Even pairing
  if (len(sinkLabelList) % 2 == 0 and max(sinkLabelList) >= 21):
    arrowPushingCode = str(sinkLabelList[-2]) + ',' + str(sinkLabelList[-1]) + '=' + str(sinkLabelList[-1])
    arrowPushingList.append(arrowPushingCode)
    del sinkLabelList[-1]
  #Odd pairing
  while (len(sourceLabelList) != 0 and max(sinkLabelList) >= 22):
    arrowPushingCode = str(sinkLabelList[-3]) + ',' + str(sinkLabelList[-2]) + '=' + str(sinkLabelList[-2]) + ',' + str(sinkLabelList[-1])
    arrowPushingList.append(arrowPushingCode)
    del sinkLabelList[-1]
    del sinkLabelList[-1]

  #Deal with source/sink connecting bond
  atom_11_exists = False
  mol = Chem.MolFromSmarts(reactantsSMILES)
  for atom in mol.GetAtoms():
    if(atom.GetAtomMapNum() == 11):
      atom_11_exists = True
  #lone-pair
  sourceSinkArrow = ""
  #if (sourceSiteOrbital == "lp"):
  if(not atom_11_exists):
    sourceSinkArrow = "10=20"
  #signle or double
  #elif (sourceSiteOrbital == "sigma" or sourceSiteOrbital == "pi"):
  else:
    sourceSinkArrow = "10,11=10,20"

  #store arrow pushing code for the new connecting bond
  arrowPushingCode = str(sourceSinkArrow)

  #Sort Arrow-Pushing Codes numerically (Similar to Reaction Predictor)
  arrowPushingList.sort()

  #Create one string of text for Arrow-Pushing Codes
  for singleArrowPush in arrowPushingList:
    arrowPushingCode = str(arrowPushingCode) + ';' + str(singleArrowPush)

  #get rid of any whitespace
  arrowPushingCode.replace(' ', '')

  #Return Arrow-Pushing Code
  return arrowPushingCode
#=====END of getSingleArrowPushingCode()=====

#==============================================================================

#getSingleProductSMILES - Return a product SMILES String from a reactants SMILES String and arrow-pushing code
def getSingleProductSMILES(reactantsSMILES, arrowPushingCode):

  #Check to ensure the reactants SMILES is a valid SMILES String
  reactantsMol = Chem.MolFromSmiles(reactantsSMILES, sanitize=False)
  partialSanitize(reactantsMol)

  #Split the input string into separate electron arrow-pushes
  mappingSectionList = arrowPushingCode.split(';')

  #Create lists to store the source and sink sides of each electron arrow-push
  sourceList = []
  sinkList = []

  #Loop through each electron arrow-push
  for arrowPush in mappingSectionList:
    #Section off the source and sink side based on the equal sign
    splitArrowPush = arrowPush.split('=')

    #Left side is source side
    sourceSide = splitArrowPush[0].split(',')

    #Right side is sink side
    sinkSide = splitArrowPush[1].split(',')

    #Covnert the Source-Side string List to an int List
    for j in range(0, len(sourceSide)):
      sourceSide[j] = int (sourceSide[j])

    #Covnert the Sink-Side string List to an int List
    for j in range(0, len(sinkSide)):
      sinkSide[j] = int (sinkSide[j])

    #Append the new data to a Source and Sink list
    sourceList.append(sourceSide)
    sinkList.append(sinkSide)

  #Make new bonds from Atom Map Number at a specific index value
  #Create a new editable molecule to add new bonds and formal charge values
  edMol = Chem.EditableMol(reactantsMol)

  #Loop through each electron arrow-push for the specific reaction
  for j in range(0, len(sourceList)):
    #Special case when atomMapNum1=atomMapNum2
    if (len(sourceList[j]) + len(sinkList[j]) == 2):
      #change to atomMapNum1=atomMapNum1,atomMapNum2
      sinkList[j].append(sourceList[j][0])

    #Change the bond order and formal charge in the molecule
    #===Arrow Sources===
    #Execute when the source is a bond (2 atom mapping numbers for source)
    if (len(sourceList[j]) == 2):
      #Grab both atom index values in source bond from the atom map number
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sourceList[j][0])
      index2 = getIndexFromAtomMapNum(edMol.GetMol(), sourceList[j][1])

      #Store the source bond for the electron arrow-push
      sourceBond = edMol.GetMol().GetBondBetweenAtoms(index1, index2)

      #For a Zero-Ordered bond
      if (sourceBond == None):
        #Print an error message for trying to set a -1 ordered bond
        print("ERROR, REMOVING A ZERO ORDERED BOND!!!")
      else:
        #For a First-Ordered bond
        if (sourceBond.GetBondTypeAsDouble() == 1.0):
          #Change bond order from 1 to 0
          edMol.RemoveBond(index1, index2)

        #For a Second-Ordered bond
        elif (sourceBond.GetBondTypeAsDouble() == 2.0):
          #Change bond order from 2 to 1
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.SINGLE)

        #For a Third-Ordered bond
        elif (sourceBond.GetBondTypeAsDouble() == 3.0):
          #Change bond order from 3 to 2
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.DOUBLE)

        #For a Fourth-Ordered bond
        elif (sourceBond.GetBondTypeAsDouble() == 4.0):
          #Change bond order from 4 to 3
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.TRIPLE)

      #Get the updated index value for each atom map number in the bond
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sourceList[j][0])
      index2 = getIndexFromAtomMapNum(edMol.GetMol(), sourceList[j][1])

      #Get the formal charge for each atom
      charge1 = edMol.GetMol().GetAtomWithIdx(index1).GetFormalCharge()
      charge2 = edMol.GetMol().GetAtomWithIdx(index2).GetFormalCharge()

      #Add 1 to the formal charge for each atom
      charge1 += 1
      charge2 += 1

      #Store the atom that is apart of the source bond
      atom1 = edMol.GetMol().GetAtomWithIdx(index1)
      atom2 = edMol.GetMol().GetAtomWithIdx(index2)

      #Set the new formal charge of the atom
      atom1.SetFormalCharge(charge1)
      atom2.SetFormalCharge(charge2)

      #Replace the atom in the editable molecule
      edMol.ReplaceAtom(index1, atom1)
      edMol.ReplaceAtom(index2, atom2)

    #Execute when the source is a lone pair (1 atom mapping number for source)
    else:
      #FC = Valence - (2*lp + numBonds)
      #Losing a lone pair means gaining +2 charge to atom
      #Store the index of of the specified atom map number
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sourceList[j][0])

      #Store the formal charge of the atom
      charge1 = edMol.GetMol().GetAtomWithIdx(index1).GetFormalCharge()

      #Increase atom charge by 2 for losing lone pair
      charge1 += 2

      #Store the specified atom
      atom1 = edMol.GetMol().GetAtomWithIdx(index1)

      #Change the new formal charge of the atom
      atom1.SetFormalCharge(charge1)

      #Replace the atom in the editable molecule with the new atom
      edMol.ReplaceAtom(index1, atom1)

    #===Arrow Sinks===
    #Execute when the sink is a bond (2 atom mapping numbers for sink)
    if (len(sinkList[j]) == 2):
      #Get the index value from the specified atom map numbers in the sink bond
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sinkList[j][0])
      index2 = getIndexFromAtomMapNum(edMol.GetMol(), sinkList[j][1])

      #Store the bond for the sink of the electron arrow-push
      sinkBond = edMol.GetMol().GetBondBetweenAtoms(index1, index2)

      #For a Zero-Ordered bond
      if (sinkBond == None):
        #Change bond order from 0 to 1
        edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.SINGLE)
      else:
        #For a First-Ordered bond
        if (sinkBond.GetBondTypeAsDouble() == 1.0):
          #Change bond order from 1 to 2
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.DOUBLE)

        #For a Second-Ordered bond
        elif (sinkBond.GetBondTypeAsDouble() == 2.0):
          #Change bond order from 2 to 3
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.TRIPLE)

        #For a Third-Ordered bond
        elif (sinkBond.GetBondTypeAsDouble() == 3.0):
          #Change bond order from 3 to 4
          edMol.RemoveBond(index1, index2)
          edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.QUADRUPLE)

        ##For a n-Ordered bond
        #elif (sinkBond.GetBondTypeAsDouble() == n):
        #  #Change bond order from n to (n+1)
        #  edMol.RemoveBond(index1, index2)
        #  edMol.AddBond(index1, index2, order=Chem.rdchem.BondType.(n+1)))

        #For a Fourth-Ordered bond
        elif (sinkBond.GetBondTypeAsDouble() == 4.0):
          #Error for setting a bond order greater than 4
          print("ERROR, TRYING TO ADD A BOND ORDER TO A BOND ORDER OF 4!!!")

      #FC = Valence - (2*lp + numBonds)
      #Gaining bond means gaining -1 charge to each atom
      #Get the updated index value for each atom map number in the bond
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sinkList[j][0])
      index2 = getIndexFromAtomMapNum(edMol.GetMol(), sinkList[j][1])

      #Get the formal charge of each atom in the bond
      charge1 = edMol.GetMol().GetAtomWithIdx(index1).GetFormalCharge()
      charge2 = edMol.GetMol().GetAtomWithIdx(index2).GetFormalCharge()

      #subtract 1 to the formal charge of each atom
      charge1 -= 1
      charge2 -= 1

      #Store each atom in the sink bond
      atom1 = edMol.GetMol().GetAtomWithIdx(index1)
      atom2 = edMol.GetMol().GetAtomWithIdx(index2)

      #update the formal charge of each stored atom
      atom1.SetFormalCharge(charge1)
      atom2.SetFormalCharge(charge2)

      #Replace the atom in the molecule with an atom of updated formal charge
      edMol.ReplaceAtom(index1, atom1)
      edMol.ReplaceAtom(index2, atom2)

    #Execute when the sink is a lone pair (1 atom mapping number for sink)
    else:
      #FC = Valence - (2*lp + numBonds)
      #Gaining a lone pair means adding -2 charge to atom
      #Store the index of of the specified atom map number
      index1 = getIndexFromAtomMapNum(edMol.GetMol(), sinkList[j][0])

      #Store the formal charge of the sink atom
      charge1 = edMol.GetMol().GetAtomWithIdx(index1).GetFormalCharge()

      #subtract 2 to formal charge
      charge1 -= 2

      #Store sink atom
      atom1 = edMol.GetMol().GetAtomWithIdx(index1)

      #Set the new formal charge of the atom
      atom1.SetFormalCharge(charge1)

      #replace the atom in the molecule with the atom of updated formal charge
      edMol.ReplaceAtom(index1, atom1)

  #Check to see if the product is a proper SMILES String
  newMol = edMol.GetMol()
  partialSanitize(newMol)

  #Return the products SMILES String
  return Chem.MolToSmiles(newMol)
#=====END of getSingleProductSMILES()=====

#==============================================================================

#getSingleSMIRKS - Return a proper SMIRKS String, combining the individual data into one
def getSingleSMIRKS(reactantsSMILES, productsSMILES, arrowPushingCode):
  return str(reactantsSMILES) + ">>" + str(productsSMILES) + ' ' + str(arrowPushingCode)
#=====END of getSingleSMIRKS()=====

#==============================================================================

#getSingleABReactionConstant - Return the log(k_1) value
#if in range [-3,3), we consider the rate constant unreliable, replace with a constant for later processing (we remove these rate constants)
def getSingleABReactionConstant(ConBase_pKa, Acid_pKa):
  #Return the Reaction Contant log(k_1) = (9 + ConBase_pKa - Acid_pKa
  if (ConBase_pKa - Acid_pKa) >= -3 and (ConBase_pKa - Acid_pKa) < 3:
    return 6.2831853
  elif (ConBase_pKa - Acid_pKa) >= 3:
    return 9.00
  else:
    #equation from literature
    return round((9 + ConBase_pKa - Acid_pKa),2)

#=====END of getSingleABReactionConstant()=====

#==============================================================================


  
  
def getDataFromlocalCSVs(el_csv_path, nu_csv_path):
    dataDictionary = dict()  # Create an empty dictionary to store data

    # Sheets to look into
    csv_files = {"ConBase Final": nu_csv_path, "Acid Final": el_csv_path}

    # Loop through CSV files
    for currentSheet, csv_path in csv_files.items():
        try:
            # Attempt to read CSV file
            df = pd.read_csv(csv_path)
            # Define the number of items you want to print from each column
            num_items_to_print = 5  # Adjust as needed

            # Loop through columns in the CSV file
            for columnName in df.columns:
                # Handle any reading of blank columns
                if columnName == '' or columnName.startswith("Unnamed:"):
                    continue

                # Get the data in the column as a list
                dataColumn = df[columnName].tolist()

                # Create a key for the data dictionary through the notation: currentSheet/currentColumn
                key = f"{currentSheet}/{columnName}"
                
                
                # Convert from string to float array if necessary
                if columnName == "sN" or columnName == 'ConBase_pKa' or columnName == 'Acid_pKa':
                    dataColumn = [float(value) for value in dataColumn]

                

                # Add values to the data dictionary
                dataDictionary[key] = dataColumn

        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}")

    # Return the dictionary containing all of the data in the file
    return dataDictionary


def _worker(task):
    """
    Runs in a separate process.
    `task` is a tuple:
       (index, nu_smiles, ac_smiles, N, E)
    We return (index, row_dict) so the parent can
    sort/collect rows in input order.
    """
    (idx, nu_smiles, ac_smiles, N, E) = task

    RDLogger.DisableLog('rdApp.*')      # silence RDKit in every worker

    # 1) ── Build reactant SMILES ──────────────────────────────────────────────
    nu_mol = Chem.MolFromSmiles(nu_smiles, sanitize=False)
    ac_mol = Chem.MolFromSmiles(ac_smiles, sanitize=False)
    partialSanitize(nu_mol)
    partialSanitize(ac_mol)
    react_smiles = getSingleReactantSMILES([nu_mol, ac_mol])

    # 2) ── Arrow pushing + products + SMIRKS ─────────────────────────────────
    arrow = getSingleArrowPushingCode(react_smiles, "", "")
    prod_smiles = getSingleProductSMILES(react_smiles, arrow)
    smirks = getSingleSMIRKS(react_smiles, prod_smiles, arrow)

    # 3) ── log(k1) ───────────────────────────────────────────────────────────
    logk = getSingleABReactionConstant(N, E)

    row = {
        "SMIRKS":      smirks,
        "log(k_1)":    logk,
        "Acid_pKa":    E,
        "ConBase_pKa": N
    }
    return idx, row
# -----------------------------------------------------------------------------


def make_dataframe_mp(data_dict, max_workers=None, chunk=1_000):
    """Parallel version of `printTrainingDataDirectly`."""


    nu    = data_dict["ConBase Final/SMILES Labelled"]
    nu_pK = data_dict["ConBase Final/ConBase_pKa"]
    ac    = data_dict["Acid Final/SMILES Labelled"]
    ac_pK = data_dict["Acid Final/Acid_pKa"]


    #testing smaller cases
    #MAX_NU = 1000
    #MAX_AC = 1000
    #nu    = nu[:MAX_NU]
    #nu_pK = nu_pK[:MAX_NU]
    #ac    = ac[:MAX_AC]
    #ac_pK = ac_pK[:MAX_AC]

    tasks = (
        (i*len(ac) + j, nu[i], ac[j], nu_pK[i], ac_pK[j])
        for i, j in product(range(len(nu)), range(len(ac)))
    )

    rows = [None] * (len(nu) * len(ac))        # pre-allocate
    total = len(rows)

    # Pick sensible default: all physical cores if unspecified
    if max_workers is None:
        max_workers = mp.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # executor.map keeps order automatically, but we chunk so that large
        # lists don’t have to be materialised in RAM all at once.
        for idx, row in exe.map(_worker, tasks, chunksize=chunk):
            rows[idx] = row
            if idx % 10_000 == 0:      # progress heartbeat
                print(f"done {idx/total:.1%}", flush=True)

    df = pd.DataFrame(rows,
                      columns=["SMIRKS", "log(k_1)", "Acid_pKa", "ConBase_pKa"])
    return df



#MAIN (Starting Execution Point)
if (__name__ == "__main__"):

  parser = argparse.ArgumentParser(
        description="Process ACID and CONBASE files and produce an output.")
  parser.add_argument(
      "--acid_path",
      required=True,
      help="path to the ACID input file")
  parser.add_argument(
      "--conbase_path",
      required=True,
      help="path to the CONBASE input file")
  parser.add_argument(
      "--output",
      required=True,
      help="path where the output should be written")
  parser.add_argument("--nprocs",
      type=int, 
      default=None,
      help="number of worker processes (default: all cores)")

  args = parser.parse_args()


  #Get data from Data File from Dashuta Acids and Bases
  dataDictionary = getDataFromlocalCSVs(args.acid_path, args.conbase_path)
  #Error check for function return
  if (len(dataDictionary) == 0):
    #Exit the program
    print("Exiting program due to logical error...")
    sys.exit()

  #Print unsorted training data directly to a CSV file
  trainingDataDF = make_dataframe_mp(dataDictionary, max_workers=args.nprocs)

  print(f"Total reactions generated: ", len(trainingDataDF))

  #=========================
  # Now Begin the Sorting Process
  #=========================

  trainingDataDF = trainingDataDF.replace(np.nan, '')

  pd.options.mode.chained_assignment = None  # default='warn'
  #filter to only have certain pka values
  trainingDataDF_new = trainingDataDF[ (trainingDataDF['log(k_1)'] >= 3.0000) ]
  trainingDataDF_new['log(k_1)'] = trainingDataDF_new['log(k_1)'].astype(str) # Convert numerical values into string format in column log(k_1)
  trainingDataDF_new = trainingDataDF_new.replace("6.2831853", "") # Replace value of 2pi = 6.2831853 with empty space
  trainingDataDF_new['log(k_1)'] = pd.to_numeric(trainingDataDF_new['log(k_1)'], errors='coerce') #Return remaining string values into numerical format


  print("number of reactions with log(k_1) >= 3: ", len(trainingDataDF_new))
    
  trainingDataDF_new.to_csv(args.output, index=False)