#!/usr/bin/env python3
"""
generate_carbon_acid_combinations_mp.py
Multiprocess generator for carbon-acid/hetero-base training data.

Run:

  python generate_carbon_acid_combinations_mp.py \
         --acid_path   Acid.csv \
         --conbase_path ConBase.csv \
         --output      CarbonPTset.csv \
         --nprocs      32
"""
# ──────────────────────────────────────────────────────────────────────────────
# Imports and RDKit boiler-plate (unchanged)                                   |
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
import argparse, sys, os, random, math
import multiprocessing as mp
from itertools import product
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.warning')

# ---------- helper functions already present in your “working” script ---------
# * partialSanitize
# * getSingleReactantSMILES
# * getSingleArrowPushingCode
# * getSingleProductSMILES
# * getSingleSMIRKS
# * getIndexFromAtomMapNum
# -----------------------------------------------------------------------------


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


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load both CSVs into one dictionary                                         |
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_COLS_CONBASE = {"ConBase_pKa", "pB", "qB"}
NUMERIC_COLS_ACID    = {"Acid_pKa", "beta", "pC", "qC", "log_k_o"}

def getDataFromlocalCSVs(el_csv_path: str, nu_csv_path: str, base_class: str):
    """
    Reads the two CSVs (“ConBase Final” and “Acid Final” sheets in Colab) into
    a flat dict of numpy/str lists keyed as  SHEET/COLUMN. Everything the
    worker needs is converted to Python floats up-front.
    """
    data = {}

    for sheet_name, path, numeric_cols in (
        ("ConBase Final", nu_csv_path, NUMERIC_COLS_CONBASE),
        ("Acid Final",    el_csv_path, NUMERIC_COLS_ACID),
    ):
        df = pd.read_csv(path)

        if sheet_name == "Acid Final":
            df = df[df["Base Class"] == base_class]

        for col in df.columns:
            if col.startswith("Unnamed") or col == "":
                continue
            key = f"{sheet_name}/{col}"
            series = df[col]
            if col in numeric_cols:
                data[key] = series.astype(float).tolist()
            else:
                data[key] = series.tolist()
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 2. Carbon-acid rate constant                                                 |
# ──────────────────────────────────────────────────────────────────────────────
def getSingleABReactionConstant(beta, ConBase_pKa, Acid_pKa,
                                qB,  pB,           qC, pC, log_k_o):
    """
    log(k1) = β * (pK_HB - pK_HC + log10( (pB*qC)/(qB*pC) ))
            + log(k_o) + log10(qB*pC)
    """
    return (
        beta *
        (ConBase_pKa - Acid_pKa + math.log10((pB * qC) / (qB * pC)))
        + log_k_o
        + math.log10(qB * pC)
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Worker function                                                            |
# ──────────────────────────────────────────────────────────────────────────────
def _worker(task):
    (idx,
     nu_smiles, ac_smiles,
     beta, N, E, qB, pB, qC, pC, log_k0) = task

    # silence RDKit *inside* each child
    RDLogger.DisableLog('rdApp.*')

    # 1. reactant SMILES
    nu_mol = Chem.MolFromSmiles(nu_smiles,  sanitize=False)
    ac_mol = Chem.MolFromSmiles(ac_smiles,  sanitize=False)
    partialSanitize(nu_mol);  partialSanitize(ac_mol)
    react_smiles = getSingleReactantSMILES([nu_mol, ac_mol])

    # 2. products + SMIRKS
    arrow      = getSingleArrowPushingCode(react_smiles, "", "")
    prod_smiles = getSingleProductSMILES(react_smiles, arrow)
    smirks      = getSingleSMIRKS(react_smiles, prod_smiles, arrow)

    # 3. log(k1)
    logk = getSingleABReactionConstant(beta, N, E, qB, pB, qC, pC, log_k0)

    return idx, {
        "SMIRKS": smirks,
        "log(k_1)": logk,
        "Acid_pKa": E,
        "ConBase_pKa": N,
        "beta": beta,
        "qB": qB, "pB": pB,
        "qC": qC, "pC": pC,
        "log_k_o": log_k0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Parallel driver                                                            |
# ──────────────────────────────────────────────────────────────────────────────
def make_dataframe_mp(data, max_workers=None, chunk=10_000):

    nu_smiles  = data["ConBase Final/SMILES Labelled"]
    nu_pKa     = data["ConBase Final/ConBase_pKa"]
    pB_list    = data["ConBase Final/pB"]
    qB_list    = data["ConBase Final/qB"]

    ac_smiles  = data["Acid Final/SMILES Labelled"]
    ac_pKa     = data["Acid Final/Acid_pKa"]
    beta_list  = data["Acid Final/beta"]
    pC_list    = data["Acid Final/pC"]
    qC_list    = data["Acid Final/qC"]
    logk0_list = data["Acid Final/log_k_o"]

    n_nu, n_ac = len(nu_smiles), len(ac_smiles)
    rows = [None] * (n_nu * n_ac)

    if max_workers is None:
        max_workers = mp.cpu_count()

    tasks = (
        (i*n_ac + j,
         nu_smiles[i], ac_smiles[j],
         beta_list[j],        # β from acid
         nu_pKa[i], ac_pKa[j],
         qB_list[i], pB_list[i],
         qC_list[j], pC_list[j],
         logk0_list[j])
        for i, j in product(range(n_nu), range(n_ac))
    )

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for idx, row in exe.map(_worker, tasks, chunksize=chunk):
            rows[idx] = row
            if idx % 10_000 == 0:
                print(f"done {idx/len(rows):.1%}", flush=True)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI / main                                                                 |
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate carbon-acid training-set combinations.")
    ap.add_argument("--acid_path",    required=True)
    ap.add_argument("--conbase_path", required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--nprocs", type=int, default=None)
    ap.add_argument("--base_class", type=str, required=True)
    args = ap.parse_args()

    data = getDataFromlocalCSVs(args.acid_path, args.conbase_path, args.base_class)
    if not data:
        sys.exit("No data read – check input paths.")

    df = make_dataframe_mp(data, max_workers=args.nprocs)
    print("Total reactions generated:", len(df))

    # keep only plausible rate constants (≥ –1)
    df = df[df["log(k_1)"] >= -1.0].reset_index(drop=True)
    print("Entries with log(k_1) ≥ -1:", len(df))

    df.to_csv(args.output, index=False)
    print("Written to", args.output)


if __name__ == "__main__":
    main()
