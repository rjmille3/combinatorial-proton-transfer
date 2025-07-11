# combinatorial-proton-transfer
This repository contains the code used for combinatorially generating proton transfer reactions.

We can set up a simple anaconda environment as follows:

conda create -n combinatorial python=3.8

conda activate combinatorial

pip install -r requirements.txt

Here are some example commands:

python gen_rxns.py \
       --acid_path      raw_data/51M/Acid.csv \
       --conbase_path   raw_data/51M/ConBase.csv \
       --output         51M_combinatorial.csv

python gen_rxns_carbon_acid.py \
       --acid_path      raw_data/16K_Carbon_Acid/Carbon_Acids.csv \
       --conbase_path   raw_data/16K_Carbon_Acid/R2NH.csv \
       --base_class     "R2NH" \
       --output         R2NH_combinatorial.csv
