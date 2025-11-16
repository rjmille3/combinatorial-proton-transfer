# combinatorial-proton-transfer
This repository contains the code used for combinatorially generating proton transfer reactions.

We can set up a simple anaconda environment as follows:

conda create -n combinatorial python=3.8

conda activate combinatorial

pip install -r requirements.txt

Here are some example commands:

<pre markdown>
python gen_rxns.py \
       --acid_path      raw_data/51M/Acid.csv \
       --conbase_path   raw_data/51M/ConBase.csv \
       --output         51M_combinatorial.csv
</pre>
<pre markdown>
python gen_rxns_carbon_acid.py \
       --acid_path      raw_data/5K_Carbon_Acid/Carbon_Acids.csv \
       --conbase_path   raw_data/5K_Carbon_Acid/R2NH.csv \
       --base_class     "R2NH" \
       --output         R2NH_combinatorial.csv
</pre>

The "--acid_path" argument specifies a path to the csv file containing the acid data.

The "--conbase_path" argument specifies a path to the csv file containing the base data.

The "--output" argument specifies the location of the output file

The "--base_class" argument specifies which base class to use when generating reactions. This only applies to the carbon acid data.
