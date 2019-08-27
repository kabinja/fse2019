# Dataset for The Importance of Accounting for Real-World Labelling When Predicting Software Vulnerabilities

This repo contains the replication set for the paper "The Importance of Accounting for Real-World Labelling When Predicting Software Vulnerabilities" presented at ESEC/FSE '19.

## Getting started

### Prerequisite

* Python 3.6 or later

### Get the environment ready

Before anything make sure to install all the requirements using the command 

    pip install -u requirements.txt

This command installs all the dependencies required for running the scripts.
We advise to install the dependencies in a dedicated virtual environment using mkvirtualenv.

## Running the scripts

The repo contains 5 entry points generating results. Each script is executed by simply calling it using the pythong interpreter:

    python script_name.py

1. file_distribution.py: Generates the graphics displaying for each project the number of files and the percentage of vulnerable files per release.
2. before_after_fix.py: Computes the performances of vulnerability classification before and after a vulnerability was fixed.
3. perfoamnces.py: Computes the performances of the vulnerability prediction model for different approaches.
4. score_topn.py: Compute the top 10 score of the vulnerability prediction model for the different approaches.
5. data_leakage.py: Computes the data leakage introduce by using every data available during this evaluation against using realistic data.

### Results

All the output are stored in the results/figures and results/tables folders as individual pdf files.
