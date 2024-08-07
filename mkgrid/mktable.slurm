#!/bin/bash
#SBATCH -A AST20015
#SBATCH -p development
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=fsps
#SBATCH --time=2:00:00

hostname; pwd; date

redshift=3
python create_jwst_filter_table.py $redshift

redshift=4
python create_jwst_filter_table.py $redshift

redshift=5
python create_jwst_filter_table.py $redshift

redshift=6
python create_jwst_filter_table.py $redshift

redshift=7
python create_jwst_filter_table.py $redshift
