#!/bin/bash
#SBATCH -A AST20015
#SBATCH -p nvdimm
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=rd-cat-147
#SBATCH --output=%x-%j.out
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yueying.ni@cfa.harvard.edu
#SBATCH --dependency=6318208

hostname; pwd; date

src='/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_147'
target='/home1/06431/yueyingn/scratch3/asterix/subfind-idx/subfind_147'
subgrpname='Subfind-SubGrpIndex2'
photodir='/home1/06431/yueyingn/scratch3/asterix/photometric/PIG_147'

ROOT='/home1/06431/yueyingn/work2/ASTRID-II/subfind/scripts/'
python $ROOT/reduce-galaxy-halo-jwst.py --src "$src" --dest "$target" --subgrpname "$subgrpname" --photodir "$photodir"
