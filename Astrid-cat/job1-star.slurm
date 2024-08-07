#!/bin/bash
#SBATCH -A AST20015
#SBATCH -p development
#SBATCH -N 5
#SBATCH --ntasks-per-node 4
#SBATCH --job-name=mag-147
#SBATCH --output=%x-%j.out
#SBATCH --time=00:30:00

hostname; pwd; date
ROOT='/home1/06431/yueyingn/work2/ASTRID-II/4-galaxy-color/scripts'

grid='/work2/06431/yueyingn/ASTRID-II/4-galaxy-color/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z4.0.hdf5'
src='/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_147'
target='/home1/06431/yueyingn/scratch3/asterix/photometric/PIG_147'

for ((i=0;i<20;i++)); do
    srun -N 1 -n 1 python $ROOT/jwst-mag4-redshift-chunk.py $grid $src $target 20 $i &> $i-mag.out &
done
wait
date


