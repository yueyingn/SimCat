#!/bin/bash
#SBATCH --partition=cmbas
#SBATCH --job-name=cat
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

hostname; pwd; date

basePath='/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/L75n1820TNG/output/'

snapNum=13
FileLgrid='/mnt/home/yni/ceph/jwst-TNG/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z6.0.hdf5'
python reduce-TNG-galaxy-halo-jwst-cat.py --basePath "$basePath" --snapNum $snapNum --FileLgrid "$FileLgrid"

snapNum=17
FileLgrid='/mnt/home/yni/ceph/jwst-TNG/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z5.0.hdf5'
python reduce-TNG-galaxy-halo-jwst-cat.py --basePath "$basePath" --snapNum $snapNum --FileLgrid "$FileLgrid"

snapNum=21 
FileLgrid='/mnt/home/yni/ceph/jwst-TNG/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z4.0.hdf5'
python reduce-TNG-galaxy-halo-jwst-cat.py --basePath "$basePath" --snapNum $snapNum --FileLgrid "$FileLgrid"

snapNum=25
FileLgrid='/mnt/home/yni/ceph/jwst-TNG/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z3.0.hdf5'
python reduce-TNG-galaxy-halo-jwst-cat.py --basePath "$basePath" --snapNum $snapNum --FileLgrid "$FileLgrid"
