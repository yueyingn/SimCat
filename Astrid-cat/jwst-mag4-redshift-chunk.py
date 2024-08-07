"""
use fsps made table to get wide band filter magnitudes for star particles
Here we prepare the grid with redshifts
src is orginal PIG file
dest is target directory
write 4/jwst_mag to dest

example: 

grid='/work2/06431/yueyingn/ASTRID-II/4-galaxy-color/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z4.0.hdf5'
src='/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_147'
target='/home1/06431/yueyingn/scratch3/asterix/photometric/PIG_147'

for ((i=0;i<20;i++)); do
    srun -N 1 -n 1 python $ROOT/jwst-mag4-redshift-chunk.py $grid $src $target 20 $i &> $i-mag.out &
done
wait
date

"""

import numpy as np
from bigfile import BigFile
import scipy.interpolate as interpolate
from scipy import integrate
from scipy import ndimage
import h5py
import sys


h=hh=0.6774

def age(a0,a1,omm=0.3089,oml=0.6911):
    """
    return age between a0 and a1 in yrs
    """
    Mpc_to_km = 3.086e+19
    sec_to_yr = 3.17098e-8
    f = lambda a : 1.0/(a*(np.sqrt(omm*(a)**(-3)+oml)))
    t = 1./(hh*100.0)*integrate.quad(f,a0,a1)[0]
    t *= Mpc_to_km*sec_to_yr
    return t

def get_mags(pig,start,end,L_grid):
    Header = BigFile(pig).open('Header')
    redshift = 1./Header.attrs['Time'][0] - 1
    a_cur=1./(1.+redshift)
    SFT_space=np.linspace(0,a_cur,100)
    age_space=[age(SFT,a_cur) for SFT in SFT_space]
    gen_SFT_to_age=interpolate.interp1d(SFT_space,age_space,fill_value='extrapolate')

    Z4=BigFile(pig).open('4/Metallicity')[start:end]
    sft4=BigFile(pig).open('4/StarFormationTime')[start:end]
    Age4 = gen_SFT_to_age(sft4)/1e9 # to Gyr
    Age4[Age4<0] = 1e-9
    M4 = BigFile(pig).open('4/Mass')[start:end]*10**10/hh # to solar
    
    p = {'LogMetallicity_bins': np.log10(Z4),'LogAgeInGyr_bins': np.log10(Age4)}
    coordinates = [[np.interp(p[parameter], L_grid[parameter], range(len(L_grid[parameter])))] 
                   for parameter in ['LogMetallicity_bins','LogAgeInGyr_bins']]

    mag = np.zeros([len(M4),4])
    for i,band in enumerate(['jwst_f090w','jwst_f150w','jwst_f277w','jwst_f444w']):
        m1 = ndimage.map_coordinates(L_grid[band], coordinates, order=1)[0] # interpolate grid
        mag[:,i] = m1 - 2.5*np.log10(M4) # lum = -2.5log(lum_per_mass*Mass)
        
    return mag

#--------------------------------------
def main():
    import sys
    from bigfile import BigFile
    
    L_grid = h5py.File(sys.argv[1],'r')
    src = sys.argv[2]
    dest = sys.argv[3]
    Njob = int(sys.argv[4])
    partition = int(sys.argv[5])

    Nstar = BigFile(src).open('Header').attrs['NumPartInGroupTotal'][4]
    print ("Nstar",Nstar)
    Nchunk = int(Nstar/Njob)
    print("Nchunk",Nchunk)

    dest = BigFile(dest)
    try:
        dest["4/jwst_mag"]
    except:
        starz = BigFile(src)['4/Metallicity']
        dest.create("4/jwst_mag", size=starz.size, dtype=np.dtype('4f'), Nfile=starz.Nfile)

    i = partition
    start = int(i*Nchunk)
    if i==Njob-1:
        end = Nstar
    else:
        end = int(i*Nchunk+Nchunk)

    stellar_mag = get_mags(src,start,end,L_grid)
    dest['4/jwst_mag'].write(start, stellar_mag)
    print ('start->end',start,end,'done',flush=True)


if __name__ == '__main__':
    main()


