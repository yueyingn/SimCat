import numpy as np
import fsps
import h5py
import os, sys

Zsolar = 0.012
photo_bands = ['jwst_f090w','jwst_f150w','jwst_f277w','jwst_f444w']
redshift = float(sys.argv[1])

nZ = 20
nAge = 200
LogMetallicity_bins = np.linspace(-4,-1,nZ)
LogAgeInGyr_bins = np.linspace(-3.8,1.15,nAge)

#-------------------------------------------------------
sp = fsps.StellarPopulation(imf_type=1, zcontinuous=1)   # IMF type 1: Chabrier type
print ("library:",sp.libraries)
sp.params['zred'] = redshift

mags = []
for i in range(0,len(LogMetallicity_bins)):
    print (i)
    sp.params['logzsol'] = np.log10(10**(LogMetallicity_bins[i])/Zsolar)
    
    mband_age = []
    for x in LogAgeInGyr_bins:
        mband_age.append(sp.get_mags(tage=10**x,bands=photo_bands,redshift=redshift))
    mband_age = np.array(mband_age)
    
    mags.append(mband_age)
mags = np.array(mags)

print ("mag.shape:",mags.shape)

# write output
with h5py.File('jwst_f090_f150_f277_f444_photometrics_z%.1f.hdf5'%redshift, 'w') as f:
    f['LogMetallicity_bins'] = LogMetallicity_bins
    f['LogAgeInGyr_bins'] = LogAgeInGyr_bins
    for i in range(0,len(photo_bands)):
        band_name = photo_bands[i]
        f[band_name] = mags[:,:,i]




        
