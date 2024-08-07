"""
example:

basePath='/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/L75n1820TNG/output/'

snapNum=21 
FileLgrid='/mnt/home/yni/ceph/jwst-TNG/mygrid-jwst/jwst_f090_f150_f277_f444_photometrics_z4.0.hdf5'
python reduce-TNG-galaxy-halo-jwst-cat.py --basePath "$basePath" --snapNum $snapNum --FileLgrid "$FileLgrid"


"""
import numpy as np
import argparse
import scipy.interpolate as interpolate
from scipy import integrate
from scipy import ndimage
import h5py
import os,sys
import glob
import illustris_python as il

#--------------------------
parser = argparse.ArgumentParser(description='reduce-TNG-catalog')
parser.add_argument('--basePath',required=True,type=str,help='path of snapshot directory')
parser.add_argument('--snapNum',required=True,type=int,help='snapshot number')
parser.add_argument('--FileLgrid',required=True,type=str,help='path of the jwst mag table at the given redshift')
args = parser.parse_args()
#--------------------------

basePath = args.basePath
snapNum = int(args.snapNum)
FileLgrid = args.FileLgrid
L_grid = h5py.File(FileLgrid,'r')

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


# load the redshift to get the star age function
snap_file = glob.glob(basePath+'snapdir_%03d'%snapNum+'/*.hdf5')[0]
redshift = 1./h5py.File(snap_file,'r')['Header'].attrs[u'Time']-1
print ("z = %.3f"%redshift)
a_cur=1./(1.+redshift)
SFT_space=np.linspace(0,a_cur,100)
age_space=[age(SFT,a_cur) for SFT in SFT_space]
gen_SFT_to_age=interpolate.interp1d(SFT_space,age_space,fill_value='extrapolate')

def get_mags_subhalo(f4,Lgrid):
    """
    f4: subhalo catalog with information of sft,mass,metallicity
    return mags in 4 filters
    """
    Z4 = f4['GFM_Metallicity']
    sft4 = f4['GFM_StellarFormationTime']
    M4 = f4['Masses']*1e10/hh # to Msun
    Age4 = gen_SFT_to_age(sft4)/1e9 # to Gyr
    Age4[Age4<0] = 1e-9
    
    p = {'LogMetallicity_bins': np.log10(Z4),'LogAgeInGyr_bins': np.log10(Age4)}
    coordinates = [[np.interp(p[parameter], L_grid[parameter], range(len(L_grid[parameter])))]
                   for parameter in ['LogMetallicity_bins','LogAgeInGyr_bins']]
    
    mags = []
    for i,band in enumerate(['jwst_f090w','jwst_f150w','jwst_f277w','jwst_f444w']):
        m1 = ndimage.map_coordinates(L_grid[band], coordinates, order=1)[0] # interpolate grid
        # mag for each star particles
        mag4 = m1 - 2.5*np.log10(M4) # lum = -2.5log(lum_per_mass*Mass)
        lumsum = np.sum(10**(-0.4*mag4)) # lum for the entire galaxy
        mags.append(-2.5*np.log10(lumsum)) # mag for the entire galaxy
        
    return np.array(mags)


# load data
#--------------------------
fields_subhalo = ['SubhaloMassType','SubhaloGrNr','SubhaloHalfmassRadType',
                  'SubhaloPos','SubhaloVel','SubhaloSFR']
SubCat = il.groupcat.loadSubhalos(basePath,snapNum,fields=fields_subhalo)

# select out >1e8 galaxies
m4 = SubCat['SubhaloMassType'][:,4]*1e10/hh
sidxs = np.argwhere(m4>1e8)[:,0] # subhalo idx of galaxies with m4>1e8
print ("num galaxies with m4>1e8:",len(sidxs))

# get the host halo information
FOFidxs = SubCat['SubhaloGrNr'][sidxs] # idx of FOF group
fields_halo = ['GroupMass','GroupFirstSub']
HaloCat = il.groupcat.loadHalos(basePath,snapNum,fields=fields_halo)
hm = HaloCat['GroupMass'][FOFidxs]*1e10/hh
FirstSub = HaloCat['GroupFirstSub'][FOFidxs]
sgpIDs = sidxs - FirstSub

# get jwst color magnitudes for each galaxy
field4 = ['Masses','GFM_StellarFormationTime','GFM_Metallicity']

f090s=[]
f150s=[]
f277s=[]
f444s=[]

for i in range(0,len(sidxs)):
    if i%1000==0:
        print (i,end=" ",flush=True)
    sub_id = sidxs[i]
    stars = il.snapshot.loadSubhalo(basePath,snapNum,sub_id,'stars',field4)
    mags = get_mags_subhalo(stars,L_grid)
    f090s.append(mags[0])
    f150s.append(mags[1])
    f277s.append(mags[2])
    f444s.append(mags[3])

#--------------------------
data = np.zeros(len(sidxs),dtype={'names':('GalaxyMass','GalaxyPos','GalaxyVel',
                                           'GalaxyRhalf','SFR','HaloMass',
                                           'jwst_f090w','jwst_f150w','jwst_f277w','jwst_f444w',
                                           'FOFID','sgpID'),
                                    'formats':('d','3d','3d',
                                               'd','d','d',
                                               'd','d','d','d',
                                               'i','i')})

data['GalaxyMass'] = m4[sidxs]
data['GalaxyPos'] = SubCat['SubhaloPos'][sidxs]
data['GalaxyVel'] = SubCat['SubhaloVel'][sidxs]
data['GalaxyRhalf'] = SubCat['SubhaloHalfmassRadType'][:,4][sidxs]
data['SFR'] = SubCat['SubhaloSFR'][sidxs]
data['HaloMass'] = hm
data['jwst_f090w'] = np.array(f090s)
data['jwst_f150w'] = np.array(f150s)
data['jwst_f277w'] = np.array(f277s)
data['jwst_f444w'] = np.array(f444s)
data['FOFID'] = np.array(FOFidxs)
data['sgpID'] = np.array(sgpIDs)


ofilename = 'TNG100_galaxy_halo_catalog'+'_'+'z%d'%(round(redshift))
np.save(ofilename,data)

