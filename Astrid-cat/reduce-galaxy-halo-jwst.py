import numpy as np
from bigfile import BigFile
import argparse
import os
import sys

#--------------------------
parser = argparse.ArgumentParser(description='reduce-by-subgrp')
parser.add_argument('--src',required=True,type=str,help='path of the PIG file directory')
parser.add_argument('--dest',required=True,type=str,help='path of the subcolumn file directory')
parser.add_argument('--subgrpname',required=True,type=str,help='column name for subgrp reduce')
parser.add_argument('--photodir',required=True,type=str,help='column name for photometrics')

args = parser.parse_args()
#--------------------------

num4_thr = 100 

snap = str(args.src[-3:])
src = BigFile(args.src)
dest = BigFile(args.dest)

hh = src.open('Header').attrs['HubbleParam'][0]
BoxSize = src.open('Header').attrs['BoxSize'][0]

LengthByType = src.open('FOFGroups/LengthByType')[:]
OffsetByType = src.open('FOFGroups/OffsetByType')[:]

gas_off = OffsetByType[:,0]
gas_len = LengthByType[:,0]
star_off = OffsetByType[:,4]
star_len = LengthByType[:,4]
haloMass = src.open('FOFGroups/Mass')[:]*10**10/hh

del LengthByType 
del OffsetByType


subgrpcolumn0 = '0/'+str(args.subgrpname)

subgrpcolumn4 = '4/'+str(args.subgrpname)
SubGrpID4 = dest.open(subgrpcolumn4)[:star_off[-1]]


mag4 = np.float32(BigFile(str(args.photodir))['4/jwst_mag'][:star_off[-1]])


M4 = src.open('4/Mass')[:]*10**10/hh # in Msun unit
pos4 = np.float32(src.open('4/Position')[:])
vel4 = np.float32(src.open('4/Velocity')[:])

# make approximation for number of halo that counts
fof_num = len(star_len[star_len>5])
print ("Number of FOF Groups with stellar particle:",fof_num)

gas_end = gas_off[fof_num]
print ("chunk gas at",gas_end)
sfr0 = src.open('0/StarFormationRate')[0:gas_end] # 0/sfr is already in unit of Msun/yr
SubGrpID0 = dest.open(subgrpcolumn0)[0:gas_end]


def get_galaxy_mag(m):
    lumsum = np.sum(10**(-0.4*m))
    return -2.5*np.log10(lumsum)

def get_galaxy_info(mass,pos,vel):
    """
    return the mass center position, velocity, half mass radius
    given periodic boundary condition
    mass: (N,)
    pos: (N,3)
    vel: (N,3)
    """
    ref = pos[0]
    dr = pos - ref
    dr[dr>0.5*BoxSize] -= BoxSize
    dr[dr<=-0.5*BoxSize] += BoxSize
    
    # mass weighted pos and vel
    # -------------------------------------------
    total_mass = np.sum(mass)
    
    weighted_positions = mass[:, np.newaxis] * dr
    mcpos = np.sum(weighted_positions, axis=0) / total_mass
    mcpos += ref
    
    weighted_velocities = mass[:, np.newaxis] * vel
    mcvel = np.sum(weighted_velocities, axis=0) / total_mass
    
    # half mass radius
    # -------------------------------------------
    r3d = np.linalg.norm(pos-mcpos,axis=-1)
    sorted_indices = np.argsort(r3d)
    sorted_mass = mass[sorted_indices]
    sorted_r3d = r3d[sorted_indices]
    
    cumulative_mass = np.cumsum(sorted_mass)
    # Find the index where cumulative mass exceeds half the total mass
    half_mass_index = np.argmax(cumulative_mass >= np.sum(mass) / 2)
    # Calculate half-mass radius
    half_mass_radius = sorted_r3d[half_mass_index]
    
    return mcpos,mcvel,half_mass_radius

    
# reduce quantity for subhalos
# ----------------------------
gms=[]
galaxypos=[]
galaxyvel=[]
sfrs=[]
rhalfs=[]
fofIDs=[]
sgpIDs=[]
hms=[]
f090s=[]
f150s=[]
f277s=[]
f444s=[]


for fidx in range(fof_num):
    if fidx%10000==0:
        print (fidx,flush=True)

    if (star_len[fidx]<num4_thr): # we only reduce galaxy with at least num4_thr particles
        continue
        
    start0,end0 = gas_off[fidx],gas_off[fidx+1]
    start4,end4 = star_off[fidx],star_off[fidx+1]
    
    sgrpID0 = SubGrpID0[start0:end0]
    sgrpID4 = SubGrpID4[start4:end4]
    
    gassfr = sfr0[start0:end0]
    starm = M4[start4:end4]
    starpos = pos4[start4:end4]
    starvel = vel4[start4:end4]
    hm = haloMass[fidx]
    starmag = mag4[start4:end4]

    for sid in np.unique(sgrpID4):
        if sid>0:
            mask4 = (sgrpID4==sid)
            if (len(starm[mask4])>num4_thr):
                mask0 = sgrpID0==sid
                sfrs.append(np.sum(gassfr[mask0]))
                gms.append(np.sum(starm[mask4]))
                mcpos,mcvel,half_mass_radius = get_galaxy_info(starm[mask4],starpos[mask4],starvel[mask4])
                galaxypos.append(mcpos)
                galaxyvel.append(mcvel)
                rhalfs.append(half_mass_radius)
                fofIDs.append(fidx)
                sgpIDs.append(sid)
                hms.append(hm)
                f090s.append(get_galaxy_mag(starmag[mask4][:,0]))
                f150s.append(get_galaxy_mag(starmag[mask4][:,1]))
                f277s.append(get_galaxy_mag(starmag[mask4][:,2]))
                f444s.append(get_galaxy_mag(starmag[mask4][:,3]))
                
                    
# output
# ----------------------------
data = np.zeros(len(gms),dtype={'names':('GalaxyMass','GalaxyPos','GalaxyVel',
                                         'GalaxyRhalf','SFR','HaloMass',
                                         'jwst_f090w','jwst_f150w','jwst_f277w','jwst_f444w',
                                         'FOFID','sgpID'),
                                'formats':('d','3d','3d',
                                           'd','d','d',
                                           'd','d','d','d',
                                           'i','i')})

data['GalaxyMass'] = np.array(gms)
data['GalaxyPos'] = np.array(galaxypos)
data['GalaxyVel'] = np.array(galaxyvel)
data['GalaxyRhalf'] = np.array(rhalfs)
data['SFR'] = np.array(sfrs)
data['HaloMass'] = np.array(hms)
data['jwst_f090w'] = np.array(f090s)
data['jwst_f150w'] = np.array(f150s)
data['jwst_f277w'] = np.array(f277s)
data['jwst_f444w'] = np.array(f444s)
data['FOFID'] = np.array(fofIDs)
data['sgpID'] = np.array(sgpIDs)

ofilename = 'ASTRID_galaxy_halo_catalog'+'_'+snap
np.save(ofilename,data)
