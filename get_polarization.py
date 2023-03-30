# %%
import numpy as np
from numpy.linalg import norm
#import  matplotlib.pyplot as plt
import pandas as pd
def get_neighbour_rij_nw(nxyz, wcxyz, boxlength, nneighbour):
    nxyz_expand = np.expand_dims(nxyz, axis=1)  # expand the j indices. Now dim1 is natoms, dim2 is 1, dim3 is xyz
    wcxyz_expand = np.expand_dims(wcxyz, axis=0)  # expand the i indices. Now dim1 is 1, dim2 is nwannier, dim3 is xyz
    vecrij = wcxyz_expand - nxyz_expand - np.round((wcxyz_expand - nxyz_expand) / boxlength) * boxlength  # dim1 is natoms, dim2 is nwanniers, dim3 is xyz
    rij = np.linalg.norm(vecrij, axis=2)

    rij_indmin = np.argsort(rij, axis=1)
    i_indexarray = np.expand_dims(np.arange(rij.shape[0]), axis=1)

    i_indexarray2 = np.expand_dims(np.arange(rij.shape[0]), axis=(1, 2))
    ixyz_indexarray2 = np.expand_dims(np.arange(3), axis=(0, 1))
    rij_indmin2 = np.expand_dims(rij_indmin, axis=2)

    rij_min = rij[i_indexarray, rij_indmin]
    vecrij_min = vecrij[i_indexarray2, rij_indmin2, ixyz_indexarray2]

    return rij_indmin[:, 0:nneighbour], rij_min[:, 0:nneighbour], vecrij_min[:, 0:nneighbour, :]

# %%
# %% 
traj = np.loadtxt("xyz_hist.txt")
w_traj = np.loadtxt("wxyz_hist.txt")
# %%
noxygen = 1000
natoms = 3000
nwannier = 4000
Lx = traj[0, 1]
ang_unit = 0.529177
qO = 6
qH = 1
qw = -2
debye_unit = 0.3934303
d_spce = 2.351*debye_unit
d_e0 = 2.900*debye_unit
nframes = 1000
nshift = 500
isteps = np.arange(nshift, nframes, 1)
# %% 

dipoles_mag_all = ()
dipoles_mag_all_spce = ()
dipoles_mag_all_e0 = ()
for istep in isteps:
    nxyz = traj[((natoms+3) * (istep - 1)+3): ((natoms+3)*istep), : ]
    Oxyz = nxyz[:noxygen, :]
    Hxyz = nxyz[noxygen:,]
    wxyz = w_traj[((nwannier+1) * (istep - 1)+1): ((nwannier+1)*istep), : ]
    data_Ow = get_neighbour_rij_nw(Oxyz, wxyz, Lx, 4)
    data_OH = get_neighbour_rij_nw(Oxyz, Hxyz, Lx, 2)
    
    dipoles = qw * np.sum(data_Ow[2], axis=1) + qH * np.sum(data_OH[2], axis=1)
    dipoles_mag = np.linalg.norm(dipoles, axis = -1)
    dipoles_spce = d_spce*dipoles / np.hstack((dipoles_mag.reshape(noxygen,1),)*3)
    dipoles_e0 = d_e0*dipoles / np.hstack((dipoles_mag.reshape(noxygen,1),)*3)
    #dipoles_e0 = dipoles / np.hstack((np.full((noxygen,1), d_e0),)*3)
    dipoles_mag_all += (dipoles,)
    dipoles_mag_all_spce += (dipoles_spce,)
    dipoles_mag_all_e0 += (dipoles_e0,)
# %%
dipoles_mag_allconfig = np.concatenate(dipoles_mag_all,axis=0)
dipoles_mag_allconfig_spce = np.concatenate(dipoles_mag_all_spce,axis=0)
dipoles_mag_allconfig_e0 = np.concatenate(dipoles_mag_all_e0,axis=0)

dipoles_mag_allconfig_D = dipoles_mag_allconfig * ang_unit
dipoles_mag_allconfig_D_spce = dipoles_mag_allconfig_spce * ang_unit
dipoles_mag_allconfig_D_e0 = dipoles_mag_allconfig_e0 * ang_unit
Lx = Lx * ang_unit

polarization_mag = (1000*np.sum(dipoles_mag_allconfig_D, axis=0)) / (Lx*Lx*Lx*(nframes-nshift+1))
polarization_mag_spce = (1000*np.sum(dipoles_mag_allconfig_D_spce, axis=0)) / (Lx*Lx*Lx*(nframes-nshift+1))
polarization_mag_e0 = (1000*np.sum(dipoles_mag_allconfig_D_e0, axis=0)) / (Lx*Lx*Lx*(nframes-nshift+1))
polarization_mag = np.linalg.norm(polarization_mag, axis=-1)
polarization_mag_spce = np.linalg.norm(polarization_mag_spce, axis=-1)
polarization_mag_e0 = np.linalg.norm(polarization_mag_e0, axis=-1)

fnew =open('polarization_SCFNN.txt', 'a')
np.savetxt(fnew, (polarization_mag,polarization_mag_spce,polarization_mag_e0))
fnew.close()
