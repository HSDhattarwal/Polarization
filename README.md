# Polarization

SCFNN model outputs the atomic and Wannier center (WC) positions in a MD simulations. These positions can used to determine the net polarization in the system. This code takes separate trajectory files for atomic and WC coordinates. Since there are no molecules defined in the machine learning based simulations, a separate function determines molecules and corresponding WCs on the basis of close contacts. 
