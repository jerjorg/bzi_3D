"""Construct pseudopotentials for various elements. This approach is taken from
`Grosso <http://www.sciencedirect.com/science/article/pii/B9780123850300000050>`_.
"""

import itertools
import numpy as np
from numpy.linalg import norm
from BZI.symmetry import shells, make_ptvecs, make_rptvecs, Lattice
from BZI.sampling import sphere_pts

# Conversions
angstrom_to_Bohr = 1.889725989
Ry_to_eV = 13.605698066

class EmpiricalPP(object):
    """Create an empirical pseudopotential.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        form_factors (list): a list of pseudopotential form factors. Every 
            energy shell up to the cutoff energy should be accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.  
        atomic_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy (float): the total energy.

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        form_factors (list): a list of pseudopotential form factors. Every 
            energy shell up to the cutoff energy should be accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.  
        rlat_pts (list): a list of reciprocal lattice points included in the 
            Fourier expansion.
        energy_shells (list): a list of spherical shells that points in rlat_pts
            reside on.
        atomic_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        total_energy (float): the total energy.

    Example:
        >>> centering_type = "face"
        >>> lat_const = 4.0
        >>> lat_consts = [lat_const]*3
        >>> lat_angles = [np.pi/2]*3
        >>> lattice = Lattice(centering_type, lat_consts, lat_angles)
        >>> pff = [0.019, 0.055]
        >>> cutoff = 4*(2*np.pi/lat_const)**2
        >>> atomic_positions = [0.,0.,0.]
        >>> nvalence_electrons = 3
        >>> PP = EmpiricalPP(lattice, pff, cutoff, nvalence_electrons,
        >>>                  atomic_positions)
    """
    
    def __init__(self, lattice, form_factors, energy_cutoff, atomic_positions,
                 nvalence_electrons, energy_shift=None, fermi_level=None):
        self.lattice = lattice
        self.form_factors = form_factors
        self.energy_cutoff = energy_cutoff
        if np.shape(atomic_positions) == (3,):
            msg = ("Please provide a list of atomic positions instead of an "
                   "individual atomic position.")
            raise ValueError(msg.format(atomic_positions))
        else:
            self.atomic_positions = atomic_positions
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                   self.energy_cutoff)
        self.find_energy_shells()
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.

    def find_energy_shells(self):
        """Find the spherical shells of constant energy on which the points in
        rlat_pts reside. These are ordered from least to greatest.
        """

        shells = []
        for rpt in self.rlat_pts:
            if any(np.isclose(np.dot(rpt, rpt), shells)):
                continue
            else:
                shells.append(np.dot(rpt, rpt))
        self.energy_shells = np.sort(shells)
                
    def eval(self, kpoint, neigvals):
        """Evaluate the empirical pseudopotential eigenvalues at the provided
        k-point. Only return the lowest 'neigvals' eigenvalues.
        """
        
        size = len(self.rlat_pts)
        H = np.zeros([size, size], dtype=complex)
        for i,rpt1 in enumerate(self.rlat_pts):
            for j in range(i+1):
                rpt2 = self.rlat_pts[j]
                h = rpt2 - rpt1
                r2 = np.dot(h,h)
                for ap in self.atomic_positions:
                    if i == j:
                        H[i,j] = np.dot(kpoint + rpt1, kpoint + rpt1)
                        break
                    else:
                        try:
                            k = np.where(np.isclose(self.energy_shells, r2))[0][0]
                            H[i,j] += self.form_factors[k]*np.exp(-1j*np.dot(h,ap))
                        except:
                            continue
        return np.sort(np.linalg.eigvalsh(H))[:neigvals]*Ry_to_eV

    def hamiltonian(self, kpoint):
        """Evaluate the empirical pseudopotential Hamiltonian at the provided
        k-point. This function is typically used to verify the Hamiltonian is 
        Hermitian.
        """
        
        size = len(self.rlat_pts)
        H = np.zeros([size, size], dtype=complex)
        for i,rpt1 in enumerate(self.rlat_pts):
            for j,rpt2 in enumerate(self.rlat_pts):
                h = rpt2 - rpt1
                r2 = np.dot(h,h)
                for ap in self.atomic_positions:
                    if i == j:
                        H[i,j] = np.dot(kpoint + rpt1, kpoint + rpt1)
                        break
                    else:
                        try:
                            k = np.where(np.isclose(self.energy_shells, r2))[0][0]
                            H[i,j] = self.form_factors[k]*np.exp(-1j*np.dot(h,ap))
                        except:
                            continue
        return H*Ry_to_eV

class CohenEmpiricalPP(object):
    """Create an empirical pseudopotential after Cohen's derivation.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        sym_form_factors (list): a list of symmetric, pseudopotential form 
            factors. Every energy shell up to the cutoff energy should be
            accounted for.
        antisym_form_factors (list): a list of anti-symmetric, pseudopotential
            form factors. Every energy shell up to the cutoff energy should be 
            accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.  
        atomic_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy (float): the total energy.

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        sym_form_factors (list): a list of symmetric, pseudopotential form 
            factors. Every energy shell up to the cutoff energy should be
            accounted for.
        antisym_form_factors (list): a list of anti-symmetric, pseudopotential
            form factors. Every energy shell up to the cutoff energy should be 
            accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.  
        rlat_pts (list): a list of reciprocal lattice points included in the 
            Fourier expansion.
        energy_shells (list): a list of spherical shells that points in rlat_pts
            reside on.
        atomic_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy (float): the total energy.
    """

    def __init__(self, lattice, sym_form_factors, antisym_form_factors,
                 energy_cutoff, atomic_positions, nvalence_electrons,
                 energy_shift=None, fermi_level=None):
        
        self.lattice = lattice
        self.sym_form_factors = sym_form_factors
        self.antisym_form_factors = antisym_form_factors
        self.energy_cutoff = energy_cutoff
        if np.shape(atomic_positions) == (3,):
            msg = ("Please provide a list of atomic positions instead of an "
                   "individual atomic position.")
            raise ValueError(msg.format(atomic_positions))
        else:
            self.atomic_positions = atomic_positions
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                    self.energy_cutoff)
        self.find_energy_shells()
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.

    def find_energy_shells(self):
        """Find the spherical shells of constant energy on which the points in
        rlat_pts reside. These are ordered from least to greatest.
        """

        shells = []
        for rpt in self.rlat_pts:
            if any(np.isclose(np.dot(rpt, rpt), shells)):
                continue
            else:
                shells.append(np.dot(rpt, rpt))
        self.energy_shells = np.sort(shells)
                    
    def eval(self, kpoint, neigvals):
        """Evaluate the empirical pseudopotential eigenvalues at the provided
        k-point. Only return the lowest 'neigvals' eigenvalues.
        """
        
        size = len(self.rlat_pts)
        H = np.zeros([size, size], dtype=complex)
        for i,rpt1 in enumerate(self.rlat_pts):
            for j in range(i+1):
                rpt2 = self.rlat_pts[j]
                h = rpt2 - rpt1
                r2 = np.dot(h,h)
                for ap in self.atomic_positions:
                    if i == j:
                        H[i,j] = np.dot(kpoint + rpt1, kpoint + rpt1)
                        break
                    else:
                        try:
                            k = np.where(np.isclose(self.energy_shells, r2))[0][0]
                            H[i,j] = self.sym_form_factors[k]*(
                                np.cos(np.dot(h,ap))) + (
                                    1j*self.antisym_form_factors[k]*(
                                        np.sin(np.dot(h,ap))))
                        except:
                            continue
        return np.sort(np.linalg.eigvalsh(H))[:neigvals]*Ry_to_eV

    def hamiltonian(self, kpoint):
        """Evaluate the empirical pseudopotential Hamiltonian at the provided
        k-point. This function is typically used to verify the Hamiltonian is 
        Hermitian.
        """
        
        size = len(self.rlat_pts)
        H = np.zeros([size, size], dtype=complex)
        for i,rpt1 in enumerate(self.rlat_pts):
            for j,rpt2 in enumerate(self.rlat_pts):
                h = rpt2 - rpt1
                r2 = np.dot(h,h)
                for ap in self.atomic_positions:
                    if i == j:
                        H[i,j] = np.dot(kpoint + rpt1, kpoint + rpt1)
                        break
                    else:
                        try:
                            k = np.where(np.isclose(self.energy_shells, r2))[0][0]
                            H[i,j] = self.sym_form_factors[k]*(
                                np.cos(np.dot(h, ap))) + (
                                    1j*self.antisym_form_factors[k]*
                                    np.sin(np.dot(h, ap)))
                        except:
                            continue
        return H*Ry_to_eV

class FreeElectronModel():
    """This is the popular free electron model. In this model the potential is
    zero everywhere. It is useful for testing.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
    """
    
    def __init__(self, lattice, nvalence_electrons, degree, energy_shift=None,
                 fermi_level=None):
        self.lattice = lattice
        self.degree = degree
        if nvalence_electrons > 2:
            msg = ("The free electron model can only have two valuenc electrons"
                   " since there is only one band.")
            raise ValueError(msg.format(nvalence_electrons))
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.

    def eval(self, x, neigvals):
        # There's only one eigenvalue so neigvals isn't needed in general but
        # it is when running tests on functions that take an instance of the
        # pseudopotential classes.
        return [np.linalg.norm(x)**self.degree]
    
#### W pseudopotentials ####
def W1(spt):
    """W1 is another toy model that we often work with. It is also convenient
    for unit testing since integrating it can be performed analytically.
    
    Args:
        spt (list or numpy.ndarray): a sampling point
    """
    return [np.product(np.exp([np.cos(2*np.pi*pt) for pt in spt]))]

def W2(spt):
    """W2 is another toy model. It is also convenient for unit testing since 
    integrating it can be performed analytically.
    
    Args:
        spt (list or numpy.ndarray): a sampling point
    """
    
    return[-np.cos(np.sum([np.cos(2*np.pi*pt) for pt in spt]))]

#### 
# Pseudo-potential from Band Structures and Pseudopotential Form Factors for
# Fourteen Semiconductors of the Diamond. and. Zinc-blende Structures* by
# Cohen and Bergstresser
#### 

# Define the pseudo-potential form factors taken from Cohen and Bergstesser
# V3S stands for the symmetric form factor for reciprocal lattice vectors of
# squared magnitude 3.
# V4A stands for the antisymmetric form factor for reciprocal lattice vectors
# of squared magnitude 4.
# These are assigned to new variables below and never get referenced.
#            V3S   V8S   V11S  V3A   V4A   V11A
Si_pff =   [-0.21, 0.04, 0.08, 0.00, 0.00, 0.00]
Ge_pff =   [-0.23, 0.01, 0.06, 0.00, 0.00, 0.00]
Sn_pff =   [-0.20, 0.00, 0.04, 0.00, 0.00, 0.00]
GaP_pff =  [-0.22, 0.03, 0.07, 0.12, 0.07, 0.02]
GaAs_pff = [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
AlSb_pff = [-0.21, 0.02, 0.06, 0.06, 0.04, 0.02]
InP_pff =  [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
GaSb_pff = [-0.22, 0.00, 0.05, 0.06, 0.05, 0.01]
InAs_pff = [-0.22, 0.00, 0.05, 0.08, 0.05, 0.03]
InSb_pff = [-0.20, 0.00, 0.04, 0.06, 0.05, 0.01]
ZnS_pff =  [-0.22, 0.03, 0.07, 0.24, 0.14, 0.04]
ZnSe_pff = [-0.23, 0.01, 0.06, 0.18, 0.12, 0.03]
ZnTe_pff = [-0.22, 0.00, 0.05, 0.13, 0.10, 0.01]
CdTe_pff = [-0.20, 0.00, 0.04, 0.15, 0.09, 0.04]

# Here the pseudopotential form factors are reorganized to work with the code.
Si_spff =    [0.00, -0.21, 0.00, 0.04, 0.08]
Si_apff =    [0.00, 0.00, 0.00, 0.00, 0.00]
Ge_spff =    [0.00, -0.23, 0.00, 0.01, 0.06]
Ge_apff =    [0.00, 0.00, 0.00, 0.00, 0.00]
Sn_spff =    [0.00, -0.20, 0.00, 0.00, 0.04]
Sn_apff =    [0.00, 0.00, 0.00, 0.00, 0.00]
GaP_spff =   [0.00, -0.22, 0.00, 0.03, 0.07]
GaP_apff =   [0.00, 0.12, 0.07, 0.00, 0.02]
GaAs_spff =  [0.00, -0.23, 0.00, 0.01, 0.06]
GaAs_apff =  [0.00, 0.07, 0.05, 0.00, 0.01]
AlSb_spff =  [0.00, -0.21, 0.00, 0.02, 0.06]
AlSb_apff =  [0.00, 0.06, 0.04, 0.00, 0.02]
InP_spff =   [0.00, -0.23, 0.00, 0.01, 0.06]
InP_apff =   [0.00, 0.07, 0.05, 0.00, 0.01]
GaSb_spff =  [0.00, -0.22, 0.00, 0.00, 0.05]
GaSb_apff =  [0.00, 0.06, 0.05, 0.00, 0.01]
InAs_spff =  [0.00, -0.22, 0.00, 0.00, 0.05]
InAs_apff =  [0.00, 0.08, 0.05, 0.00, 0.03]
InSb_spff =  [0.00, -0.20, 0.00, 0.00, 0.04]
InSb_apff =  [0.00, 0.06, 0.05, 0.00, 0.01]
ZnS_spff =   [0.00, -0.22, 0.00, 0.03, 0.07]
ZnS_apff =   [0.00, 0.24, 0.14, 0.00, 0.04]
ZnSe_spff =  [0.00, -0.23, 0.00, 0.01, 0.06]
ZnSe_apff =  [0.00, 0.18, 0.12, 0.00, 0.03]
ZnTe_spff =  [0.00, -0.22, 0.00, 0.00, 0.05]
ZnTe_apff =  [0.00, 0.13, 0.10, 0.00, 0.01]
CdTe_spff =  [0.00, -0.20, 0.00, 0.00, 0.04]
CdTe_apff =  [0.00, 0.15, 0.09, 0.00, 0.04]    

#### Pseudopotential of Si ####
Si_lat_centering = "face"
Si_lat_const = 5.43*angstrom_to_Bohr # the lattice constant in Bohr
Si_lat_consts = [Si_lat_const]*3
Si_lat_angles = [np.pi/2]*3
Si_lattice = Lattice(Si_lat_centering, Si_lat_consts, Si_lat_angles)

Si_energy_cutoff = 11.*(2*np.pi/Si_lat_const)**2
Si_atomic_positions = [Si_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Si_nvalence_electrons = 4
Si_PP = CohenEmpiricalPP(Si_lattice, Si_spff, Si_apff, Si_energy_cutoff,
                         Si_atomic_positions, Si_nvalence_electrons)

#### Pseudopotential of Ge ####
Ge_lat_centering = "face"
Ge_lat_const = 5.66*angstrom_to_Bohr # the lattice constant in Bohr
Ge_lat_consts = [Ge_lat_const]*3
Ge_lat_angles = [np.pi/2]*3
Ge_lattice = Lattice(Ge_lat_centering, Ge_lat_consts, Ge_lat_angles)

Ge_cutoff_energy = 11.*(2*np.pi/Ge_lat_const)**2
Ge_atomic_positions = [Ge_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Ge_nvalence_electrons = 4
Ge_PP = CohenEmpiricalPP(Ge_lattice, Ge_spff, Ge_apff, Ge_cutoff_energy,
                         Ge_atomic_positions, Ge_nvalence_electrons)

#### Pseudopotential of Sn ####
cSn_lat_centering = "face"
cSn_lat_const = 6.49*angstrom_to_Bohr # the lattice constant in Bohr
cSn_lat_consts = [cSn_lat_const]*3
cSn_lat_angles = [np.pi/2]*3
cSn_lattice = Lattice(cSn_lat_centering, cSn_lat_consts, cSn_lat_angles)

cSn_cutoff_energy = 11.*(2*np.pi/cSn_lat_const)**2
cSn_atomic_positions = [cSn_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
cSn_nvalence_electrons = 4
cSn_PP = CohenEmpiricalPP(cSn_lattice, Sn_spff, Sn_apff, cSn_cutoff_energy,
                          cSn_atomic_positions, cSn_nvalence_electrons)

#### Pseudopotential of GaP ####
GaP_lat_centering = "face"
GaP_lat_const = 5.44*angstrom_to_Bohr # the lattice constant in Bohr
GaP_lat_consts = [GaP_lat_const]*3
GaP_lat_angles = [np.pi/2]*3
GaP_lattice = Lattice(GaP_lat_centering, GaP_lat_consts, GaP_lat_angles)

GaP_cutoff_energy = 11.*(2*np.pi/GaP_lat_const)**2
GaP_atomic_positions = [GaP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaP_nvalence_electrons = 8
GaP_PP = CohenEmpiricalPP(GaP_lattice, GaP_spff, GaP_apff, GaP_cutoff_energy,
                          GaP_atomic_positions, GaP_nvalence_electrons)

#### Pseudopotential of GaAs ####
GaAs_lat_centering = "face"
GaAs_lat_const = 5.64*angstrom_to_Bohr # the lattice constant in Bohr
GaAs_lat_consts = [GaAs_lat_const]*3
GaAs_lat_angles = [np.pi/2]*3
GaAs_lattice = Lattice(GaAs_lat_centering, GaAs_lat_consts, GaAs_lat_angles)

GaAs_cutoff_energy = 11.*(2*np.pi/GaAs_lat_const)**2
GaAs_atomic_positions = [GaAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaAs_nvalence_electrons = 8
GaAs_PP = CohenEmpiricalPP(GaAs_lattice, GaAs_spff, GaAs_apff, GaAs_cutoff_energy,
                           GaAs_atomic_positions, GaAs_nvalence_electrons)

#### Pseudopotential of AlSb ####
AlSb_lat_centering = "face"
AlSb_lat_const = 6.13*angstrom_to_Bohr # the lattice constant in Bohr
AlSb_lat_consts = [AlSb_lat_const]*3
AlSb_lat_angles = [np.pi/2]*3
AlSb_lattice = Lattice(AlSb_lat_centering, AlSb_lat_consts, AlSb_lat_angles)

AlSb_cutoff_energy = 11.*(2*np.pi/AlSb_lat_const)**2
AlSb_atomic_positions = [AlSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
AlSb_nvalence_electrons = 8
AlSb_PP = CohenEmpiricalPP(AlSb_lattice, AlSb_spff, AlSb_apff,
                           AlSb_cutoff_energy, AlSb_atomic_positions,
                           AlSb_nvalence_electrons)

#### Pseudopotential of InP ####
InP_lat_centering = "face"
InP_lat_const = 5.86*angstrom_to_Bohr # the lattice constant in Bohr
InP_lat_consts = [InP_lat_const]*3
InP_lat_angles = [np.pi/2]*3
InP_lattice = Lattice(InP_lat_centering, InP_lat_consts, InP_lat_angles)

InP_cutoff_energy = 11.*(2*np.pi/InP_lat_const)**2
InP_atomic_positions = [InP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InP_nvalence_electrons = 8
InP_PP = CohenEmpiricalPP(InP_lattice, InP_spff, InP_apff, InP_cutoff_energy,
                          InP_atomic_positions, InP_nvalence_electrons)

#### Pseudopotential of GaSb ####
GaSb_lat_centering = "face"
GaSb_lat_const = 6.12*angstrom_to_Bohr # the lattice constant in Bohr
GaSb_lat_consts = [GaSb_lat_const]*3
GaSb_lat_angles = [np.pi/2]*3
GaSb_lattice = Lattice(GaSb_lat_centering, GaSb_lat_consts, GaSb_lat_angles)

GaSb_cutoff_energy = 11.*(2*np.pi/GaSb_lat_const)**2
GaSb_atomic_positions = [GaSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaSb_nvalence_electrons = 8
GaSb_PP = CohenEmpiricalPP(GaSb_lattice, GaSb_spff, GaSb_apff,
                           GaSb_cutoff_energy, GaSb_atomic_positions,
                           GaSb_nvalence_electrons)

#### Pseudopotential of InAs ####
InAs_lat_centering = "face"
InAs_lat_const = 6.04*angstrom_to_Bohr # the lattice constant in Bohr
InAs_lat_consts = [InAs_lat_const]*3
InAs_lat_angles = [np.pi/2]*3
InAs_lattice = Lattice(InAs_lat_centering, InAs_lat_consts, InAs_lat_angles)

InAs_cutoff_energy = 11.*(2*np.pi/InAs_lat_const)**2
InAs_atomic_positions = [InAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InAs_nvalence_electrons = 8
InAs_PP = CohenEmpiricalPP(InAs_lattice, InAs_spff, InAs_apff,
                           InAs_cutoff_energy, InAs_atomic_positions,
                           InAs_nvalence_electrons)

#### Pseudopotential of InSb ####
InSb_lat_centering = "face"
InSb_lat_const = 6.48*angstrom_to_Bohr # the lattice constant in Bohr
InSb_lat_consts = [InSb_lat_const]*3
InSb_lat_angles = [np.pi/2]*3
InSb_lattice = Lattice(InSb_lat_centering, InSb_lat_consts, InSb_lat_angles)

InSb_cutoff_energy = 11.*(2*np.pi/InSb_lat_const)**2
InSb_atomic_positions = [InSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InSb_nvalence_electrons = 8
InSb_PP = CohenEmpiricalPP(InSb_lattice, InSb_spff, InSb_apff,
                           InSb_cutoff_energy, InSb_atomic_positions,
                           InSb_nvalence_electrons)

#### Pseudopotential of ZnS ####
ZnS_lat_centering = "face"
ZnS_lat_const = 5.41*angstrom_to_Bohr # the lattice constant in Bohr
ZnS_lat_consts = [ZnS_lat_const]*3
ZnS_lat_angles = [np.pi/2]*3
ZnS_lattice = Lattice(ZnS_lat_centering, ZnS_lat_consts, ZnS_lat_angles)

ZnS_cutoff_energy = 11.*(2*np.pi/ZnS_lat_const)**2
ZnS_atomic_positions = [ZnS_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnS_nvalence_electrons = 18
ZnS_PP = CohenEmpiricalPP(ZnS_lattice, ZnS_spff, ZnS_apff,
                          ZnS_cutoff_energy, ZnS_atomic_positions,
                          ZnS_nvalence_electrons)

#### Pseudopotential of ZnSe ####
ZnSe_lat_centering = "face"
ZnSe_lat_const = 5.65*angstrom_to_Bohr # the lattice constant in Bohr
ZnSe_lat_consts = [ZnSe_lat_const]*3
ZnSe_lat_angles = [np.pi/2]*3
ZnSe_lattice = Lattice(ZnSe_lat_centering, ZnSe_lat_consts, ZnSe_lat_angles)

ZnSe_cutoff_energy = 11.*(2*np.pi/ZnSe_lat_const)**2
ZnSe_atomic_positions = [ZnSe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnSe_nvalence_electrons = 18
ZnSe_PP = CohenEmpiricalPP(ZnSe_lattice, ZnSe_spff, ZnSe_apff,
                           ZnSe_cutoff_energy, ZnSe_atomic_positions,
                           ZnSe_nvalence_electrons)

#### Pseudopotential of ZnTe ####
ZnTe_lat_centering = "face"
ZnTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
ZnTe_lat_consts = [ZnTe_lat_const]*3
ZnTe_lat_angles = [np.pi/2]*3
ZnTe_lattice = Lattice(ZnTe_lat_centering, ZnTe_lat_consts, ZnTe_lat_angles)

ZnTe_cutoff_energy = 11.*(2*np.pi/ZnTe_lat_const)**2
ZnTe_atomic_positions = [ZnTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnTe_nvalence_electrons = 18
ZnTe_PP = CohenEmpiricalPP(ZnTe_lattice, ZnTe_spff, ZnTe_apff,
                           ZnTe_cutoff_energy, ZnTe_atomic_positions,
                           ZnTe_nvalence_electrons)

#### Pseudopotential of CdTe ####
CdTe_lat_centering = "face"
CdTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
CdTe_lat_consts = [CdTe_lat_const]*3
CdTe_lat_angles = [np.pi/2]*3
CdTe_lattice = Lattice(CdTe_lat_centering, CdTe_lat_consts, CdTe_lat_angles)

CdTe_cutoff_energy = 11.*(2*np.pi/CdTe_lat_const)**2
CdTe_atomic_positions = [CdTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
CdTe_nvalence_electrons = 18
CdTe_PP = CohenEmpiricalPP(CdTe_lattice, CdTe_spff, CdTe_apff,
                           CdTe_cutoff_energy, CdTe_atomic_positions,
                           CdTe_nvalence_electrons)

# The end of Cohen's 14 pseudopotentials with diamond or zinc-blende structure.

#### Toy Pseudopotential ####
Toy_lat_centering = "prim"
Toy_lat_const = 1.
Toy_lat_consts = [Toy_lat_const]*3
Toy_lat_angles = [np.pi/2]*3
Toy_lattice = Lattice(Toy_lat_centering, Toy_lat_consts, Toy_lat_angles)

Toy_pff = [0.0]
Toy_energy_cutoff = 2*(2*np.pi/Toy_lat_const)**2
Toy_atomic_positions = [[0.]*3]
Toy_nvalence_electrons = 3
Toy_PP = EmpiricalPP(Toy_lattice, Toy_pff, Toy_energy_cutoff, Toy_atomic_positions,
                     Toy_nvalence_electrons)

# Free electron Pseudopotential ####
free_lat_centering = "prim"
free_lat_const = 1.
free_lat_consts = [free_lat_const]*3
free_lat_angles = [np.pi/2]*3
free_lattice = Lattice(free_lat_centering, free_lat_consts, free_lat_angles)

free_pff = [0.0]
free_energy_cutoff = 2*(2*np.pi/free_lat_const)**2
free_atomic_positions = [[0.]*3]
free_nvalence_electrons = 3
free_PP = EmpiricalPP(free_lattice, free_pff, free_energy_cutoff,
                      free_atomic_positions, free_nvalence_electrons)

# The following pseudopotentials come from: 
# Cohen, Marvin L., and Volker Heine. "The fitting of pseudopotentials to
# experimental data and their subsequent application." Solid state physics 24
# (1970): 37-248. APA

#### Pseudopotential of Al ####
Al_centering_type = "face"
Al_lat_const = 4.05*angstrom_to_Bohr
Al_lat_consts = [Al_lat_const]*3
Al_lat_angles = [np.pi/2]*3
Al_lattice = Lattice(Al_centering_type, Al_lat_consts, Al_lat_angles)

Al_pff = [0.0179, 0.0562]
Al_energy_cutoff = 4*(2*np.pi/Al_lat_const)**2
Al_atomic_positions = [[0.,0.,0.]]
Al_nvalence_electrons = 3
Al_PP = EmpiricalPP(Al_lattice, Al_pff, Al_energy_cutoff, Al_atomic_positions,
                    Al_nvalence_electrons)

#### Pseudopotential of Li ####
Li_centering_type = "body"
Li_lat_const = 2.968*angstrom_to_Bohr
Li_lat_consts = [Li_lat_const]*3
Li_lat_angles = [np.pi/2]*3
Li_lattice = Lattice(Li_centering_type, Li_lat_consts, Li_lat_angles)

Li_pff = [0.11, 0.0]
Li_energy_cutoff = 4*(2*np.pi/Li_lat_const)**2
Li_atomic_positions = [[0.,0.,0.]]
Li_nvalence_electrons = 1
Li_PP = EmpiricalPP(Li_lattice, Li_pff, Li_energy_cutoff, Li_atomic_positions,
                    Li_nvalence_electrons)

#### Pseudopotential of Na ####
Na_centering_type = "body"
Na_lat_const = 3.633*angstrom_to_Bohr
Na_lat_consts = [Na_lat_const]*3
Na_lat_angles = [np.pi/2]*3
Na_lattice = Lattice(Na_centering_type, Na_lat_consts, Na_lat_angles)

Na_pff = [0.0158]
Na_energy_cutoff = 2*(2*np.pi/Na_lat_const)**2
Na_atomic_positions = [[0.]*3]
Na_nvalence_electrons = 1
Na_PP = EmpiricalPP(Na_lattice, Na_pff, Na_energy_cutoff, Na_atomic_positions,
                    Na_nvalence_electrons)
    
#### Pseudopotential of K ####
K_centering_type = "body"
K_lat_const = 9.873*angstrom_to_Bohr
K_lat_consts = [K_lat_const]*3
K_lat_angles = [np.pi/2]*3
K_lattice = Lattice(K_centering_type, K_lat_consts, K_lat_angles)

K_pff = [0.0075, -0.009]
K_energy_cutoff = 4*(2*np.pi/K_lat_const)**2
K_atomic_positions = [[0.]*3]
K_nvalence_electrons = 1
K_PP = EmpiricalPP(K_lattice, K_pff, K_energy_cutoff, K_atomic_positions,
                   K_nvalence_electrons)

#### Pseudopotential of Rb ####
Rb_centering_type = "body"
Rb_lat_const = 5.585*angstrom_to_Bohr
Rb_lat_consts = [Rb_lat_const]*3
Rb_lat_angles = [np.pi/2]*3
Rb_lattice = Lattice(Rb_centering_type, Rb_lat_consts, Rb_lat_angles)

Rb_pff = [-0.002]
Rb_energy_cutoff = 2*(2*np.pi/Rb_lat_const)**2
Rb_atomic_positions = [[0.]*3]
Rb_nvalence_electrons = 1
Rb_PP = EmpiricalPP(Rb_lattice, Rb_pff, Rb_energy_cutoff, Rb_atomic_positions,
                    Rb_nvalence_electrons)

#### Pseudopotential of Cs ####
Cs_centering_type = "body"
Cs_lat_const = 6.141*angstrom_to_Bohr
Cs_lat_consts = [Cs_lat_const]*3
Cs_lat_angles = [np.pi/2]*3
Cs_lattice = Lattice(Cs_centering_type, Cs_lat_consts, Cs_lat_angles)

Cs_pff = [-0.03]
Cs_energy_cutoff = 2*(2*np.pi/Cs_lat_const)**2
Cs_atomic_positions = [[0.]*3]
Cs_nvalence_electrons = 1
Cs_PP = EmpiricalPP(Cs_lattice, Cs_pff, Cs_energy_cutoff, Cs_atomic_positions,
                    Cs_nvalence_electrons)
    
#### Pseudopotential of Cu ####
Cu_centering_type = "face"
Cu_lat_const = 3.615*angstrom_to_Bohr
Cu_lat_consts = [Cu_lat_const]*3
Cu_lat_angles = [np.pi/2]*3
Cu_lattice = Lattice(Cu_centering_type, Cu_lat_consts, Cu_lat_angles)

Cu_pff = [0.282, 0.18]
Cu_energy_cutoff = 4*(2*np.pi/Cu_lat_const)**2
Cu_atomic_positions = [[0.]*3]
Cu_nvalence_electrons = 11
Cu_PP = EmpiricalPP(Cu_lattice, Cu_pff, Cu_energy_cutoff, Cu_atomic_positions,
                    Cu_nvalence_electrons)

#### Pseudopotential of Ag ####
Ag_centering_type = "face"
Ag_lat_const = 4.0853*angstrom_to_Bohr
Ag_lat_consts = [Ag_lat_const]*3
Ag_lat_angles = [np.pi/2]*3
Ag_lattice = Lattice(Ag_centering_type, Ag_lat_consts, Ag_lat_angles)

Ag_pff = [0.195, 0.121]
Ag_energy_cutoff = 4*(2*np.pi/Ag_lat_const)**2
Ag_atomic_positions = [[0.]*3]
Ag_nvalence_electrons = 11
Ag_PP = EmpiricalPP(Ag_lattice, Ag_pff, Ag_energy_cutoff, Ag_atomic_positions,
                    Ag_nvalence_electrons)

#### Pseudopotential of Au ####
Au_centering_type = "face"
Au_lat_const = 4.0782*angstrom_to_Bohr
Au_lat_consts = [Au_lat_const]*3
Au_lat_angles = [np.pi/2]*3
Au_lattice = Lattice(Au_centering_type, Au_lat_consts, Au_lat_angles)

Au_pff = [0.252, 0.152]
Au_energy_cutoff = 4*(2*np.pi/Au_lat_const)**2
Au_atomic_positions = [[0.]*3]
Au_nvalence_electrons = 11
Au_PP = EmpiricalPP(Au_lattice, Au_pff, Au_energy_cutoff, Au_atomic_positions,
                    Au_nvalence_electrons)

#### Pseudopotential of Pb ####
Pb_centering_type = "face"
Pb_lat_const = 4.9508*angstrom_to_Bohr
Pb_lat_consts = [Pb_lat_const]*3
Pb_lat_angles = [np.pi/2]*3
Pb_lattice = Lattice(Pb_centering_type, Pb_lat_consts, Pb_lat_angles)

Pb_pff = [-0.084, -0.039]
Pb_energy_cutoff = 4*(2*np.pi/Pb_lat_const)**2
Pb_atomic_positions = [[0.]*3]
Pb_nvalence_electrons = 4
Pb_PP = EmpiricalPP(Pb_lattice, Pb_pff, Pb_energy_cutoff, Pb_atomic_positions,
                    Pb_nvalence_electrons)

#### Pseudopotential of Mg ####
Mg_centering_type = "prim"
Mg_lat_const_a = 3.184
Mg_lat_const_c = 5.249
Mg_lat_consts = np.array([Mg_lat_const_a, Mg_lat_const_a,
                          Mg_lat_const_c])*angstrom_to_Bohr
Mg_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Mg_lattice = Lattice(Mg_centering_type, Mg_lat_consts, Mg_lat_angles)


Mg_pff = [0., 0., .026, 0., 0., 0., .014, .036, 0., 0., .058]
Mg_cutoff_vec = (Mg_lattice.reciprocal_vectors[:,0] +
                 Mg_lattice.reciprocal_vectors[:,1] +
                 2*Mg_lattice.reciprocal_vectors[:,2])
Mg_energy_cutoff = np.dot(Mg_cutoff_vec, Mg_cutoff_vec)
Mg_atomic_positions = [[0.]*3]
Mg_nvalence_electrons = 2
Mg_PP = EmpiricalPP(Mg_lattice, Mg_pff, Mg_energy_cutoff, Mg_atomic_positions,
                    Mg_nvalence_electrons)

#### Pseudopotential of Zn ####
Zn_centering_type = "prim"
Zn_lat_const_a = 2.627
Zn_lat_const_c = 5.207
Zn_lat_consts = np.array([Zn_lat_const_a, Zn_lat_const_a,
                          Zn_lat_const_c])*angstrom_to_Bohr
Zn_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Zn_lattice = Lattice(Zn_centering_type, Zn_lat_consts, Zn_lat_angles)

Zn_pff = [0., -0.022, 0.02, 0.063]
Zn_cutoff_vec = (Zn_lattice.reciprocal_vectors[:,0] +
                 Zn_lattice.reciprocal_vectors[:,2])
Zn_energy_cutoff = np.dot(Zn_cutoff_vec, Zn_cutoff_vec)
Zn_atomic_positions = [[0.]*3]
Zn_nvalence_electrons = 12
Zn_PP = EmpiricalPP(Zn_lattice, Zn_pff, Zn_energy_cutoff, Zn_atomic_positions,
                    Zn_nvalence_electrons)


# See Band structure and Fermi surface of Zine and Cadmium by Stark and
# Falicov for Cd form factors.
#### Pseudopotential of Cd ####
Cd_centering_type = "prim"
Cd_lat_const_a = 2.9684
Cd_lat_const_c = 5.5261
Cd_lat_consts = np.array([Cd_lat_const_a, Cd_lat_const_a,
                          Cd_lat_const_c])*angstrom_to_Bohr
Cd_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Cd_lattice = Lattice(Cd_centering_type, Cd_lat_consts, Cd_lat_angles)

Cd_pff = [0., -0.017, 0., 0., 0., 0., 0., 0.0235, 0.029, 0., 0.03]
Cd_cutoff_vec = (Cd_lattice.reciprocal_vectors[:,0] +
                 Cd_lattice.reciprocal_vectors[:,1] +
                 2*Cd_lattice.reciprocal_vectors[:,2])
Cd_energy_cutoff = np.dot(Cd_cutoff_vec, Cd_cutoff_vec)
Cd_atomic_positions = [[0.]*3]
Cd_nvalence_electrons = 12
Cd_PP = EmpiricalPP(Cd_lattice, Cd_pff, Cd_energy_cutoff, Cd_atomic_positions,
                    Cd_nvalence_electrons)

#### Pseudopotential of Hg ####
Hg_centering_type = "prim"
Hg_lat_const = 2.9863*angstrom_to_Bohr
Hg_lat_consts = [Hg_lat_const]*3
Hg_lat_angles = [70.446*np.pi/180]*3
Hg_lattice = Lattice(Hg_centering_type, Hg_lat_consts, Hg_lat_angles)

Hg_pff = [-0.018, 0.028, 0.028]
Hg_cutoff_vec = (Hg_lattice.reciprocal_vectors[:, 0] +
                 Hg_lattice.reciprocal_vectors[:, 1])

Hg_energy_cutoff = np.dot(Hg_cutoff_vec, Hg_cutoff_vec)
Hg_atomic_positions = [[0.]*3]
Hg_nvalence_electrons = 12
Hg_PP = EmpiricalPP(Hg_lattice, Hg_pff, Hg_energy_cutoff, Hg_atomic_positions,
                    Hg_nvalence_electrons)

#### Pseudopotential of In ####
In_centering_type = "body"
In_lat_const_a = 3.2992*angstrom_to_Bohr
In_lat_const_c = 4.9049*angstrom_to_Bohr
In_lat_consts = [In_lat_const_a, In_lat_const_a, In_lat_const_c]
In_lat_angles = [np.pi/2]*3
In_lattice = Lattice(In_centering_type, In_lat_consts, In_lat_angles)

In_pff = [0., 0., 0., 0., 0., 0., 0., -0.020, 0., 0., 0., -0.047]
In_cutoff_vec = (In_lattice.reciprocal_vectors[:, 0] +
                 In_lattice.reciprocal_vectors[:, 1] +
                 In_lattice.reciprocal_vectors[:, 2])
In_energy_cutoff = np.dot(In_cutoff_vec, In_cutoff_vec)
In_atomic_positions = [[0.]*3]
In_nvalence_electrons = 3
In_PP = EmpiricalPP(In_lattice, In_pff, In_energy_cutoff, In_atomic_positions,
                    In_nvalence_electrons)

#### Pseudopotential of Sn ####
Sn_centering_type = "body"
Sn_lat_const_a = 3.47*angstrom_to_Bohr
Sn_lat_const_c = 4.87*angstrom_to_Bohr
Sn_lat_consts = [Sn_lat_const_a, Sn_lat_const_a, Sn_lat_const_c]
Sn_lat_angles = [np.pi/2]*3
Sn_lattice = Lattice(Sn_centering_type, Sn_lat_consts, Sn_lat_angles)

Sn_pff = [0.]*5 + [-0.056, 0., -0.069] + [0.]*13 + [0.033, 0., 0.051]
Sn_cutoff_vec = (2*Sn_lattice.reciprocal_vectors[:, 0] +
                 Sn_lattice.reciprocal_vectors[:, 1] + 
                 Sn_lattice.reciprocal_vectors[:, 2])
Sn_energy_cutoff = np.dot(Sn_cutoff_vec, Sn_cutoff_vec)
Sn_atomic_positions = [[0.]*3]
Sn_nvalence_electrons = 4
Sn_PP = EmpiricalPP(Sn_lattice, Sn_pff, Sn_energy_cutoff, Sn_atomic_positions,
                    Sn_nvalence_electrons)


