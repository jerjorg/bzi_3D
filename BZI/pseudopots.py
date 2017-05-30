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
        atomic_basis (list): a list of atomic positions.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        form_factors (list): a list of pseudopotential form factors. Every 
            energy shell up to the cutoff energy should be accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.  
        rlat_pts (list): a list of reciprocal lattice points included in the 
            Fourier expansion.
        energy_shells (list): a list of spherical shells that points in rlat_pts
            reside on.
        atomic_basis (list): a list of atomic positions.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.

    Example:
        >>> centering_type = "face"
        >>> lat_const = 4.0
        >>> lat_consts = [lat_const]*3
        >>> lat_angles = [np.pi/2]*3
        >>> lattice = Lattice(centering_type, lat_consts, lat_angles)
        >>> pff = [0.019, 0.055]
        >>> cutoff = 4*(2*np.pi/lat_const)**2
        >>> atomic_basis = [0.,0.,0.]
        >>> PP = EmpiricalPP(lattice, pff, cutoff, atomic_basis)
    """
    
    def __init__(self, lattice, form_factors, energy_cutoff, atomic_positions,
                 energy_shift=None, fermi_level=None):
        self.lattice = lattice
        self.form_factors = form_factors
        self.energy_cutoff = energy_cutoff
        if np.shape(atomic_positions) == (3,):
            msg = ("Please provide a list of atomic positions instead of an "
                   "individual atomic position.")
            raise ValueError(msg.format(atomic_positions))
        else:
            self.atomic_positions = atomic_positions
        self.atomic_positions = atomic_positions
        self.rlat_pts = sphere_pts(lattice.reciprocal_vectors,
                                    energy_cutoff)
        self.find_energy_shells()
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.

    def find_energy_shells(self):
        """Find the spherical shells of constant on which the points in
        :py:data:`rlat_pts` reside. Return the radius squared of these shells.
        """

        shells = []
        for rpt in self.rlat_pts:
            r2 = np.dot(rpt, rpt)
            if any(np.isclose(r2,shells)):
                continue
            else:
                shells.append(r2)
        self.energy_shells = np.sort(shells)[1:]
        
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
        atomic_basis (list): a list of atomic positions.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.

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
        atomic_basis (list): a list of atomic positions.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
    """
    
    def __init__(self, lattice, sym_form_factors, antisym_form_factors,
                 energy_cutoff, atomic_positions, energy_shift=None,
                 fermi_level=None):
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
        self.atomic_positions = atomic_positions
        self.rlat_pts = sphere_pts(lattice.reciprocal_vectors,
                                    energy_cutoff)
        self.find_energy_shells()
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.

    def find_energy_shells(self):
        """Find the spherical shells of constant energy on which the points in
        :py:data:`rlat_pts` reside. Return the radius squared of these shells.
        """

        shells = []
        for rpt in self.rlat_pts:
            r2 = np.dot(rpt, rpt)
            if any(np.isclose(r2,shells)):
                continue
            else:
                shells.append(r2)
        self.energy_shells = np.sort(shells)[1:]
        
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
    
#### Toy Pseudopotential ####
Toy_lat_center = "prim"
Toy_lat_consts = [1., 1., 1.]
Toy_lat_angles = [np.pi/2]*3
Toy_lattice = Lattice(Toy_lat_center, Toy_lat_consts, Toy_lat_angles)
# Toy lattice vectors
# Toy_lv = make_ptvecs(Toy_lat_center, Toy_lat_const_list, Toy_lat_angles)
# Toy_pff = [0.2]
#Toy_shells = [[0.,0.,0], [0.,0.,1.]]
#nested_shells = [shells(i, Toy_lv) for i in Toy_shells]
#Toy_rlat_pts = np.array(list(itertools.chain(*nested_shells)))
Toy_rlat_pts = sphere_pts(Toy_lattice.vectors, 1.)

# The number of contributing reciprocal lattice points determines the size
# of the Hamiltonian.
nToy = len(Toy_rlat_pts)

def ToyPP(kpt):
    """Evaluate a Toy pseudopotential at a given k-point.

    Args:
        kpoint (numpy.ndarray): a sampling point.

    Return:
        (numpy.ndarray): the sorted eigenvalues of the Hamiltonian at the provided
        sampling point.
    """
    
    # Initialize the Toy pseudopotential Hamiltonian.
    Toy_H = np.empty([nToy, nToy])

    # Construct the Toy Hamiltonian.
    for (i,k1) in enumerate(Toy_rlat_pts):
        for (j,k2) in enumerate(Toy_rlat_pts):
            if np.isclose(norm(k2 - k1), 1.) == True:
                Toy_H[i,j] = Toy_pff[0]
            elif i == j:
                Toy_H[i,j] = norm(kpt + k1)**2
            else:
                Toy_H[i,j] = 0.
                
    return np.sort(np.linalg.eigvals(Toy_H))

#### Free electron Pseudopotential ####
Free_lat_center = "prim"
Free_lat_const_list = [1.]*3
Free_lat_angles = [np.pi/2]*3
# Free lattice vectors
Free_lv = make_ptvecs(Free_lat_center, Free_lat_const_list, Free_lat_angles)

Free_pff = [0.2]
Free_shells = [[0.,0.,0], [0.,0.,1.]]
nested_shells = [shells(i, Free_lv) for i in Free_shells]
Free_rlat_pts = np.array(list(itertools.chain(*nested_shells)))

def FreePP(pt):
    """Evaluate the free-electron pseudopotential at a given point. The method
    employed here ignores band sorting issues.
    """
    return np.asarray([norm(pt + rpt)**2 for rpt in Free_rlat_pts])


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
# radius       3     4     8    11
Si_spff =    [-0.21, 0.00, 0.04, 0.08]
Si_apff =    [0.00, 0.00, 0.00, 0.00]
Ge_spff =    [-0.23, 0.00, 0.01, 0.06]
Ge_apff =    [0.00, 0.00, 0.00, 0.00]
Sn_spff =    [-0.20, 0.00, 0.00, 0.04]
Sn_apff =    [0.00, 0.00, 0.00, 0.00]
GaP_spff =   [-0.22, 0.00, 0.03, 0.07]
GaP_apff =   [0.12, 0.07, 0.00, 0.02]
GaAs_spff =  [-0.23, 0.00, 0.01, 0.06]
GaAs_apff =  [0.07, 0.05, 0.00, 0.01]
AlSb_spff =  [-0.21, 0.00, 0.02, 0.06]
AlSb_apff =  [0.06, 0.04, 0.00, 0.02]
InP_spff =   [-0.23, 0.00, 0.01, 0.06]
InP_apff =   [0.07, 0.05, 0.00, 0.01]
GaSb_spff =  [-0.22, 0.00, 0.00, 0.05]
GaSb_apff =  [0.06, 0.05, 0.00, 0.01]
InAs_spff =  [-0.22, 0.00, 0.00, 0.05]
InAs_apff =  [0.08, 0.05, 0.00, 0.03]
InSb_spff =  [-0.20, 0.00, 0.00, 0.04]
InSb_apff =  [0.06, 0.05, 0.00, 0.01]
ZnS_spff =   [-0.22, 0.00, 0.03, 0.07]
ZnS_apff =   [0.24, 0.14, 0.00, 0.04]
ZnSe_spff =  [-0.23, 0.00, 0.01, 0.06]
ZnSe_apff =  [0.18, 0.12, 0.00, 0.03]
ZnTe_spff =  [-0.22, 0.00, 0.00, 0.05]
ZnTe_apff =  [0.13, 0.10, 0.00, 0.01]
CdTe_spff =  [-0.20, 0.00, 0.00, 0.04]
CdTe_apff =  [0.15, 0.09, 0.00, 0.04]    

#### Pseudopotential of Si ####
Si_lat_centering = "face"
Si_lat_const = 5.43*angstrom_to_Bohr # the lattice constant in Bohr
Si_lat_consts = [Si_lat_const]*3
Si_lat_angles = [np.pi/2]*3
Si_lattice = Lattice(Si_lat_centering, Si_lat_consts, Si_lat_angles)

Si_r = 11.*(2*np.pi/Si_lat_const)**2
Si_tau = [Si_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Si_PP = CohenEmpiricalPP(Si_lattice, Si_spff, Si_apff, Si_r, Si_tau)

#### Pseudopotential of Ge ####
Ge_lat_centering = "face"
Ge_lat_const = 5.66*angstrom_to_Bohr # the lattice constant in Bohr
Ge_lat_consts = [Ge_lat_const]*3
Ge_lat_angles = [np.pi/2]*3
Ge_lattice = Lattice(Ge_lat_centering, Ge_lat_consts, Ge_lat_angles)

Ge_r = 11.*(2*np.pi/Ge_lat_const)**2
Ge_tau = [Ge_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Ge_PP = CohenEmpiricalPP(Ge_lattice, Ge_spff, Ge_apff, Ge_r, Ge_tau)

#### Pseudopotential of Sn ####
Sn_lat_centering = "face"
Sn_lat_const = 6.49*angstrom_to_Bohr # the lattice constant in Bohr
Sn_lat_consts = [Sn_lat_const]*3
Sn_lat_angles = [np.pi/2]*3
Sn_lattice = Lattice(Sn_lat_centering, Sn_lat_consts, Sn_lat_angles)

Sn_r = 11.*(2*np.pi/Sn_lat_const)**2
Sn_tau = [Sn_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Sn_PP = CohenEmpiricalPP(Sn_lattice, Sn_spff, Sn_apff, Sn_r, Sn_tau)

#### Pseudopotential of GaP ####
GaP_lat_centering = "face"
GaP_lat_const = 5.44*angstrom_to_Bohr # the lattice constant in Bohr
GaP_lat_consts = [GaP_lat_const]*3
GaP_lat_angles = [np.pi/2]*3
GaP_lattice = Lattice(GaP_lat_centering, GaP_lat_consts, GaP_lat_angles)

GaP_r = 11.*(2*np.pi/GaP_lat_const)**2
GaP_tau = [GaP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaP_PP = CohenEmpiricalPP(GaP_lattice, GaP_spff, GaP_apff, GaP_r, GaP_tau)

#### Pseudopotential of GaAs ####
GaAs_lat_centering = "face"
GaAs_lat_const = 5.64*angstrom_to_Bohr # the lattice constant in Bohr
GaAs_lat_consts = [GaAs_lat_const]*3
GaAs_lat_angles = [np.pi/2]*3
GaAs_lattice = Lattice(GaAs_lat_centering, GaAs_lat_consts, GaAs_lat_angles)

GaAs_r = 11.*(2*np.pi/GaAs_lat_const)**2
GaAs_tau = [GaAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaAs_PP = CohenEmpiricalPP(GaAs_lattice, GaAs_spff, GaAs_apff, GaAs_r, GaAs_tau)

#### Pseudopotential of AlSb ####
AlSb_lat_centering = "face"
AlSb_lat_const = 6.13*angstrom_to_Bohr # the lattice constant in Bohr
AlSb_lat_consts = [AlSb_lat_const]*3
AlSb_lat_angles = [np.pi/2]*3
AlSb_lattice = Lattice(AlSb_lat_centering, AlSb_lat_consts, AlSb_lat_angles)

AlSb_r = 11.*(2*np.pi/AlSb_lat_const)**2
AlSb_tau = [AlSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
AlSb_PP = CohenEmpiricalPP(AlSb_lattice, AlSb_spff, AlSb_apff, AlSb_r, AlSb_tau)

#### Pseudopotential of InP ####
InP_lat_centering = "face"
InP_lat_const = 5.86*angstrom_to_Bohr # the lattice constant in Bohr
InP_lat_consts = [InP_lat_const]*3
InP_lat_angles = [np.pi/2]*3
InP_lattice = Lattice(InP_lat_centering, InP_lat_consts, InP_lat_angles)

InP_r = 11.*(2*np.pi/InP_lat_const)**2
InP_tau = [InP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InP_PP = CohenEmpiricalPP(InP_lattice, InP_spff, InP_apff, InP_r, InP_tau)

#### Pseudopotential of GaSb ####
GaSb_lat_centering = "face"
GaSb_lat_const = 6.12*angstrom_to_Bohr # the lattice constant in Bohr
GaSb_lat_consts = [GaSb_lat_const]*3
GaSb_lat_angles = [np.pi/2]*3
GaSb_lattice = Lattice(GaSb_lat_centering, GaSb_lat_consts, GaSb_lat_angles)

GaSb_r = 11.*(2*np.pi/GaSb_lat_const)**2
GaSb_tau = [GaSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaSb_PP = CohenEmpiricalPP(GaSb_lattice, GaSb_spff, GaSb_apff, GaSb_r, GaSb_tau)

#### Pseudopotential of InAs ####
InAs_lat_centering = "face"
InAs_lat_const = 6.04*angstrom_to_Bohr # the lattice constant in Bohr
InAs_lat_consts = [InAs_lat_const]*3
InAs_lat_angles = [np.pi/2]*3
InAs_lattice = Lattice(InAs_lat_centering, InAs_lat_consts, InAs_lat_angles)

InAs_r = 11.*(2*np.pi/InAs_lat_const)**2
InAs_tau = [InAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InAs_PP = CohenEmpiricalPP(InAs_lattice, InAs_spff, InAs_apff, InAs_r, InAs_tau)

#### Pseudopotential of InSb ####
InSb_lat_centering = "face"
InSb_lat_const = 6.48*angstrom_to_Bohr # the lattice constant in Bohr
InSb_lat_consts = [InSb_lat_const]*3
InSb_lat_angles = [np.pi/2]*3
InSb_lattice = Lattice(InSb_lat_centering, InSb_lat_consts, InSb_lat_angles)

InSb_r = 11.*(2*np.pi/InSb_lat_const)**2
InSb_tau = [InSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InSb_PP = CohenEmpiricalPP(InSb_lattice, InSb_spff, InSb_apff, InSb_r, InSb_tau)

#### Pseudopotential of ZnS ####
ZnS_lat_centering = "face"
ZnS_lat_const = 5.41*angstrom_to_Bohr # the lattice constant in Bohr
ZnS_lat_consts = [ZnS_lat_const]*3
ZnS_lat_angles = [np.pi/2]*3
ZnS_lattice = Lattice(ZnS_lat_centering, ZnS_lat_consts, ZnS_lat_angles)

ZnS_r = 11.*(2*np.pi/ZnS_lat_const)**2
ZnS_tau = [ZnS_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnS_PP = CohenEmpiricalPP(ZnS_lattice, ZnS_spff, ZnS_apff, ZnS_r, ZnS_tau)

#### Pseudopotential of ZnSe ####
ZnSe_lat_centering = "face"
ZnSe_lat_const = 5.65*angstrom_to_Bohr # the lattice constant in Bohr
ZnSe_lat_consts = [ZnSe_lat_const]*3
ZnSe_lat_angles = [np.pi/2]*3
ZnSe_lattice = Lattice(ZnSe_lat_centering, ZnSe_lat_consts, ZnSe_lat_angles)

ZnSe_r = 11.*(2*np.pi/ZnSe_lat_const)**2
ZnSe_tau = [ZnSe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnSe_PP = CohenEmpiricalPP(ZnSe_lattice, ZnSe_spff, ZnSe_apff, ZnSe_r, ZnSe_tau)

#### Pseudopotential of ZnTe ####
ZnTe_lat_centering = "face"
ZnTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
ZnTe_lat_consts = [ZnTe_lat_const]*3
ZnTe_lat_angles = [np.pi/2]*3
ZnTe_lattice = Lattice(ZnTe_lat_centering, ZnTe_lat_consts, ZnTe_lat_angles)

ZnTe_r = 11.*(2*np.pi/ZnTe_lat_const)**2
ZnTe_tau = [ZnTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnTe_PP = CohenEmpiricalPP(ZnTe_lattice, ZnTe_spff, ZnTe_apff, ZnTe_r, ZnTe_tau)

#### Pseudopotential of CdTe ####
CdTe_lat_centering = "face"
CdTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
CdTe_lat_consts = [CdTe_lat_const]*3
CdTe_lat_angles = [np.pi/2]*3
CdTe_lattice = Lattice(CdTe_lat_centering, CdTe_lat_consts, CdTe_lat_angles)

CdTe_r = 11.*(2*np.pi/CdTe_lat_const)**2
CdTe_tau = [CdTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
CdTe_PP = CohenEmpiricalPP(CdTe_lattice, CdTe_spff, CdTe_apff, CdTe_r, CdTe_tau)

# The end of Cohen's 14 pseudopotentials with diamond or zinc-blende structure.

# The following pseudopotentials come from: 
#Cohen, Marvin L., and Volker Heine. "The fitting of pseudopotentials to
# experimental data and their subsequent application." Solid state physics 24
# (1970): 37-248. APA

#### Pseudopotential of Al ####
Al_centering_type = "face"
Al_lat_const = 4.05*angstrom_to_Bohr
Al_lat_consts = [Al_lat_const]*3
Al_lat_angles = [np.pi/2]*3
Al_lattice = Lattice(Al_centering_type, Al_lat_consts, Al_lat_angles)

Al_pff = [0.0179, 0.0562]
Al_cutoff = 4*(2*np.pi/Al_lat_const)**2
Al_atomic_basis = [[0.,0.,0.]]
Al_PP = EmpiricalPP(Al_lattice, Al_pff, Al_cutoff, Al_atomic_basis)

#### Pseudopotential of Li ####
Li_centering_type = "body"
Li_lat_const = 2.968*angstrom_to_Bohr
Li_lat_consts = [Li_lat_const]*3
Li_lat_angles = [np.pi/2]*3
Li_lattice = Lattice(Li_centering_type, Li_lat_consts, Li_lat_angles)

Li_pff = [0.11, 0.0]
Li_cutoff = 4*(2*np.pi/Li_lat_const)**2
Li_atomic_basis = [[0.,0.,0.]]
Li_PP = EmpiricalPP(Li_lattice, Li_pff, Li_cutoff, Li_atomic_basis)

#### Pseudopotential of Na ####
Na_centering_type = "body"
Na_lat_const = 3.633*angstrom_to_Bohr
Na_lat_consts = [Na_lat_const]*3
Na_lat_angles = [np.pi/2]*3
Na_lattice = Lattice(Na_centering_type, Na_lat_consts, Na_lat_angles)

Na_pff = [0.0158]
Na_cutoff = 2*(2*np.pi/Na_lat_const)**2
Na_atomic_basis = [[0.]*3]
Na_PP = EmpiricalPP(Na_lattice, Na_pff, Na_cutoff, Na_atomic_basis)
    
#### Pseudopotential of K ####
K_centering_type = "body"
K_lat_const = 9.873*angstrom_to_Bohr
K_lat_consts = [K_lat_const]*3
K_lat_angles = [np.pi/2]*3
K_lattice = Lattice(K_centering_type, K_lat_consts, K_lat_angles)

K_pff = [0.0075, -0.009]
K_cutoff = 4*(2*np.pi/K_lat_const)**2
K_atomic_basis = [[0.]*3]
K_PP = EmpiricalPP(K_lattice, K_pff, K_cutoff, K_atomic_basis)

#### Pseudopotential of Rb ####
Rb_centering_type = "body"
Rb_lat_const = 5.585*angstrom_to_Bohr
Rb_lat_consts = [Rb_lat_const]*3
Rb_lat_angles = [np.pi/2]*3
Rb_lattice = Lattice(Rb_centering_type, Rb_lat_consts, Rb_lat_angles)

Rb_pff = [-0.002]
Rb_cutoff = 2*(2*np.pi/Rb_lat_const)**2
Rb_atomic_basis = [[0.]*3]
Rb_PP = EmpiricalPP(Rb_lattice, Rb_pff, Rb_cutoff, Rb_atomic_basis)

#### Pseudopotential of Cs ####
Cs_centering_type = "body"
Cs_lat_const = 6.141*angstrom_to_Bohr
Cs_lat_consts = [Cs_lat_const]*3
Cs_lat_angles = [np.pi/2]*3
Cs_lattice = Lattice(Cs_centering_type, Cs_lat_consts, Cs_lat_angles)

Cs_pff = [-0.03]
Cs_cutoff = 2*(2*np.pi/Cs_lat_const)**2
Cs_atomic_basis = [[0.]*3]
Cs_PP = EmpiricalPP(Cs_lattice, Cs_pff, Cs_cutoff, Cs_atomic_basis)
    
#### Pseudopotential of Cu ####
Cu_centering_type = "face"
Cu_lat_const = 3.615*angstrom_to_Bohr
Cu_lat_consts = [Cu_lat_const]*3
Cu_lat_angles = [np.pi/2]*3
Cu_lattice = Lattice(Cu_centering_type, Cu_lat_consts, Cu_lat_angles)

Cu_pff = [0.282, 0.18]
Cu_cutoff = 4*(2*np.pi/Cu_lat_const)**2
Cu_atomic_basis = [[0.]*3]
Cu_PP = EmpiricalPP(Cu_lattice, Cu_pff, Cu_cutoff, Cu_atomic_basis)

#### Pseudopotential of Ag ####
Ag_centering_type = "face"
Ag_lat_const = 4.0853*angstrom_to_Bohr
Ag_lat_consts = [Ag_lat_const]*3
Ag_lat_angles = [np.pi/2]*3
Ag_lattice = Lattice(Ag_centering_type, Ag_lat_consts, Ag_lat_angles)

Ag_pff = [0.195, 0.121]
Ag_cutoff = 4*(2*np.pi/Ag_lat_const)**2
Ag_atomic_basis = [[0.]*3]
Ag_PP = EmpiricalPP(Ag_lattice, Ag_pff, Ag_cutoff, Ag_atomic_basis)

#### Pseudopotential of Au ####
Au_centering_type = "face"
Au_lat_const = 4.0782*angstrom_to_Bohr
Au_lat_consts = [Au_lat_const]*3
Au_lat_angles = [np.pi/2]*3
Au_lattice = Lattice(Au_centering_type, Au_lat_consts, Au_lat_angles)

Au_pff = [0.252, 0.152]
Au_cutoff = 4*(2*np.pi/Au_lat_const)**2
Au_atomic_basis = [[0.]*3]
Au_PP = EmpiricalPP(Au_lattice, Au_pff, Au_cutoff, Au_atomic_basis)

#### Pseudopotential of Pb ####
Pb_centering_type = "face"
Pb_lat_const = 4.9508*angstrom_to_Bohr
Pb_lat_consts = [Pb_lat_const]*3
Pb_lat_angles = [np.pi/2]*3
Pb_lattice = Lattice(Pb_centering_type, Pb_lat_consts, Pb_lat_angles)

Pb_pff = [-0.084, -0.039]
Pb_cutoff = 4*(2*np.pi/Pb_lat_const)**2
Pb_atomic_basis = [[0.]*3]
Pb_PP = EmpiricalPP(Pb_lattice, Pb_pff, Pb_cutoff, Pb_atomic_basis)
