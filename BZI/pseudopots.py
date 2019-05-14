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


class EmpiricalPseudopotential(object):
    """Create an empirical pseudopotential.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        form_factors (list): a list of pseudopotential form factors. Every 
            energy shell up to the cutoff energy should be accounted for.
        energy_cutoff (float): the cutoff energy of the Fourier expansion.
        atom_labels (list): a list of atom labels. The first atom type is 0,
            the following atom labels should be 1, 2, and so on.
        atom_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        material (str): a string describing the empirical pseudopotential,
            such as the chemical formula.
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
        atom_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons
        material (str): a string describing the empirical pseudopotential,
            such as the chemical formula.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy (float): the total energy.

    Example:
        >>> centering_type = "face"
        >>> lat_const = 4.0
        >>> lat_consts = [lat_const]*3
        >>> lat_angles = [np.pi/2]*3
        >>> lattice = Lattice(centering_type, lat_consts, lat_angles)
        >>> pff = [0.019, 0.055]
        >>> cutoff = 4*(2*np.pi/lat_const)**2
        >>> atom_positions = [0.,0.,0.]
        >>> Nvalence_electrons = 3
        >>> EPM = EmpiricalPseudopotential(lattice, pff, cutoff, nvalence_electrons,
        >>>                  atom_positions)
    """
    
    def __init__(self, lattice, form_factors, energy_cutoff, atom_labels, atom_positions,
                 nvalence_electrons, material, energy_shift=None,
                 fermi_level=None, total_energy=None):
        self.material = material
        self.lattice = lattice
        self.form_factors = form_factors
        self.energy_cutoff = energy_cutoff
        self.atom_labels = atom_labels        
        if np.shape(atom_positions) == (3,):
            msg = ("Please provide a list of atomic positions instead of a "
                   "single atomic position.")
            raise ValueError(msg.format(atom_positions))
        else:
            self.atom_positions = atom_positions
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                   self.energy_cutoff)
        self.find_energy_shells()
        self.energy_cutoff = self.energy_shells[len(self.form_factors)]
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                   self.energy_cutoff)
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        self.total_energy = total_energy or 0.        
        self.init_hamiltonian = self.hamiltonian([0.]*3) - np.diag(
            np.diag(self.hamiltonian([0.]*3)))
        
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

    
    def eval(self, kpoint, neigvals, adjust=False):
        """Evaluate the empirical pseudopotential eigenvalues at the provided
        k-point. Only return the lowest 'neigvals' eigenvalues.

        Args:
            kpoint (numpy.ndarray): a k-point
            neigvals (int): the number of eigenvalues to return
            adjust (bool): if true, the Fourier expansion will be performed about
                the k-point being considered.
        """
        

        if adjust:
            # Find all the reciprocal lattice points within a sphere surrounding the
            # k-point being considered.
            rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                  self.energy_cutoff, offset=kpoint)
            
            # Calculate the diagonal elements of the Hamiltonian.
            diag = np.eye(len(rlat_pts))*list(map(lambda x: np.dot(x,x),
                                                  rlat_pts + kpoint))
            
            # For some reason the lattice points need to be double nested for
            # numpy.tile to work.
            nested_rlatpts = np.array([[rlp] for rlp in rlat_pts])
            
            # Create a matrix of reciprocal lattice points where the first lattice
            # point is repeated acroass the first row, the second lattice point the
            # second row, and so on.
            rlatpt_mat = np.tile(nested_rlatpts,(len(rlat_pts),1))
            
            # Create a matrix of the differences of the lattice points. Each element
            # is given by a_i - a_j.
            rlp_diff = rlatpt_mat - np.transpose(rlatpt_mat, (1,0,2))
            
            # Find the norm squared of the difference.
            r2_mat = np.apply_along_axis(norm, 2, rlp_diff)**2
            H = np.zeros(np.shape(r2_mat))
            for i in range(1,len(self.form_factors)):
                if self.form_factors[i] == 0.:
                    continue
                else:
                    H[np.isclose(r2_mat, self.energy_shells[i])] = self.form_factors[i]
                
            return np.sort(np.linalg.eigvalsh(H + diag))[:neigvals]*Ry_to_eV

        else:
            diag = np.eye(len(self.rlat_pts))*list(map(lambda x: np.dot(x,x),
                                                       self.rlat_pts + kpoint))
            if np.allclose(self.atom_positions, [[0.]*3]):
                H = self.init_hamiltonian + diag*Ry_to_eV
            else:
                # For some reason the lattice points need to be double nested for
                # numpy.tile to work.
                nested_rlatpts = np.array([[rlp] for rlp in self.rlat_pts])
                # Create a matrix of reciprocal lattice points where the first
                # lattice point is repeated acroass the first row, the second
                # lattice point the second row, and so on.
                rlatpt_mat = np.tile(nested_rlatpts,(len(self.rlat_pts),1))
                # Create a matrix of the differences of the lattice points. Each
                # element is given by a_i - a_j.
                rlp_diff = rlatpt_mat - np.transpose(rlatpt_mat, (1,0,2))
                # Calculate the phase portion of the Hamiltonian matrix elements.
                phase_mat = np.dot(rlp_diff, np.sum(self.atom_positions,0))                
                H = self.init_hamiltonian*np.exp(-1j*phase_mat) + diag*Ry_to_eV
            return np.sort(np.linalg.eigvalsh(H))[:neigvals]
        
    def hamiltonian(self, kpoint):
        """Evaluate the empirical pseudopotential Hamiltonian at the provided
        k-point. This function is typically used to verify the Hamiltonian is 
        Hermitian.
        """
        
        # Calculate the diagonal elements of the Hamiltonian.
        diag = np.eye(len(self.rlat_pts))*list(map(lambda x: np.dot(x,x),
                                                   self.rlat_pts + kpoint))
        
        # For some reason the lattice points need to be double nested for
        # numpy.tile to work.
        nested_rlatpts = np.array([[rlp] for rlp in self.rlat_pts])
        # Create a matrix of reciprocal lattice points where the first lattice
        # point is repeated acroass the first row, the second lattice point the
        # second row, and so on.
        rlatpt_mat = np.tile(nested_rlatpts,(len(self.rlat_pts),1))
        # Create a matrix of the differences of the lattice points. Each element
        # is given by a_i - a_j.
        rlp_diff = rlatpt_mat - np.transpose(rlatpt_mat, (1,0,2))
        # Find the norm squared of the difference.
        r2_mat = np.apply_along_axis(norm, 2, rlp_diff)**2
        H = np.zeros(np.shape(r2_mat))
        for i in range(1,len(self.form_factors)):
            if self.form_factors[i] == 0.:
                continue
            else:
                H[np.isclose(r2_mat, self.energy_shells[i])] = self.form_factors[i]
        return (H + diag)*Ry_to_eV


class CohenEmpiricalPseudopotential(object):
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
        atom_labels (list): a list of atom labels. The first atom type is 0,
            the following atom labels should be 1, 2, and so on.
        atom_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons.
        material (str): a string describing the empirical pseudopotential,
            such as the chemical formula.
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
        atom_positions (list): a list of atomic positions.
        nvalence_electrons (int): the number of valence electrons.
        material (str): a string describing the empirical pseudopotential,
            such as the chemical formula.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_energy (float): the total energy.
    """

    def __init__(self, lattice, sym_form_factors, antisym_form_factors,
                 energy_cutoff, atom_labels, atom_positions, nvalence_electrons,
                 material, energy_shift=None, fermi_level=None,
                 total_energy=None):

        self.material = material
        self.lattice = lattice
        self.sym_form_factors = sym_form_factors
        self.antisym_form_factors = antisym_form_factors
        self.energy_cutoff = energy_cutoff
        self.atom_labels = atom_labels
        if np.shape(atom_positions) == (3,):
            msg = ("Please provide a list of atomic positions instead of an "
                   "individual atomic position.")
            raise ValueError(msg.format(atom_positions))
        else:
            self.atom_positions = atom_positions
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                    self.energy_cutoff)
        self.find_energy_shells()
        self.energy_cutoff = self.energy_shells[len(self.sym_form_factors)]
        self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
                                   self.energy_cutoff)
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        self.total_energy = total_energy or 0.

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
                    
    # The version of eval that I'm fixing.
    def eval(self, kpoint, neigvals):
        """Evaluate the empirical pseudopotential Hamiltonian at the provided
        k-point. This function is typically used to verify the Hamiltonian is 
        Hermitian.
        """

        # self.rlat_pts = sphere_pts(self.lattice.reciprocal_vectors,
        #                            self.energy_cutoff, -np.array(kpoint))
                
        # Calculate the diagonal elements of the Hamiltonian.
        diag = np.eye(len(self.rlat_pts))*np.apply_along_axis(norm, 1,
                                                              self.rlat_pts + kpoint)**2
        # For some reason the lattice points need to be double nested for
        # numpy.tile to work.
        nested_rlatpts = np.array([[rlp] for rlp in self.rlat_pts])

        # Create a matrix of reciprocal lattice points where the first lattice
        # point is repeated acroass the first row, the second lattice point the
        # second row, and so on.
        rlatpt_mat = np.tile(nested_rlatpts,(len(self.rlat_pts),1))
        
        # Create a matrix of the differences of the lattice points. Each element
        # is given by a_i - a_j.
        rlp_diff = rlatpt_mat - np.transpose(rlatpt_mat, (1,0,2))

        # Find the norm squared of the difference.
        r2_mat = np.apply_along_axis(norm, 2, rlp_diff)**2

        # The symmetric part of the Hamiltonian.
        sff = np.zeros(np.shape(r2_mat))
        for i in range(1,len(self.sym_form_factors)):
            if self.sym_form_factors[i] == 0.:
                continue
            else:
                sff[np.isclose(r2_mat, self.energy_shells[i])] = self.sym_form_factors[i]
                
        sff *= np.cos(np.sum(rlp_diff*np.sum(self.atom_positions, 0), 2))

        # The Anti-symmetric part of the Hamiltonian.
        asff = np.zeros(np.shape(r2_mat), dtype=complex)
        for i in range(1,len(self.antisym_form_factors)):
            if self.antisym_form_factors[i] == 0.:
                continue
            else:
                asff[np.isclose(r2_mat, self.energy_shells[i])] = self.antisym_form_factors[i]
                
        asff *= 1j*np.sin(np.sum(rlp_diff*np.sum(self.atom_positions, 0), 2))

        H = (diag + sff + asff)
        return np.sort(np.linalg.eigvalsh(H))[:neigvals]*Ry_to_eV
    
    def hamiltonian(self, kpoint):
        """Evaluate the empirical pseudopotential Hamiltonian at the provided
        k-point. This function is typically used to verify the Hamiltonian is 
        Hermitian.
        """
        # Calculate the diagonal elements of the Hamiltonian.
        diag = np.eye(len(self.rlat_pts))*np.apply_along_axis(norm, 1,
                                                              self.rlat_pts + kpoint)**2
        # For some reason the lattice points need to be double nested for
        # numpy.tile to work.
        nested_rlatpts = np.array([[rlp] for rlp in self.rlat_pts])
        # Create a matrix of reciprocal lattice points where the first lattice
        # point is repeated acroass the first row, the second lattice point the
        # second row, and so on.
        rlatpt_mat = np.tile(nested_rlatpts,(len(self.rlat_pts),1))
        # Create a matrix of the differences of the lattice points. Each element
        # is given by a_i - a_j.
        rlp_diff = rlatpt_mat - np.transpose(rlatpt_mat, (1,0,2))
        # Find the norm squared of the difference.
        r2_mat = np.apply_along_axis(norm, 2, rlp_diff)**2

        # The symmetry part of the Hamiltonian.
        sff = np.zeros(np.shape(r2_mat))
        for i in range(1,len(self.sym_form_factors)):
            if self.sym_form_factors[i] == 0.:
                continue
            else:
                sff[np.isclose(r2_mat, self.energy_shells[i])] = self.sym_form_factors[i]
                
        sff *= np.cos(np.sum(rlp_diff*np.sum(self.atom_positions, 0), 2))

        # The anti-symmetry part of the Hamiltonian.
        asff = np.zeros(np.shape(r2_mat), dtype=complex)
        for i in range(1,len(self.antisym_form_factors)):
            if self.antisym_form_factors[i] == 0.:
                continue
            else:
                asff[np.isclose(r2_mat, self.energy_shells[i])] = self.antisym_form_factors[i]
                
        asff *= 1j*np.sin(np.sum(rlp_diff*np.sum(self.atom_positions, 0), 2))
        
        return (diag + sff + asff)*Ry_to_eV


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
        total_enery(float): the total energy

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons is 1.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        fermi_level_ans (float): the exact, analytical value for the Fermi 
            level.
        total_enery (float): the total energy
        total_enery_ans (float): the exact, analytical value for the total
            energy.
    """
    
    def __init__(self, lattice, degree, energy_shift=None,
                 fermi_level=None, total_energy=None):
        self.lattice = lattice
        self.degree = degree
        # The free electron pseudopotential can only have one valence electron
        # because the energy dispersion relation isn't periodic and with two
        # valence electrons the Fermi surface would extend outside the unit 
        # cell.
        self.nvalence_electrons = 1
        self.atom_labels = [0]
        self.atom_positions = [[0]*3]
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        self.name = "free electron model"
        # occupied_volume = self.lattice.reciprocal_volume*self.nvalence_electrons/2.
        # self.fermi_level_ans = (3*occupied_volume/(4*np.pi))**(self.degree/3.)
        
        nfilled_states = self.nvalence_electrons/2.
        
        # self.fermi_level_ans = (3*nfilled_states/(4*np.pi))**(self.degree/3.)
        # self.fermi_level_ans = (3*nfilled_states/(4*np.pi*c))**(self.degree/3.)
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))
        
        self.total_energy = total_energy or 0.
        # rf = self.fermi_level_ans**(1./degree)
        # self.total_energy_ans = 4*np.pi/(3. + self.degree)*rf**(3. + self.degree)

        
    def eval(self, kpoint, neigvals):
        # There's only one eigenvalue so neigvals isn't needed in general but
        # it is when running tests on functions that take an instance of the
        # pseudopotential classes.

        return [np.linalg.norm(kpoint)**self.degree]

    def set_degree(self, degree):
        self.degree = degree
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))
        
    def number_of_states(self, energy):
        """Evaluate the exact number of states for the free electron model.
        """
        return energy**(3./self.degree)/(3*np.pi**2)

    def density_of_states(self, energy):
        """Evaluate the exact density of states for the free electron model.
        """
        return energy**((3. - self.degree)/self.degree)/(self.degree*np.pi**2)
        
class SingleFreeElectronModel():
    """This is the popular free electron model. In this model the potential is
    zero everywhere. It is useful for testing.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        Fermi_level (float): the fermi level.
        total_enery(float): the total energy

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons is 1.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        fermi_level_ans (float): the exact, analytical value for the Fermi 
            level.
        total_enery (float): the total energy
        total_enery_ans (float): the exact, analytical value for the total
            energy.
        material (str): the name of the pseudopotential.

    """
    
    def __init__(self, lattice, degree, energy_shift=None,
                 fermi_level=None, total_energy=None):
        self.material = "Free electron model"
        self.lattice = lattice
        self.degree = degree
        # The free electron pseudopotential can only have one valence electron
        # because the energy dispersion relation isn't periodic and with two
        # valence electrons the Fermi surface would extend outside the unit 
        # cell.
        # self.name = "free electron model"
        self.nvalence_electrons = 1
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.
        nfilled_states = self.nvalence_electrons/2.
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))        
        self.total_energy = total_energy or 0.

    def eval(self, kpoint, neigvals):
        # There's only one eigenvalue so neigvals isn't needed in general but
        # it is when running tests on functions that take an instance of the
        # pseudopotential classes.
        
        return [np.linalg.norm(kpoint)**self.degree]

    def set_degree(self, degree):
        self.degree = degree
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))

class MultipleFreeElectronModel():
    """This is the popular free electron model. In this model the potential is
    zero everywhere. It is useful for testing.

    Args:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        total_enery(float): the total energy

    Attributes:
        lattice (:py:obj:`BZI.symmetry.lattice`): an instance of Lattice.
        nvalence_electrons (int): the number of valence electrons is 1.
        degree (int): the degree of the radial dispersion relation.
        energy_shift (float): an energy shift typically used to place the Fermi
            level at the correct position.
        fermi_level (float): the fermi level.
        fermi_level_ans (float): the exact, analytical value for the Fermi 
            level.
        total_enery (float): the total energy
        total_enery_ans (float): the exact, analytical value for the total
            energy.
        material (str): the material or model name.
    """
    
    def __init__(self, lattice, degree, nvalence_electrons, energy_shift=None,
                 fermi_level=None, total_energy=None):
        self.material = "Free electron model"
        self.lattice = lattice
        self.degree = degree
        self.nvalence_electrons = nvalence_electrons
        self.energy_shift = energy_shift or 0.
        self.fermi_level = fermi_level or 0.        
        nfilled_states = self.nvalence_electrons/2.        
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))
        
        self.total_energy = total_energy or 0.
        
    def eval(self, kpoint, neigvals):
        
        kpoint = np.array(kpoint)
        l0 = np.linalg.norm(self.lattice.reciprocal_vectors[:,0])
        l1 = np.linalg.norm(self.lattice.reciprocal_vectors[:,1])
        l2 = np.linalg.norm(self.lattice.reciprocal_vectors[:,2])
        
        pts = np.array([[0,0,0], [-l0, 0, 0], [l0, 0, 0],
                        [0, -l1, 0], [0, l1, 0],
                        [0, 0, -l2], [0, 0, l2],
                        [l0, l1, 0], [l0, -l1, 0],
                        [-l0, l1, 0], [-l0, -l1, 0],
                        [l0, 0, l2], [l0, 0, -l2],
                        [-l0, 0, l2], [-l0, 0, -l2],
                        [0, l1, l2], [0, l1, -l2],
                        [0, -l1, l2], [0, -l1, -l2]])

        return [np.linalg.norm(kpoint - pt)**self.degree for pt in pts][:neigvals]
        
    def set_degree(self, degree):
        self.degree = degree
        self.fermi_level_ans = (3*np.pi**2*self.nvalence_electrons)**(self.degree/3.)
        self.total_energy_ans = ((4.*np.pi*(3.*np.pi**2*self.nvalence_electrons)**
                                  ((self.degree + 3.)/3.))/(self.degree + 3.))
                
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
# Pseudopotential from Band Structures and Pseudopotential Form Factors for
# Fourteen Semiconductors of the Diamond. and. Zinc-blende Structures* by
# Cohen and Bergstresser
####

# Define the pseudopotential form factors taken from Cohen and Bergstesser
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

# Here the pseudopotential form factors are reorganized into symmetry and anti-symmetric
# terms in order to be easier to work with.
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
Si_lattice = Lattice(Si_lat_centering, Si_lat_consts, Si_lat_angles,
                     convention="angular")

# Si_energy_cutoff = (11 + 1)*(2*np.pi/Si_lat_const)**2
Si_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Si_lattice.reciprocal_vectors.T)))*4
Si_atom_positions = [Si_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Si_atom_labels = np.zeros(len(Si_atom_positions))
Si_nvalence_electrons = 4
Si_EPM = CohenEmpiricalPseudopotential(Si_lattice, Si_spff, Si_apff, Si_energy_cutoff,
                                       Si_atom_labels, Si_atom_positions,
                                       Si_nvalence_electrons, material="Si")

#### Pseudopotential of Ge ####
Ge_lat_centering = "face"
Ge_lat_const = 5.66*angstrom_to_Bohr # the lattice constant in Bohr
Ge_lat_consts = [Ge_lat_const]*3
Ge_lat_angles = [np.pi/2]*3
Ge_lattice = Lattice(Ge_lat_centering, Ge_lat_consts, Ge_lat_angles,
                     convention="angular")

Ge_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Ge_lattice.reciprocal_vectors.T)))*4
Ge_atom_positions = [Ge_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
Ge_atom_labels = np.zeros(len(Ge_atom_positions))
Ge_nvalence_electrons = 4
Ge_EPM = CohenEmpiricalPseudopotential(Ge_lattice, Ge_spff, Ge_apff, Ge_energy_cutoff,
                                       Ge_atom_labels, Ge_atom_positions,
                                       Ge_nvalence_electrons, material="Ge")

#### Pseudopotential of Sn ####
cSn_lat_centering = "face"
cSn_lat_const = 6.49*angstrom_to_Bohr # the lattice constant in Bohr
cSn_lat_consts = [cSn_lat_const]*3
cSn_lat_angles = [np.pi/2]*3
cSn_lattice = Lattice(cSn_lat_centering, cSn_lat_consts, cSn_lat_angles,
                      convention="angular")

cSn_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                 cSn_lattice.reciprocal_vectors.T)))*4
cSn_atom_positions = [cSn_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
cSn_atom_labels = np.zeros(len(cSn_atom_positions))
cSn_nvalence_electrons = 4
cSn_EPM = CohenEmpiricalPseudopotential(cSn_lattice, Sn_spff, Sn_apff, cSn_energy_cutoff,
                                        cSn_atom_labels, cSn_atom_positions,
                                        cSn_nvalence_electrons, material="Sn")

#### Pseudopotential of GaP ####
GaP_lat_centering = "face"
GaP_lat_const = 5.44*angstrom_to_Bohr # the lattice constant in Bohr
GaP_lat_consts = [GaP_lat_const]*3
GaP_lat_angles = [np.pi/2]*3
GaP_lattice = Lattice(GaP_lat_centering, GaP_lat_consts, GaP_lat_angles,
                      convention="angular")

GaP_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                 GaP_lattice.reciprocal_vectors.T)))*4
GaP_atom_positions = [GaP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaP_atom_labels = np.zeros(len(GaP_atom_positions))
GaP_nvalence_electrons = 8
GaP_EPM = CohenEmpiricalPseudopotential(GaP_lattice, GaP_spff, GaP_apff, GaP_energy_cutoff,
                                        GaP_atom_labels, GaP_atom_positions,
                                        GaP_nvalence_electrons, material="GaP")

#### Pseudopotential of GaAs ####
GaAs_lat_centering = "face"
GaAs_lat_const = 5.64*angstrom_to_Bohr # the lattice constant in Bohr
GaAs_lat_consts = [GaAs_lat_const]*3
GaAs_lat_angles = [np.pi/2]*3
GaAs_lattice = Lattice(GaAs_lat_centering, GaAs_lat_consts, GaAs_lat_angles,
                       convention="angular")

GaAs_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  GaAs_lattice.reciprocal_vectors.T)))*4
GaAs_atom_positions = [GaAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaAs_atom_labels = np.zeros(len(GaAs_atom_positions))
GaAs_nvalence_electrons = 8
GaAs_EPM = CohenEmpiricalPseudopotential(GaAs_lattice, GaAs_spff, GaAs_apff,
                                         GaAs_energy_cutoff, GaAs_atom_labels,
                                         GaAs_atom_positions, GaAs_nvalence_electrons,
                                         material="GaAs")

#### Pseudopotential of AlSb ####
AlSb_lat_centering = "face"
AlSb_lat_const = 6.13*angstrom_to_Bohr # the lattice constant in Bohr
AlSb_lat_consts = [AlSb_lat_const]*3
AlSb_lat_angles = [np.pi/2]*3
AlSb_lattice = Lattice(AlSb_lat_centering, AlSb_lat_consts, AlSb_lat_angles,
                       convention="angular")

AlSb_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  AlSb_lattice.reciprocal_vectors.T)))*4
AlSb_atom_positions = [AlSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
AlSb_atom_labels = np.zeros(len(AlSb_atom_positions))
AlSb_nvalence_electrons = 8
AlSb_EPM = CohenEmpiricalPseudopotential(AlSb_lattice, AlSb_spff, AlSb_apff,
                                         AlSb_energy_cutoff, AlSb_atom_labels,
                                         AlSb_atom_positions, AlSb_nvalence_electrons,
                                         material="AlSb")

#### Pseudopotential of InP ####
InP_lat_centering = "face"
InP_lat_const = 5.86*angstrom_to_Bohr # the lattice constant in Bohr
InP_lat_consts = [InP_lat_const]*3
InP_lat_angles = [np.pi/2]*3
InP_lattice = Lattice(InP_lat_centering, InP_lat_consts, InP_lat_angles,
                      convention="angular")


InP_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                 InP_lattice.reciprocal_vectors.T)))*4
InP_atom_positions = [InP_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InP_atom_labels = np.zeros(len(InP_atom_positions))
InP_nvalence_electrons = 8
InP_EPM = CohenEmpiricalPseudopotential(InP_lattice, InP_spff, InP_apff,
                                        InP_energy_cutoff, InP_atom_labels,
                                        InP_atom_positions, InP_nvalence_electrons,
                                        material="InP")

#### Pseudopotential of GaSb ####
GaSb_lat_centering = "face"
GaSb_lat_const = 6.12*angstrom_to_Bohr # the lattice constant in Bohr
GaSb_lat_consts = [GaSb_lat_const]*3
GaSb_lat_angles = [np.pi/2]*3
GaSb_lattice = Lattice(GaSb_lat_centering, GaSb_lat_consts, GaSb_lat_angles,
                       convention="angular")

GaSb_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  GaSb_lattice.reciprocal_vectors.T)))*4
GaSb_atom_positions = [GaSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
GaSb_atom_labels = np.zeros(len(GaSb_atom_positions))
GaSb_nvalence_electrons = 8
GaSb_EPM = CohenEmpiricalPseudopotential(GaSb_lattice, GaSb_spff, GaSb_apff,
                                         GaSb_energy_cutoff,GaSb_atom_labels, 
                                         GaSb_atom_positions, GaSb_nvalence_electrons,
                                         material="GaSb")

#### Pseudopotential of InAs ####
InAs_lat_centering = "face"
InAs_lat_const = 6.04*angstrom_to_Bohr # the lattice constant in Bohr
InAs_lat_consts = [InAs_lat_const]*3
InAs_lat_angles = [np.pi/2]*3
InAs_lattice = Lattice(InAs_lat_centering, InAs_lat_consts, InAs_lat_angles,
                       convention="angular")

InAs_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  GaSb_lattice.reciprocal_vectors.T)))*5
InAs_atom_positions = [InAs_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InAs_atom_labels = np.zeros(len(InAs_atom_positions))
InAs_nvalence_electrons = 8
InAs_EPM = CohenEmpiricalPseudopotential(InAs_lattice, InAs_spff, InAs_apff,
                                         InAs_energy_cutoff, InAs_atom_labels,
                                         InAs_atom_positions, InAs_nvalence_electrons,
                                         material="InAs")

#### Pseudopotential of InSb ####
InSb_lat_centering = "face"
InSb_lat_const = 6.48*angstrom_to_Bohr # the lattice constant in Bohr
InSb_lat_consts = [InSb_lat_const]*3
InSb_lat_angles = [np.pi/2]*3
InSb_lattice = Lattice(InSb_lat_centering, InSb_lat_consts, InSb_lat_angles,
                       convention="angular")

InSb_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  InSb_lattice.reciprocal_vectors.T)))*4
InSb_atom_positions = [InSb_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
InSb_atom_labels = np.zeros(len(InSb_atom_positions))
InSb_nvalence_electrons = 8
InSb_EPM = CohenEmpiricalPseudopotential(InSb_lattice, InSb_spff, InSb_apff,
                                         InSb_energy_cutoff, InSb_atom_labels,
                                         InSb_atom_positions, InSb_nvalence_electrons,
                                         material="InSb")

#### Pseudopotential of ZnS ####
ZnS_lat_centering = "face"
ZnS_lat_const = 5.41*angstrom_to_Bohr # the lattice constant in Bohr
ZnS_lat_consts = [ZnS_lat_const]*3
ZnS_lat_angles = [np.pi/2]*3
ZnS_lattice = Lattice(ZnS_lat_centering, ZnS_lat_consts, ZnS_lat_angles,
                      convention="angular")

ZnS_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                 ZnS_lattice.reciprocal_vectors.T)))*4
ZnS_atom_positions = [ZnS_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnS_atom_labels = np.zeros(len(ZnS_atom_positions))
ZnS_nvalence_electrons = 18
ZnS_EPM = CohenEmpiricalPseudopotential(ZnS_lattice, ZnS_spff, ZnS_apff,
                                        ZnS_energy_cutoff, ZnS_atom_labels,
                                        ZnS_atom_positions, ZnS_nvalence_electrons,
                                        material="ZnS")

#### Pseudopotential of ZnSe ####
ZnSe_lat_centering = "face"
ZnSe_lat_const = 5.65*angstrom_to_Bohr # the lattice constant in Bohr
ZnSe_lat_consts = [ZnSe_lat_const]*3
ZnSe_lat_angles = [np.pi/2]*3
ZnSe_lattice = Lattice(ZnSe_lat_centering, ZnSe_lat_consts, ZnSe_lat_angles,
                       convention="angular")

ZnSe_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  ZnSe_lattice.reciprocal_vectors.T)))*4
ZnSe_atom_positions = [ZnSe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnSe_atom_labels = np.zeros(len(ZnSe_atom_positions))
ZnSe_nvalence_electrons = 18
ZnSe_EPM = CohenEmpiricalPseudopotential(ZnSe_lattice, ZnSe_spff, ZnSe_apff,
                                         ZnSe_energy_cutoff, ZnSe_atom_labels,
                                         ZnSe_atom_positions, ZnSe_nvalence_electrons,
                                         material="ZnSe")

#### Pseudopotential of ZnTe ####
ZnTe_lat_centering = "face"
ZnTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
ZnTe_lat_consts = [ZnTe_lat_const]*3
ZnTe_lat_angles = [np.pi/2]*3
ZnTe_lattice = Lattice(ZnTe_lat_centering, ZnTe_lat_consts, ZnTe_lat_angles,
                       convention="angular")


ZnTe_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  ZnTe_lattice.reciprocal_vectors.T)))*4
ZnTe_atom_positions = [ZnTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
ZnTe_atom_labels = np.zeros(len(ZnTe_atom_positions))
ZnTe_nvalence_electrons = 18
ZnTe_EPM = CohenEmpiricalPseudopotential(ZnTe_lattice, ZnTe_spff, ZnTe_apff,
                                         ZnTe_energy_cutoff, ZnTe_atom_labels,
                                         ZnTe_atom_positions, ZnTe_nvalence_electrons,
                                         material="ZnTe")

#### Pseudopotential of CdTe ####
CdTe_lat_centering = "face"
CdTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
CdTe_lat_consts = [CdTe_lat_const]*3
CdTe_lat_angles = [np.pi/2]*3
CdTe_lattice = Lattice(CdTe_lat_centering, CdTe_lat_consts, CdTe_lat_angles,
                       convention="angular")


CdTe_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                  CdTe_lattice.reciprocal_vectors.T)))*4
CdTe_atom_positions = [CdTe_lat_const/8.*np.array([1,1,1])] # one atomic basis vector
CdTe_atom_labels = np.zeros(len(CdTe_atom_positions))
CdTe_nvalence_electrons = 18
CdTe_EPM = CohenEmpiricalPseudopotential(CdTe_lattice, CdTe_spff, CdTe_apff,
                                         CdTe_energy_cutoff, CdTe_atom_labels,
                                         CdTe_atom_positions, CdTe_nvalence_electrons,
                                         material="CdTe")

# # The end of Cohen's 14 pseudopotentials with diamond or zinc-blende structure.

#### Toy Pseudopotential ####
Toy_lat_centering = "prim"
Toy_lat_const = 1.
Toy_lat_consts = [Toy_lat_const]*3
Toy_lat_angles = [np.pi/2]*3
Toy_lattice = Lattice(Toy_lat_centering, Toy_lat_consts, Toy_lat_angles)

Toy_pff = [0.0]
Toy_energy_cutoff = 2*(2*np.pi/Toy_lat_const)**2
Toy_atom_positions = [[0.]*3]
Toy_atom_labels = np.zeros(len(Toy_atom_positions))
Toy_nvalence_electrons = 3
Toy_EPM = EmpiricalPseudopotential(Toy_lattice, Toy_pff, Toy_energy_cutoff,
                                   Toy_atom_labels, Toy_atom_positions,
                                   Toy_nvalence_electrons, material="Toy")

#### Free electron Pseudopotential ####
free_lat_centering = "prim"
free_lat_const = 1.
free_lat_consts = [free_lat_const]*3
free_lat_angles = [np.pi/2]*3
free_lattice = Lattice(free_lat_centering, free_lat_consts, free_lat_angles,
                       convention="angular")

free_pff = [0.0]
free_energy_cutoff = 2*(2*np.pi/free_lat_const)**2
free_atom_positions = [[0.]*3]
free_nvalence_electrons = 1
free_degree = 2
free_EPM = FreeElectronModel(free_lattice, free_degree)

#### Single Free electron Pseudopotential ####
single_free_lat_centering = "prim"
single_free_lat_const = 1.
single_free_lat_consts = [single_free_lat_const]*3
single_free_lat_angles = [np.pi/2]*3
single_free_lattice = Lattice(single_free_lat_centering,
                              single_free_lat_consts, single_free_lat_angles,
                              convention="angular")

single_free_pff = [0.0]
single_free_energy_cutoff = 2*(2*np.pi/single_free_lat_const)**2
single_free_atom_positions = [[0.]*3]
single_free_nvalence_electrons = 1
single_free_degree = 2
single_free_EPM = SingleFreeElectronModel(single_free_lattice, single_free_degree)

#### Multiple Free electron Pseudopotential ####
multiple_free_lat_centering = "prim"
multiple_free_lat_const = 1.
multiple_free_lat_consts = [multiple_free_lat_const]*3
multiple_free_lat_angles = [np.pi/2]*3
multiple_free_lattice = Lattice(multiple_free_lat_centering,
                                multiple_free_lat_consts, multiple_free_lat_angles,
                                convention="angular")

multiple_free_pff = [0.0]
multiple_free_energy_cutoff = 2*(2*np.pi/multiple_free_lat_const)**2
multiple_free_atom_positions = [[0.]*3]
multiple_free_nvalence_electrons = 2
multiple_free_degree = 2
multiple_free_EPM = MultipleFreeElectronModel(multiple_free_lattice,
                                             multiple_free_degree,
                                             multiple_free_nvalence_electrons)


# # The following pseudopotentials come from: 
# # Marvin L. Cohen and Volker Heine. "The fitting of pseudopotentials to
# # experimental data and their subsequent application." Solid state physics 24
# # (1970): 37-248. APA

#### Pseudopotential of Al ####
Al_centering_type = "face"
Al_lat_const = 4.05*angstrom_to_Bohr
Al_lat_consts = [Al_lat_const]*3
Al_lat_angles = [np.pi/2]*3
Al_lattice = Lattice(Al_centering_type, Al_lat_consts, Al_lat_angles,
                     convention="angular")

Al_pff = [0.0, 0.0179, 0.0562]
# Take the energy cutof as 3x the length of the longest reciprocal lattice vector squared.
# This value isn't very important because the energy cutoff changes in the empirical pseudopotential.
Al_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Al_lattice.reciprocal_vectors.T)))*3
Al_atom_positions = [[0.,0.,0.]]
Al_atom_labels = np.zeros(len(Al_atom_positions))
Al_nvalence_electrons = 3
Al_EPM = EmpiricalPseudopotential(Al_lattice, Al_pff, Al_energy_cutoff, Al_atom_labels,
                                  Al_atom_positions, Al_nvalence_electrons, material="Al")

#### Pseudopotential of Li ####
Li_centering_type = "body"
Li_lat_const = 3.51*angstrom_to_Bohr # From Materials Project
Li_lat_consts = [Li_lat_const]*3
Li_lat_angles = [np.pi/2]*3
Li_lattice = Lattice(Li_centering_type, Li_lat_consts, Li_lat_angles)
Li_lat_consts = [Li_lat_const]*3
Li_lat_angles = [np.pi/2]*3
Li_lattice = Lattice(Li_centering_type, Li_lat_consts, Li_lat_angles,
                     convention="angular")

Li_pff = [0.0, 0.11]
Li_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Li_lattice.reciprocal_vectors.T)))*3
Li_atom_positions = [[0.,0.,0.]]
Li_atom_labels = np.zeros(len(Li_atom_positions))
Li_nvalence_electrons = 1
Li_EPM = EmpiricalPseudopotential(Li_lattice, Li_pff, Li_energy_cutoff, Li_atom_labels,
                                  Li_atom_positions, Li_nvalence_electrons, material="Li")

#### Pseudopotential of Na ####
Na_centering_type = "body"
# Na_lat_const = 3.633*angstrom_to_Bohr
Na_lat_const = 4.2906*angstrom_to_Bohr
Na_lat_consts = [Na_lat_const]*3
Na_lat_angles = [np.pi/2]*3
Na_lattice = Lattice(Na_centering_type, Na_lat_consts, Na_lat_angles,
                     convention="angular")

Na_pff = [0.0, 0.0158, 0.0]
Na_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Na_lattice.reciprocal_vectors.T)))*3
Na_atom_positions = [[0.]*3]
Na_atom_labels = np.zeros(len(Na_atom_positions))
Na_nvalence_electrons = 1
Na_EPM = EmpiricalPseudopotential(Na_lattice, Na_pff, Na_energy_cutoff, Na_atom_labels,
                                  Na_atom_positions, Na_nvalence_electrons, material="Na")
    
#### Pseudopotential of K ####
K_centering_type = "body"
K_lat_const = 5.225*angstrom_to_Bohr
K_lat_consts = [K_lat_const]*3
K_lat_angles = [np.pi/2]*3
K_lattice = Lattice(K_centering_type, K_lat_consts, K_lat_angles,
                    convention="angular")

# The pseudopotential parameters come from the paper by Lee and Falicov titled
# The de Haas-van Alphen  effect  and the Fermi  surface  of potassium
# In it they say that the three parameter pseudopotential isn't adequate to
# accurately fit the Fermi surface and go on to add another term to there
# pseudopotential. We thought that the three parameter was good enough for our
# purposes.
K_pff = [0.0, 0.22/Ry_to_eV, -0.89/Ry_to_eV, 0.55/Ry_to_eV]
K_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                               K_lattice.reciprocal_vectors.T)))*4
K_atom_positions = [[0.]*3]
K_atom_labels = np.zeros(len(K_atom_positions))
K_nvalence_electrons = 1
K_EPM = EmpiricalPseudopotential(K_lattice, K_pff, K_energy_cutoff, K_atom_labels,
                                 K_atom_positions, K_nvalence_electrons, material="K")

#### Pseudopotential of Rb ####
Rb_centering_type = "body"
Rb_lat_const = 5.585*angstrom_to_Bohr # Materials Project
Rb_lat_consts = [Rb_lat_const]*3
Rb_lat_angles = [np.pi/2]*3
Rb_lattice = Lattice(Rb_centering_type, Rb_lat_consts, Rb_lat_angles,
                     convention="angular")

# Again, Rb is a very rough estimate. Values are taken from the original paper
# by Lee titled The de Haas-van Alphen effect and the Fermi surface of sodium.
# There was a large discrepancy between one quantity and experiment. They said
# This could be fixed by making the last V211 = 0.4 eV. This make the band
# structure better match results on materialsproject.org for bcc Rb so it was
# kept.
# V200 = -0.3 is within the error bounds they gave and also made the band
# structure look more accurate.
Rb_pff = [0.0, 0.225/Ry_to_eV, -0.3/Ry_to_eV, 0.4/Ry_to_eV]
Rb_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Rb_lattice.reciprocal_vectors.T)))*4
Rb_atom_positions = [[0.]*3]
Rb_atom_labels = np.zeros(len(Rb_atom_positions))
Rb_nvalence_electrons = 1
Rb_EPM = EmpiricalPseudopotential(Rb_lattice, Rb_pff, Rb_energy_cutoff, Rb_atom_labels,
                                  Rb_atom_positions, Rb_nvalence_electrons, material="Rb")

#### Pseudopotential of Cs ####
Cs_centering_type = "body"
Cs_lat_const = 6.141*angstrom_to_Bohr
Cs_lat_consts = [Cs_lat_const]*3
Cs_lat_angles = [np.pi/2]*3
Cs_lattice = Lattice(Cs_centering_type, Cs_lat_consts, Cs_lat_angles,
                     convention="angular")

Cs_pff = [0.0, -0.03]
Cs_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Cs_lattice.reciprocal_vectors.T)))*4
Cs_atom_positions = [[0.]*3]
Cs_atom_labels = np.zeros(len(Cs_atom_positions))
Cs_nvalence_electrons = 1
Cs_EPM = EmpiricalPseudopotential(Cs_lattice, Cs_pff, Cs_energy_cutoff, Cs_atom_labels,
                                  Cs_atom_positions, Cs_nvalence_electrons, material="Cs")
    
#### Pseudopotential of Cu ####
Cu_centering_type = "face"
Cu_lat_const = 3.615*angstrom_to_Bohr
Cu_lat_consts = [Cu_lat_const]*3
Cu_lat_angles = [np.pi/2]*3
Cu_lattice = Lattice(Cu_centering_type, Cu_lat_consts, Cu_lat_angles,
                     convention="angular")

Cu_pff = [0.0, 0.264, 0.246]
Cu_energy_cutoff = (4+1)*(2*np.pi/Cu_lat_const)**2
Cu_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Cu_lattice.reciprocal_vectors.T)))*4
Cu_atom_positions = [[0.]*3]
Cu_atom_labels = np.zeros(len(Cu_atom_positions))
Cu_nvalence_electrons = 11
Cu_EPM = EmpiricalPseudopotential(Cu_lattice, Cu_pff, Cu_energy_cutoff, Cu_atom_labels,
                                  Cu_atom_positions, Cu_nvalence_electrons, material="Cu")

#### Pseudopotential of Ag ####
Ag_centering_type = "face"
Ag_lat_const = 4.0853*angstrom_to_Bohr
Ag_lat_consts = [Ag_lat_const]*3
Ag_lat_angles = [np.pi/2]*3
Ag_lattice = Lattice(Ag_centering_type, Ag_lat_consts, Ag_lat_angles,
                     convention="angular")

Ag_pff = [0.0, 0.204, 0.220]
Ag_energy_cutoff = (4+1)*(2*np.pi/Ag_lat_const)**2
Ag_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Ag_lattice.reciprocal_vectors.T)))*4
Ag_atom_positions = [[0.]*3]
Ag_atom_labels = np.zeros(len(Ag_atom_positions))
Ag_nvalence_electrons = 11
Ag_EPM = EmpiricalPseudopotential(Ag_lattice, Ag_pff, Ag_energy_cutoff, Ag_atom_labels,
                                  Ag_atom_positions, Ag_nvalence_electrons, material="Ag")

#### Pseudopotential of Au ####
Au_centering_type = "face"
Au_lat_const = 4.0782*angstrom_to_Bohr
Au_lat_consts = [Au_lat_const]*3
Au_lat_angles = [np.pi/2]*3
Au_lattice = Lattice(Au_centering_type, Au_lat_consts, Au_lat_angles,
                     convention="angular")

Au_pff = [0.0, 0.252, 0.152]
Au_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Au_lattice.reciprocal_vectors.T)))*4
Au_atom_positions = [[0.]*3]
Au_atom_labels = np.zeros(len(Au_atom_positions))
Au_nvalence_electrons = 11
Au_EPM = EmpiricalPseudopotential(Au_lattice, Au_pff, Au_energy_cutoff, Au_atom_labels,
                                  Au_atom_positions, Au_nvalence_electrons, material="Au")

#### Pseudopotential of Pb ####
Pb_centering_type = "face"
Pb_lat_const = 4.9508*angstrom_to_Bohr
Pb_lat_consts = [Pb_lat_const]*3
Pb_lat_angles = [np.pi/2]*3
Pb_lattice = Lattice(Pb_centering_type, Pb_lat_consts, Pb_lat_angles,
                     convention="angular")

Pb_pff = [0.0, -0.084, -0.039]
Pb_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Pb_lattice.reciprocal_vectors.T)))*4
Pb_atom_positions = [[0.]*3]
Pb_atom_labels = np.zeros(len(Pb_atom_positions))
Pb_nvalence_electrons = 4
Pb_EPM = EmpiricalPseudopotential(Pb_lattice, Pb_pff, Pb_energy_cutoff, Pb_atom_labels,
                                  Pb_atom_positions, Pb_nvalence_electrons, material="Pb")

#### Pseudopotential of Mg ####
Mg_centering_type = "prim"
Mg_lat_const_a = 3.184
Mg_lat_const_c = 5.249
Mg_lat_consts = np.array([Mg_lat_const_a, Mg_lat_const_a,
                          Mg_lat_const_c])*angstrom_to_Bohr
Mg_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Mg_lattice = Lattice(Mg_centering_type, Mg_lat_consts, Mg_lat_angles,
                     convention="angular")

Mg_pff = [0., 0., .026, 0., 0., 0., .014, .036, 0., 0., .058]
Mg_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Mg_lattice.reciprocal_vectors.T)))*5
Mg_atom_positions = [[0.]*3]
Mg_atom_labels = np.zeros(len(Mg_atom_positions))
Mg_nvalence_electrons = 2
Mg_EPM = EmpiricalPseudopotential(Mg_lattice, Mg_pff, Mg_energy_cutoff, Mg_atom_labels,
                                  Mg_atom_positions, Mg_nvalence_electrons, material="Mg")

#### Pseudopotential of Zn ####
Zn_centering_type = "prim"
Zn_lat_const_a = 2.627
Zn_lat_const_c = 5.207
Zn_lat_consts = np.array([Zn_lat_const_a, Zn_lat_const_a,
                          Zn_lat_const_c])*angstrom_to_Bohr
Zn_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Zn_lattice = Lattice(Zn_centering_type, Zn_lat_consts, Zn_lat_angles,
                     convention="angular")

Zn_pff = [0., -0.022, 0.02, 0.063, 0.0, 0.0, 0.0, 0.0, 0.0]
Zn_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Zn_lattice.reciprocal_vectors.T)))*4
Zn_atom_positions = [[0.]*3]
Zn_atom_labels = np.zeros(len(Zn_atom_positions))
Zn_nvalence_electrons = 12
Zn_EPM = EmpiricalPseudopotential(Zn_lattice, Zn_pff, Zn_energy_cutoff, Zn_atom_labels,
                                  Zn_atom_positions, Zn_nvalence_electrons, material="Zn")

# See Band structure and Fermi surface of Zinc and Cadmium by Stark and
# Falicov for Cd form factors.
#### Pseudopotential of Cd ####
Cd_centering_type = "prim"
Cd_lat_const_a = 2.9684
Cd_lat_const_c = 5.5261
Cd_lat_consts = np.array([Cd_lat_const_a, Cd_lat_const_a,
                          Cd_lat_const_c])*angstrom_to_Bohr
Cd_lat_angles = [np.pi/2, np.pi/2, 2*np.pi/3]
Cd_lattice = Lattice(Cd_centering_type, Cd_lat_consts, Cd_lat_angles,
                     convention="angular")

Cd_pff = [0., -0.017, 0., 0., 0., 0., 0., 0.0235, 0.029, 0., 0.03]
Cd_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Cd_lattice.reciprocal_vectors.T)))*4
Cd_atom_positions = [[0.]*3]
Cd_atom_labels = np.zeros(len(Cd_atom_positions))
Cd_nvalence_electrons = 12
Cd_EPM = EmpiricalPseudopotential(Cd_lattice, Cd_pff, Cd_energy_cutoff, Cd_atom_labels,
                                  Cd_atom_positions, Cd_nvalence_electrons, material="Cd")

#### Pseudopotential of Hg ####
Hg_centering_type = "prim"
Hg_lat_const = 2.9863*angstrom_to_Bohr
Hg_lat_consts = [Hg_lat_const]*3
Hg_lat_angles = [70.446*np.pi/180]*3
Hg_lattice = Lattice(Hg_centering_type, Hg_lat_consts, Hg_lat_angles,
                     convention="angular")

Hg_pff = [-0.018, 0.028, 0.028]
Hg_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Hg_lattice.reciprocal_vectors.T)))*4
Hg_atom_positions = [[0.]*3]
Hg_atom_labels = np.zeros(len(Hg_atom_positions))
Hg_nvalence_electrons = 12
Hg_EPM = EmpiricalPseudopotential(Hg_lattice, Hg_pff, Hg_energy_cutoff, Hg_atom_labels,
                                  Hg_atom_positions, Hg_nvalence_electrons, material="Hg")

#### Pseudopotential of In ####
In_centering_type = "body"
In_lat_const_a = 3.2992*angstrom_to_Bohr
In_lat_const_c = 4.9049*angstrom_to_Bohr
In_lat_consts = [In_lat_const_a, In_lat_const_a, In_lat_const_c]
In_lat_angles = [np.pi/2]*3
In_lattice = Lattice(In_centering_type, In_lat_consts, In_lat_angles,
                     convention="angular")

In_pff = [0., 0., 0., 0., 0., 0., 0., -0.020, 0., 0., 0., -0.047]
In_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                In_lattice.reciprocal_vectors.T)))*5
In_atom_positions = [[0.]*3]
In_atom_labels = np.zeros(len(In_atom_positions))
In_nvalence_electrons = 3
In_EPM = EmpiricalPseudopotential(In_lattice, In_pff, In_energy_cutoff, In_atom_labels,
                                  In_atom_positions, In_nvalence_electrons, material="In")

#### Pseudopotential of Sn ####
Sn_centering_type = "body"
Sn_lat_const_a = 3.47*angstrom_to_Bohr
Sn_lat_const_c = 4.87*angstrom_to_Bohr
Sn_lat_consts = [Sn_lat_const_a, Sn_lat_const_a, Sn_lat_const_c]
Sn_lat_angles = [np.pi/2]*3
Sn_lattice = Lattice(Sn_centering_type, Sn_lat_consts, Sn_lat_angles,
                     convention="angular")

Sn_pff = [0.]*5 + [-0.056, 0., -0.069] + [0.]*13 + [0.033, 0., 0.051]
Sn_energy_cutoff = max(list(map(lambda x: np.dot(x,x),
                                Sn_lattice.reciprocal_vectors.T)))*9
Sn_atom_positions = [[0.]*3]
Sn_atom_labels = np.zeros(len(Sn_atom_positions))
Sn_nvalence_electrons = 4
Sn_EPM = EmpiricalPseudopotential(Sn_lattice, Sn_pff, Sn_energy_cutoff, Sn_atom_labels,
                                  Sn_atom_positions, Sn_nvalence_electrons, material="Sn")
