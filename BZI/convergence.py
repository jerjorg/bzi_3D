import numpy as np
import matplotlib.pyplot as plt
import time
from BZI.symmetry import make_ptvecs
from BZI.sampling import make_grid
from BZI.pseudopots import Al_PP
from BZI.integration import monte_carlo
from BZI.plots import PlotMesh


class Convergence(object):
    """ Compare integrations of pseudo-potentials by creating convergence plots.

    Args:
        pseudo_potential (function): a pseudo-potential function taken from 
            BZI.pseudopots
        cutoff (float): the energy cutoff of the pseudo-potential
        cell_type (str): the geometry of the integration cell
        cell_constant (float): the size of the integration cell
        offset (list): a vector that offsets the grid from the origin and is 
            given in grid coordinates.
        grid_types (list): a list of grid types
        grid_constants (list): a list of grid constants
        integration_methods (list): a list of integration methods

    Attributes:
        pseudo_potential (function): a pseudo-potential function taken from 
            BZI.pseudopots
        cell_type (str): the geometry of the integration cell.
        cell_constant (float): the size of the integration cell.
        cell_vectors (np.ndarray): an array vectors as columns of a 3x3 numpy
            array that is used to create the cell
        grid_types (list): a list of grid types
        grid_constants (list): a list of grid constants
        integration_methods (list): a list of integration methods
        answer (float): the expected result of integration
        errors (list): a list of errors for each grid type
        nspts (list): a list of the number of sampling points for each grid type
        integrals (list): a list of integral value for each grid type and constant
        times (list): a list of the amount of time taken computing the grid 
            generation and integration.
    """
    
    def __init__(self, pseudo_potential=None, cutoff=None, cell_centering=None,
                 cell_constants=None, cell_angles=None, offset=None,
                 grid_types=None, grid_constants=None,
                 integration_methods=None, origin=None, random = None):
        self.pseudo_potential = pseudo_potential or Al_PP
        self.cutoff = cutoff or 4.
        self.cell_centering = cell_centering or "prim"
        self.cell_constants = cell_constants or [1.]*3
        self.cell_angles = cell_angles or [np.pi/2]*3
        self.cell_vectors = make_ptvecs(self.cell_centering, self.cell_constants,
                                        self.cell_angles)
        self.grid_centerings = grid_centerings or ["prim", "base", "body", "face"]
        self.grid_constants = grid_constants or [1/n for n in range(2,11)]
        self.offset = offset or [0.,0.,0.]
        # self.integration_methods = integration_methods or [rectangle_method]
        self.origin = origin or [0.,0.,0.]
        self.random = random or False

    def compare_grids(self, answer, plot=False, save=False):
        self.answer = answer
        if self.random:
            nm = len(self.grid_types)
            self.nspts = [[] for _ in range(nm + 1)]
            self.errors = [[] for _ in range(nm + 1)]
            self.integrals = [[] for _ in range(nm + 1)]
            self.times = [[] for _ in range(nm + 1)]

            npts_list = [2**n for n in range(8,14)]
            for npts in npts_list:
                time1 = time.time()
                integral = monte_carlo(self.pseudo_potential,
                                       self.cell_vectors,
                                       npts,
                                       self.cutoff)
                self.nspts[nm].append(npts)
                self.integrals[nm].append(integral)
                self.times[nm].append((time.time() - time1))
                self.errors[nm].append(np.abs(self.integrals[nm][-1] - answer))
        else:
            self.nspts = [[] for _ in range(len(self.grid_types))]
            self.errors = [[] for _ in range(len(self.grid_types))]
            self.integrals = [[] for _ in range(len(self.grid_types))]
            self.times = [[] for _ in range(len(self.grid_types))]
        integration_method = self.integration_methods[0]
        for (i,grid_centering) in enumerate(self.grid_centering_list):
            for grid_consts in self.grid_constants_list:
                for grid_angles in grid_angles_list:
                    grid_vecs = make_ptvecs(grid_centering, grid_consts, grid_angles)
                    time1 = time.time()
                    npts, integral = integration_method(self.pseudo_potential,
                                                                self.cell_vectors,
                                                                grid_vecs,
                                                                self.offset,
                                                                self.origin,
                                                                self.cutoff)
                self.nspts[i].append(npts)
                self.integrals[i].append(integral)
                self.times[i].append((time.time() - time1))
                self.errors[i].append(np.abs(self.integrals[i][-1] - answer))
                
        if save:
            np.save("%s_times" %self.pseudo_potential, self.times)
            np.save("%s_integrals" %self.pseudo_potential, self.integrals)
            np.save("%s_errors" %self.pseudo_potential, self.errors)
            
        if plot:
            if self.random:
                plt.loglog(self.nspts[nm], self.errors[nm], label="random", color="orange")
            for i in range(len(self.grid_types)):
                plt.loglog(self.nspts[i], self.errors[i], label=self.grid_types[i])
            plt.xlabel("Number of samping points")
            plt.ylabel("Error")
            test = [1./n**(2./3) for n in self.nspts[0]]
            plt.loglog(self.nspts[0], test, label="1/n**(2/3)")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.show()
            plt.close()
            for i in range(len(self.grid_types)):
                plt.loglog(self.nspts[i], self.times[i], label=self.grid_types[i])
            plt.xlabel("Number of samping points")
            plt.ylabel("Time (s)")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.show()
            plt.close()
    
    def plot_grid(self,i,j):
        """Plot one of the grids in the convergence plot.
        """
        grid_vecs = make_ptvecs(self.grid_types[i], self.grid_constants[j])
        grid_pts = make_grid(self.rcell_vectors, gr_vecs, self.offset)
        
        PlotMesh(grid_pts, self.rcell_vectors, self.offset)
