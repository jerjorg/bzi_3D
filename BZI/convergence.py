import numpy as np
import matplotlib.pyplot as plt
import time
from BZI.symmetry import make_ptvecs
from BZI.sampling import make_grid
from BZI.pseudopots import W1
from BZI.integration import rectangle_method, monte_carlo
from BZI.plots import PlotMesh
class Convergence(object):
    """ Compare integrations of pseudo-potentials by creating convergence plots.

    Args:
        pseudo_potential (function): a pseudo-potential function taken from 
            BZI.pseudopots
        cutoff (float): the energy cutoff of the pseudo-potential
        cell_type (str): the geometry of the integration cell
        cell_constant (float): the size of the integration cell
        offset (list): a vector that offsets the mesh from the origin and is 
            given in mesh coordinates.
        mesh_types (list): a list of mesh types
        mesh_constants (list): a list of mesh constants
        integration_methods (list): a list of integration methods
        scale (list): a list of integers for creating custom meshes

    Attributes:
        pseudo_potential (function): a pseudo-potential function taken from 
            BZI.pseudopots
        cell_type (str): the geometry of the integration cell.
        cell_constant (float): the size of the integration cell.
        cell_vectors (np.ndarray): an array vectors as columns of a 3x3 numpy
            array that is used to create the cell
        mesh_types (list): a list of mesh types
        mesh_constants (list): a list of mesh constants
        integration_methods (list): a list of integration methods
        answer (float): the expected result of integration
        errors (list): a list of errors for each mesh type
        nspts (list): a list of the number of sampling points for each mesh type
        integrals (list): a list of integral value for each mesh type and constant
        times (list): a list of the amount of time taken computing the mesh 
            generation and integration.
        scale (list): a list of integers for creating custom meshes
    """
    
    def __init__(self, pseudo_potential=None, cutoff=None, cell_type=None,
                 cell_constant=None, offset=None, mesh_types=None,
                 mesh_constants=None, scale=None, integration_methods=None,
                 origin=None, random = None):
        self.pseudo_potential = pseudo_potential or W1
        self.cutoff = cutoff or 4.
        self.cell_type = cell_type or "sc"
        self.cell_constant = cell_constant or 1.
        self.cell_vectors = make_ptvecs(self.cell_type, self.cell_constant)
        self.mesh_types = mesh_types or ["sc", "bcc", "fcc"]
        self.mesh_constants = mesh_constants or [1/n for n in range(2,11)]
        self.offset = offset or [0.,0.,0.]
        self.integration_methods = integration_methods or [rectangle_method]
        self.scale = scale or [1.,1.,1.]
        self.origin = origin or [0.,0.,1.]
        self.random = random or False

    def compare_meshes(self, answer, plot=False, save=False):
        self.answer = answer
        if self.random == True:
            nm = len(self.mesh_types)
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
            self.nspts = [[] for _ in range(len(self.mesh_types))]
            self.errors = [[] for _ in range(len(self.mesh_types))]
            self.integrals = [[] for _ in range(len(self.mesh_types))]
            self.times = [[] for _ in range(len(self.mesh_types))]
        integration_method = self.integration_methods[0]
        for (i,mesh_type) in enumerate(self.mesh_types):
            for mesh_const in self.mesh_constants:
                mesh_vecs = make_ptvecs(mesh_type, mesh_const, self.scale)
                time1 = time.time()
                npts, integral = integration_method(self.pseudo_potential,
                                                            self.cell_vectors,
                                                            mesh_vecs,
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
            if self.random == True:
                plt.loglog(self.nspts[nm], self.errors[nm], label="random", color="orange")
            for i in range(len(self.mesh_types)):
                plt.loglog(self.nspts[i], self.errors[i], label=self.mesh_types[i])
            plt.xlabel("Number of samping points")
            plt.ylabel("Error")
            test = [1./n**(2./3) for n in self.nspts[0]]
            plt.loglog(self.nspts[0], test, label="1/n**(2/3)")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.show()
            plt.close()
            for i in range(len(self.mesh_types)):
                plt.loglog(self.nspts[i], self.times[i], label=self.mesh_types[i])
            plt.xlabel("Number of samping points")
            plt.ylabel("Time (s)")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.show()
            plt.close()
    
    def plot_mesh(self,i,j):
        """Plot one of the meshes in the convergence plot.
        """
        mesh_vecs = make_ptvecs(self.mesh_types[i], self.mesh_constants[j])
        mesh_pts = make_grid(self.cell_vectors, mesh_vecs, self.offset)
        
        PlotMesh(mesh_pts, self.cell_vectors, self.offset)
