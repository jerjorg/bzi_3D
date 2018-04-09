# Revision History

## Revision 0.3.4
- Various fixes and additions all around.

## Revision 0.3.3
- Small changes to band plot functions and unit tests.

## Revision 0.3.2
- Added a new module which contains many of the same methods
  covered in the other modules but in 2D instead of 3D. This module,
  `all_2D.py` will be used to test adaptive refinement on 2D models
  that should be easier to visualize and program.
- Added a new notebook that looks at the 2D free electron model in
  detail.

## Revision 0.3.1
- Added `plot_all_bz` which can plot the Brillouin zone, the
  irreducible wedge of the Brillouin zone, and a mesh. The mesh can be
  any list of points, such as points along symmetry points, the
  irreducible k-points, etc.
- Added `plane3pts` to make_IBZ. It finds a plane given three points
  in the plane.
- Ran into trouble with variations on reciprocal lattice
  definitions. Added arguments to `Lattice`, `get_sympts`, and
  `get_sympaths` that allow freedom in selecting a convention.

## Revision 0.3.0
- Added new modules and notebooks:
  - `make_ibz.py` is a module that creates the first Brillouin zone
	and the irreducible wedge of the Brillouin zone. New method for
	plotting these are found in `plots.py` and a notebook that plots
	each of these is `Brillouin zone plots.ipynb`.
  - `quaternions.py` has methods for rotating and reflecting vectors
	using a quaternion derivation.
  - `utilities.py` has various usefull functions such determining if a
	vector is contained in a list of vectors.
  - `Blochl Plots.ipynb` reproduces the plots found in Blochl's paper.
  - `Free Electron Tetrahedron Convergence.ipynb` tests the
	convergence of Blochl's tetrahedron method with and without weight
	corrections to rectangular integration on a free electron model.
  - `indexing tetrahedron grid.ipynb` is a notebook that works out how
	the grid and tetrahedra are indexed in Blochl's method.

## Revision 0.2.15
- Small changes to setup and run of VASP and QE jobs. Now able to
  get input parameters and run convergence tests.

## Revision 0.2.14
- Added a function that will plot the data gathered while testing
  the energy cutoffs and number bands for QE runs.

## Revision 0.2.13
- Added functions that test the input parameters for Quantum 
  Espresso runs. 

## Revision 0.2.12
- Added a function that pickles Quantum Espresso convergence data.

## Revision 0.2.11
- Added functions that will plot the density of states, integrated
  density of states, and create convergence plots of the various
  energies agains density of states sampling.

## Revision 0.2.10
- Created a function that will plot the amount of wrap-around
  soft charge in three directions agains the energy cutoff of
  the augmentation charge.

## Revision 0.2.9
- Created a function that will plot the convergence of various
  values as a function of the energy cutoff. This should help
  determine the value of `ENCUT` in the VASP INCAR file.

## Revision 0.2.8
- Created functions that will test the `NBANDS` and `ENCUT` input
  parameters in the VASP INCAR.

## Revision 0.2.7
- Added `create_incar` and `read_potcar` to `read_and_write.py`.

## Revision 0.2.6
- Added `run_VASP` that will build a file structure for DFT
  calculations and then run them with VASP.
- Minor changes to .travis.yml

## Revision 0.2.5
- Added `read_VASP` that will read many relevant values from
  the VASP input and output files. It isn't comprehensive yet,
  and more could be added later. Wrote one unit test, could stand
  to have another.
- Fixed unit tests in test_tetrahedron.py and 
  test_read_and_write.py.

## Revision 0.2.4
- Added `remove_QE_save` that removes the save file from QE runs.

## Revision 0.2.3
- Updated setup.py version, install_requires and classifiers.
- Minor fixes to unit tests.
- Added `run_QE`, which will create file structure and submit
  Quantum Espresso jobs according to template input file

## Revision 0.2.2
- Created a module named `read_and_write.py`, which will be used
  to read data from DFT output files. It will also take data
  generated from empirical pseudopotential calculations and 
  write it to file.
- Added a function `plot_QE_data` that will create convergence 
  plots of the various energies from Quantum espresso data.

## Revision 0.2.1
- Fixed density of states and number of states weights by adding 
  multiplying their weights by two. I believe that Blochl didn't
  account for spin degeneracy in these weights.
- Fixed `integration_weights` by removing division by primitive
  cell volume.
- Added a `conftest.py` module that should help with filtering
  unit test runs.
- Added a new method to `FreeElectronModel` that changes the degree
  of the free electron dispersion relation, the exact Fermi level
  and band energy.
- Added a new pseudopotential class called
  `MultipleFreeElectronModel` that is the same as the free electron 
  model except it has more than one band.
- Added `free_dos` and `free_nos` to pseudopots. They calculate
  the exact density of states and number of states for the free
  electron model.
- Added `plot_states` to plots. This plots the density of states
  and number of states of the free electron pseudopotential. The
  hope is to extend it to others in the future.
- Created `generate_states_data` and `plot_states_data` which 
  generate and plot the density of states and number of states.

## Revision 0.2.0
- Fixed a bug in `calc_total_states` in tetrahedra.py. The volume
  of the tetrahedra was incorrectly calculated when the weights
  were anything but all ones.
- Added many more unit tests to `find_irreducible_tetrahedra`. It 
  is now working.

## Revision 0.1.13
- Fixed 'brillouin_zone_mapper' in symmetry.py. The reduced_grid
  contained points outside of the Minkowski unit cell, so, in cases
  where the original pt1 was the shortest point, the translationally
  equivalent point in reduced_grid was never replaced.

## Revision 0.1.12
- Fixed `brillouin_zone_mapper`. The k-points are symmetry reduced,
  mapped into the first unit cell in the Minkowski basis, then looks
  at the translationally equivalent k-points in the eight unit cells
  that have a vertex at the origin to see if any them lie closer to
  the origin.
- Added notebook for creating unit tests `test_symmetry.ipynb`.

## Revision 0.1.11
- Had a problem with the Fermi level of different grids varying by
  more than expected. The problem was stability of a ceiling call.
- Added optional arguments to plot_mesh so that now it can be saved
  to a file.
- Made the function `get_corrected_total_energy` from tetrahedron.py
  compatible with Python 2 by making the division of the number of
  valence electrons by two become float division.
- Made the function `rectangular_fermi_level` from integration.py
  compatible with the Jupyter notebook Tetrahedra Convergence by
  converting weights[i] from a numpy.float64 to an int.
- Temporarily modified the the method `eval` in `EmpiricalPP` to make
  comparisons between different basis sets.
- Changed `brillouin_zone_mapper` so that it no longer maps points
  into Minkowski space.

## Revision 0.1.10
- Adjusted enforced line breaks in HISTORY.md such that they occur
  after 70 characters.
- Replaced conversions to different coordinate systems that involved
  single line for loops with numpy matrix multiplication in module
  BZI/symmetry in `functions shells`, `find_orbitals`, and
  `find_full_orbitals`, 
- Removed `nfind_orbitals`, an experimental symmetry reduction routine
  that was supposed to replace `find_orbitals`. Will return to replace
  it if need be.
- Added optional argument to `find_full_orbitals` that allows the user
  to find the k-points in the orbital in the first unit cell or
  outside it. Still need to remove duplicate points in orbits.

## Revision 0.1.9
- Fixed a bug in BZI/symmetry.py in `sphere_pts` where it wouldn't be
  able to find the points within a sphere when the offset was
  large. At one point the offset needed to be subtracted from the
  reciprocal lattice vectors but it was missing.
- Added a new function BZI.symmetry called
  `brillouin_zone_mapper`. This function symmetry reduces a grid,
  moves the reduced points to the first unit cell, finds the Minkowski
  basis of the provided reciprocal lattice lattice vectors, moves the
  points into Minkowski space, and finally moves all the points into
  the first Brillouin zone of the Minkowski basis.
- Added a function `minkowski_reduce_basis` which is nothing more than
  the same function from phenum.vector_utils except the basis
  input/output vectors are columns instead of rows.

## Revision 0.1.8
- Added a function `plot_paths` in plots that plots the lines between
  symmetry points that are used in creating band structure plots.
- Made `CohenEmpiricalPP` hamiltonian and eval methods faster by
  replacing some Python for loops with numpy array arithmetic.
- Added an attribute to the empirical pseudopotentials whose purpose
  is to give a description of the pseudopotential.

## Revision 0.1.7
- The function find_orbitals wasn't returning the correct symmetry
  reduction when the grid was centered at the origin. Fixed this by
  adding
  ``` new_gps = [np.round(np.dot(pg, g), 15)%1, np.round(np.dot(pg,
      	         g), 15)]
  ```
  and removing both points if there were contained in the origin grid.
- Energies were being identified as above the Fermi level when they
  were at the same energy as the Fermi level. Added a small number to
  the Fermi level so that they would be included.
- Ran into issues where more energy states than filled states were
  contributing to the total energy. Fixed this by keeping track of the
  k-points that had energies near the Fermi level added their
  contribution last.
- Added analytic solutions to the free electron pseudopotential as
  attributes.

## Revision 0.1.6
- Changed .travis.yml python version from 3.5 to 3.6.
- Modified the EmpiricalPP class in pseudopots.py so that it would run
  much faster (50x!) by eliminating for loops.
- Slight change to calc_total_states when extra .flatten() was slowing
  this down.
- Modified make_large_grid so that it would provide a translationally
  unique grid within the first unit cell. Unit tests now pass.

## Revision 0.1.5
- Separate the unit tests in test_symmetry.py so that they don't time
  out when run on travis.ci servers.

## Revision 0.1.4
- Add a notebook that compares convergence of the improved tetrahedron
  method to the rectangular method.

## Revision 0.1.3
- Add routines to tetrahedron.py that implement the improved linear
  tetrahedron method with the corrected weights.
- Made all offsets added to k-points instead of being subtracted.

## Revision 0.1.2
- Remove print statements from integration.
- Add extra arguments `grid_vecs` and `offset` to
  `find_irreducible_tetrahedrahe` since they are required for the new
  symmetry reduction routine `reduce_kpoint_list`.
- Rename module improved_tetrahedron_method.py tetrahedron.py.
- Made all offsets so that they are added to the k-point instead of
  subtracted in sampling.py and plots.py

## Revision 0.1.1
- Fix integration unit tests. Since `make_cell_points` always moves
  the points back into the unit cell, it can't be used with the
  free-electron pseudopotential. Replaced `make_cell_points` with
  `make_grid` in unit tests.
- The free electron pseudopotential can't have more than one electron
  since having two would move the Fermi surface outside the unit
  cell. This is a problem because it isn't periodic like most other
  pseudopotentials.

## Revision 0.1.0
- Many thanks to Dr. Hess for allowing integration of his code with
  BZI.  Add the following modules to BZI that were written by
  [Dr. Hess](https://www.physics.byu.edu/department/directory/hess):
  makeIBZ.py, kmeshroutines.py, and IBZTest.py (found in tests).

## Revision 0.0.14
- Add unit tests that compare k-point symmetry reduction to that
  obtained in VASP for orthorhombic, monoclinic, and tetragonal
  crystal classes.

## Revision 0.0.13
- Change the line in `find_kpt_index`
```
np.round(np.dot(L, gpt), eps).astype(int)%D
# np.dot(L, np.round(gpt).astype(int))%D
```
- Add a check that skips k-points when they don't belong to the grid
```
if not np.allclose(np.dot(invK, rot_kpt-shift),
       np.round(np.dot(invK, rot_kpt-shift))):
    continue
```
in `reduce_kpoint_list`.
- Add unit tests for simple cubic lattices.

## Revision 0.0.12
- Fix `find_orbitals` so that it correctly reduces a k-point grid when
  k-point grid includes points translationally equivalent.
- Fix naming convention in unit tests for `make_lattice_vectors`.
- Fix finite precision error in `find_kpt_index`. Replace eps = 10
  with eps = 9.
  
## Revision 0.0.11
- Add new implementation of symmetry reducing k-point grids based on
  Gus's ideas found in msg-byu/kgridGen.
  
## Revision 0.0.10
- Begin implementation of new k-point symmetry reduction function.

## Revision 0.0.9
- Make a slight change to `make_cell_points` in sampling. The points
  now always reside in the first unit cell no matter the shift.

## Revision 0.0.8
- Reverse the order of the HISTORY.md file.
- Add band energy as an attribute of the pseudopotential classes.
- Make Minor changes to modules symmetry, sampling,
  improved_tetrahedron_method.

## Revision 0.0.7
- Add improved tetrahedron method module that includes methods for
  creating a grid, splitting the grid into tetrahedra, and calculating
  the Fermi level, and total energy.
- Make minor changes to symmetry.
- Create free electron model pseudopotential.
- Add unit tests for calculating the Fermi level and total energy.

## Revision 0.0.6
- Fix bugs in `find_orbitals` and `find_full_orbitals` that were
  caused by assuming the group operators returned by `point_group`
  were in lattice coordinates.

## Revision 0.0.5
- Add roughly 15 new pseudopotentials for simple, monatomic metals.
- Add unit tests that plot the band structures of these metals.
- Remove `rectangle_method` function from integration since it relied
  on an outdated sampling method.
- Add number of valence electrons as an attribute to the
  pseudopotentials.

## Revision 0.0.4
- Add a class for the empirical pseudopotentials.
- Add a simpler function for generating the points within a
  parallelepiped, `make_cell_points`.
- Rewrite pseudopotentials in compact form with far less repeated
  code.

## Revision 0.0.3
- Generalize `make_ptvecs` so that it can handle all 14 Bravais
  lattices.
- Create `make_lattice_vectors` that also generates the 14 Bravais
  lattices.
- Create dictionaries and functions that return the high symmetry
  points for all possible Brillouin zones.
- Create a function that returns the symmetry points of a lattice with
  the provided lattice parameters by referencing the symmetry point
  dictionaries.
- Make unit tests for the symmetry point function for all possible
  Brilloun zones.
- Make unit tests for the primitive translation vectors.
- Make unit tests for the symmetry paths functions.
- Create a function that identifies the lattice type.
- Create a lattice class that stores all variables related to the
  lattice.

## Revision 0.0.2
- Add functionality to `make_grid` so that it can create grids in
  lattice coordinates.
- Add additional lattices to `make_ptvecs`.

## Revision 0.0.1
- Add integration, plots, convergence, symmetry and sampling modules.
- Create unit tests for module functions.
- Implement sphinx documentation.
- Get Travis CI and Coveralls working.
