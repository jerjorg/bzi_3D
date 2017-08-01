# Revision History

## Revision 0.1.7
- The function find_orbitals wasn't returning the correct symmetry reduction
  when the grid was centered at the origin. Fixed this by adding
  ```
  new_gps = [np.round(np.dot(pg, g), 15)%1, np.round(np.dot(pg, g), 15)]

  ```
  and removing both points if there were contained in the origin grid.
- Energies were being identified as above the Fermi level when they were
  at the same energy as the Fermi level. Added a small number to the Fermi
  level so that they would be included.
- Ran into issues where more energy states than filled states were contributing
  to the total energy. Fixed this by keeping track of the k-points that had
  energies near the Fermi level added their contribution last.
- Added analytic solutions to the free electron pseudopotential as attributes.

## Revision 0.1.6
- Changed .travis.yml python version from 3.5 to 3.6.
- Modified the EmpiricalPP class in pseudopots.py so that it would run much
  faster (50x!) by eliminating for loops.
- Slight change to calc_total_states when extra .flatten() was slowing this
  down.
- Modified make_large_grid so that it would provide a translationally unique
  grid within the first unit cell. Unit tests now pass.

## Revision 0.1.5
- Separate the unit tests in test_symmetry.py so that they don't time out when
  run on travis.ci servers.

## Revision 0.1.4
- Add a notebook that compares convergence of the improved tetrahedron method
  to the rectangular method.

## Revision 0.1.3
- Add routines to tetrahedron.py that implement the improved linear
  tetrahedron method with the corrected weights.
- Made all offsets added to k-points instead of being subtracted.

## Revision 0.1.2
- Remove print statements from integration.
- Add extra arguments `grid_vecs` and `offset` to `find_irreducible_tetrahedrahe`
  since they are required for the new symmetry reduction routine
  `reduce_kpoint_list`.
- Rename module improved_tetrahedron_method.py tetrahedron.py.
- Made all offsets so that they are added to the k-point instead of subtracted
  in sampling.py and plots.py

## Revision 0.1.1
- Fix integration unit tests. Since `make_cell_points` always moves the points
  back into the unit cell, it can't be used with the free-electron
  pseudopotential. Replaced `make_cell_points` with `make_grid` in unit tests.
- The free electron pseudopotential can't have more than one electron since
  having two would move the Fermi surface outside the unit cell. This is a
  problem because it isn't periodic like most other pseudopotentials.

## Revision 0.1.0
- Many thanks to Dr. Hess for allowing integration of his code with BZI.
  Add the following modules to BZI that were written by
  [Dr. Hess](https://www.physics.byu.edu/department/directory/hess):
  makeIBZ.py, kmeshroutines.py, and IBZTest.py (found in tests).

## Revision 0.0.14
- Add unit tests that compare k-point symmetry reduction to that obtained in
  VASP for orthorhombic, monoclinic, and tetragonal crystal classes.

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
- Fix `find_orbitals` so that it correctly reduces a k-point grid when k-point
  grid includes points translationally equivalent.
- Fix naming convention in unit tests for `make_lattice_vectors`.
- Fix finite precision error in `find_kpt_index`. Replace eps = 10 with eps = 9.
  
## Revision 0.0.11
- Add new implementation of symmetry reducing k-point grids based on Gus's
  ideas found in msg-byu/kgridGen.
  
## Revision 0.0.10
- Begin implementation of new k-point symmetry reduction function. 

## Revision 0.0.9
- Make a slight change to `make_cell_points` in sampling. The points now always
  reside in the first unit cell no matter the shift.

## Revision 0.0.8
- Reverse the order of the HISTORY.md file.
- Add band energy as an attribute of the pseudopotential classes.
- Make Minor changes to modules symmetry, sampling, improved_tetrahedron_method.

## Revision 0.0.7
- Add improved tetrahedron method module that includes methods for creating
  a grid, splitting the grid into tetrahedra, and calculating the Fermi level,
  and total energy.
- Make minor changes to symmetry.
- Create free electron model pseudopotential.
- Add unit tests for calculating the Fermi level and total energy.

## Revision 0.0.6
- Fix bugs in `find_orbitals` and `find_full_orbitals` that were caused by
  assuming the group operators returned by `point_group` were in lattice
  coordinates.

## Revision 0.0.5
- Add roughly 15 new pseudopotentials for simple, monatomic metals.
- Add unit tests that plot the band structures of these metals.
- Remove `rectangle_method` function from integration since it relied on an
  outdated sampling method.
- Add number of valence electrons as an attribute to the pseudopotentials.

## Revision 0.0.4
- Add a class for the empirical pseudopotentials.
- Add a simpler function for generating the points within a parallelepiped,
  `make_cell_points`.
- Rewrite pseudopotentials in compact form with far less repeated code.

## Revision 0.0.3
- Generalize `make_ptvecs` so that it can handle all 14 Bravais lattices.
- Create `make_lattice_vectors` that also generates the 14 Bravais lattices.
- Create dictionaries and functions that return the high symmetry points for
  all possible Brillouin zones.
- Create a function that returns the symmetry points of a lattice with the
  provided lattice parameters by referencing the symmetry point dictionaries.
- Make unit tests for the symmetry point function for all possible Brilloun
  zones.
- Make unit tests for the primitive translation vectors.
- Make unit tests for the symmetry paths functions.
- Create a function that identifies the lattice type.
- Create a lattice class that stores all variables related to the lattice.

## Revision 0.0.2
- Add functionality to `make_grid` so that it can create grids in lattice
  coordinates.
- Add additional lattices to `make_ptvecs`.

## Revision 0.0.1
- Add integration, plots, convergence, symmetry and sampling modules.
- Create unit tests for module functions.
- Implement sphinx documentation.
- Get Travis CI and Coveralls working.