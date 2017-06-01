# Revision History

## Revision 0.0.1

- Add integration, plots, convergence, symmetry and sampling modules.
- Create unit tests for module functions.
- Implement sphinx documentation.
- Get Travis CI and Coveralls working.

## Revision 0.0.2
- Add functionality to make_grid so that it can create grids in lattice
  coordinates.
- Add additional lattices to make_ptvecs.

Revision 0.0.3
- Generalize make_ptvecs so that it can handle all 14 Bravais lattices.
- Create make_lattice_vectors that also generates the 14 Bravais lattices.
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

Revision 0.0.4
- Add a class for the empirical pseudopotentials.
- Add a simpler function for generating the points within a parallelepiped,
  make_cell_points.
- Rewrite pseudopotentials in compact form with far less repeated code.

Revision 0.0.5
- Add rougly 15 new pseudopotentials for simple, monatomic metals.
- Add unit tests that plot the band structures of these metals.
- Remove rectangle_method function from integration since it relied on an
  outdated sampling method.
- Add number of valence electrons as an attribute to the pseudopotentials.
