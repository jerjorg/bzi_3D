import pytest

@pytest.fixture(scope="session")
def run(tests):
    # Tetrahedron method tests
    if tests == "all tetrahedra":
        tests = ["test_number_of_states",
                 "test_density_of_states",
                 "test_integration_weights",
                 "test_find_tetrahedra",
                 "test_corrections",
                 "test_grid_and_tetrahedra",
                 "test_find_irreducible_tetrahedra",
                 "test_integrals",
                 "test_adjacent_tetrahedra",
                 "test_convert_tet_index",
                 "test_get_grid_tetrahedra",
                 "test_find_adjacent_tetrahedra"]
        
    elif tests == "improved tetrahedra":
        tests = ["test_corrections",
                 "test_adjacent_tetrahedra",
                 "test_convert_tet_index",
                 "test_get_grid_tetrahedra",
                 "test_find_adjacent_tetrahedra"]

    elif tests == "tetrahedra physics":
        tests = ["test_number_of_states",
                 "test_density_of_states",
                 "test_integration_weights",
                 "test_integrals"]

    elif tests == "make tetrahedra":
        tests = ["test_grid_and_tetrahedra",
                 "test_find_irreducible_tetrahedra"]

    # Rectangle method tests
    elif tests == "all rectangle":
        tests = ["test_rectangular", "test_rectangular_fermi_level"]

    # Pseudopotential tests
    elif tests == "all pseudopotential":
        tests = ["test_pseudopotentials"]

    # Sampling tests
    elif tests == "all sampling":
        tests = ["test_make_grid",
                 "test_make_cell_points",
                 "test_get_minmax_indices",
                 "test_swap_columns",
                 "test_swap_rows",
                 "test_HermiteNormalForm",
                 "test_UpperHermiteNormalForm",
                 "test_make_grid2",]

    # Symmetry tests
    elif tests == "all symmetry":
        tests = ["test_make_ptvecs",
                 "test_make_lattice_vectors",
                 "test_sympts_sympaths",
                 "test_find_orbitals",
                 "test_reduce_simple_cubic",
                 "test_reduce_body_centered_cubic",
                 "test_reduce_face_centered_cubic",
                 "test_reduce_orthorhombic",
                 "test_reduce_base_centered_orthorhombic",
                 "test_reduce_body_centered_orthorhombic",
                 "test_reduce_face_centered_orthorhombic",
                 "test_reduce_monoclinic",
                 "test_reduce_base_centered_monoclinic",
                 "test_reduce_tetragonal",
                 "test_reduce_body_centered_tetragonal"]

    # Read and write tests
    elif tests == "all read_and_write":
        tests = ["test_read_QE", "test_read_VASP"]
    else:
        tests = []
    
    return tests
