import pytest

#@pytest.fixture(scope="session")
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

    # pseudopotential tests
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
                 "test_make_grid2"]

    # Symmetry tests
    elif tests == "all symmetry":
        tests = ["test_make_ptvecs",
                 "test_make_lattice_vectors",
                 "test_sympts_sympaths",
                 "test_get_orbits",
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
                 "test_reduce_body_centered_tetragonal",
                 "test_get_point_group",
                 "test_gaussian_reduction",
                 "test_reduce_lattice_vector",
                 "test_minkowski_reduce_basis",
                 "test_get_space_group",
                 "test_bring_into_cell",
                 "test_check_commensurate"]

    # Read and write tests
    elif tests == "all make_IBZ":
        tests = ["test_get_bragg_planes",
                 "test_trim_small",
                 "test_three_planes_intersect",
                 "test_find_bragg_shells",
                 "test_find_bz"]
    elif tests == "all utilities":
        tests = ["test_remove_points",
                 "test_find_point_index",
                 "test_find_point_indices",
                 "test_trim_small",
                 "test_check_contained",
                 "test_swap_rows_columns",
                 "test_inside"]
    elif tests == "all 2D":
        tests = ["test_make2D_lattice_basis",
                 "test_get_2Dlattice_type",
                 "test_HermiteNormalForm2D",
                 "test_make_cell_points2D",
                 "test_plot_mesh2D",
                 "test_get_circle_pts",
                 "test_get_perpendicular_vector2D",
                 "test_get_line_equation2D",
                 "test_square_tesselation",
                 "test_refine_square",
                 "test_get_bilin_coeffs",
                 "test_eval_bilin",
                 "test_integrate_bilinear",
                 "test_integrate_bilinear",
                 "test_find_param_intersect",
                 "test_eval_bilin",
                 "test_group_bilinear_intersections",
                 "test_bilin_density_of_states",
                 "test_get_integration_case",
                 "test_get_integration_cases"]
    else:
        tests = []

    return tests
