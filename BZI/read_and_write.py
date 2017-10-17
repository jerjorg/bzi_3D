import os
import subprocess
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


#######
####### =========================================================================
####### You must already have the element directories with populated initial_data
####### =========================================================================
#######

def run_QE(element, parameters):
    """A function that will take the values contained in parameters, create the
    file structure, and submit a series of Quantum Espresso jobs. The directory
    from which this command is run must contain run.py and run.sh files. It
    must contain a directory named 'element', and this directory must contain
    a directory named 'initial data' in which is located the template Quantum
    Espresso input file. The run.sh file runs the run.py script.
    
    Args:
        parameters (dict): a dictionary of adjustable parameters. The following
            key strings must be present: 'grid type list', 'offset list',
            'occupation list', 'smearing list', 'smearing value list',
            and 'k-points list'. Their corresponding values must be lists.
    """
    
    home_dir = os.getcwd()
    
    # Move into the element directory.
    element_dir = home_dir + "/" + element
    os.chdir(element_dir)
    
    # Locate the inital data.
    idata_loc = element_dir + "/" + "initial_data"
    
    # Make and move into the grid type directory.
    for grid_type in parameters["grid type list"]:
        grid_dir = element_dir + "/" + grid_type
        if not os.path.isdir(grid_dir):
            os.mkdir(grid_dir)
        os.chdir(grid_dir)
        
        # Make and move into the offset directory
        for offset in parameters["offset list"]:
            offset_name = str(offset).strip("[").strip("]")
            offset_name = offset_name.replace(",", "")
            
            offset_dir = grid_dir + "/" + offset_name
            if not os.path.isdir(offset_dir):
                os.mkdir(offset_dir)
            os.chdir(offset_dir)
            
            # Make and move into the occupation directory.
            for occupation in parameters["occupation list"]:
                occupation_dir = offset_dir + "/" + occupation
                if not os.path.isdir(occupation_dir):
                    os.mkdir(occupation_dir)
                os.chdir(occupation_dir)
                
                # Make and move into the smearing type directory.
                for smearing in parameters["smearing list"]:
                    smearing_dir = occupation_dir + "/" + smearing
                    if not os.path.isdir(smearing_dir):
                        os.mkdir(smearing_dir)
                    os.chdir(smearing_dir)
                    
                    # Make and move into the smearing value directory.
                    for smearing_value in parameters["smearing value list"]:
                        smearing_value_dir = smearing_dir + "/" + str(smearing_value)
                        if not os.path.isdir(smearing_value_dir):
                            os.mkdir(smearing_value_dir)
                        os.chdir(smearing_value_dir)
                            
                        # Make and move into the number of k-points directory.
                        for kpoints in parameters["k-point list"]:
                            nkpoints = np.prod(kpoints)
                            kpoints_dir = smearing_value_dir + "/" + str(nkpoints)
                            if not os.path.isdir(kpoints_dir):
                                os.mkdir(kpoints_dir)
                            os.chdir(kpoints_dir)

                            # Copy initial_data, run.sh and run.py to current directory.
                            subprocess.call("cp " + idata_loc + "/* ./", shell=True)
                            subprocess.call("cp " + home_dir + "/run.sh ./", shell=True)
                            subprocess.call("cp " + home_dir + "/run.py ./", shell=True)
                            subprocess.call('chmod +x run.py', shell=True)                            
                                                        
                            # Correctly label this job.
                            # Read in the file.
                            with open('run.sh', 'r') as file :
                                filedata = file.read()

                            # Replace the target string.
                            filedata = filedata.replace('ELEMENT', str(element))
                            filedata = filedata.replace('GRIDTYPE', grid_type)
                            filedata = filedata.replace('OFFSET', str(offset))
                            filedata = filedata.replace('OCCUPATION', occupation)
                            filedata = filedata.replace('SMEAR', smearing)
                            filedata = filedata.replace('SMRVALUE', str(smearing_value))
                            filedata = filedata.replace('KPOINT', str(nkpoints))
                            
                            # Write the file out again.
                            with open('run.sh', 'w') as file:
                                file.write(filedata)
                            
                            # Replace values in the input file.
                            # Read in the file.
                            with open(element + '.in', 'r') as file:
                                filedata = file.read()

                            # Replace the target string.
                            filedata = filedata.replace("occupation type", occupation)
                            filedata = filedata.replace("smearing method", smearing)
                            filedata = filedata.replace("smearing value", str(smearing_value))
                            
                            for i,kp in enumerate(kpoints):
                                kp_name = "kpoint" + str(i + 1)
                                filedata = filedata.replace(kp_name, str(kp))
                                
                            for j,off in enumerate(offset):
                                off_name = "offset" + str(j + 1)
                                filedata = filedata.replace(off_name, str(off))
                            
                            # Write the file out again.
                            with open(element + '.in', 'w') as file:
                                file.write(filedata)
                                
                            # Adjust time to run and memory
                            with open("run.sh", "r") as file:
                                filedata = file.read()
                            
                            if nkpoints <= 8000:    
                                filedata = filedata.replace("12:00:00", "4:00:00")
                                filedata = filedata.replace("4096", "8192")
                                
                            elif nkpoints > 8000 and nkpoints < 27000:                                    
                                filedata = filedata.replace("12:00:00", "6:00:00")
                                filedata = filedata.replace("4096", "16384")                                
                                
                            elif nkpoints >= 27000 and nkpoints < 64000:
                                filedata = filedata.replace("12:00:00", "12:00:00")
                                filedata = filedata.replace("4096", "32768")
                                
                            elif nkpoints >= 64000:
                                filedata = filedata.replace("12:00:00", "24:00:00")
                                filedata = filedata.replace("4096", "65536")
                                
                            subprocess.call('sbatch run.sh', shell=True)
                            
                            os.chdir(smearing_value_dir)
                        os.chdir(smearing_dir)
                    os.chdir(occupation_dir)
                os.chdir(offset_dir)
            os.chdir(grid_dir)
        os.chdir(element_dir)
    os.chdir(home_dir)


def read_QE(location, file_prefix):
    """Create a dictionary of most of the data created during a
    single Quantum Espresso calculation.

    Args:
        location (str): the location of the Quantum Espresso calculation.
        file_prefix (str): the prefix of the input and output files.
    """
    
    QE_data = {}
    QE_data["self-consistent calculation time"] = []
    output_file = location + "/" + file_prefix + ".out"
    input_file = location + "/" + file_prefix + ".in"

    with open(input_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if "K_POINTS" in line:
                kpt_index = i + 1
                kpt_line = f[kpt_index].split()
                total_kpoints = (float(kpt_line[0])*float(kpt_line[1])*
                                 float(kpt_line[2]))
                QE_data["number of unreduced k-points"] = total_kpoints
    
    with open(output_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            line = line.strip()

            if "bravais-lattice index" in line:
                QE_data["bravais-lattice index"] = float(line.split()[-1])
                
            if "lattice parameter" in line:
                QE_data["lattice parameter"] = line.split()[-2] + " " + line.split()[-1]
            
            if "unit-cell volume" in line:
                QE_data["unit-cell volume"] = line.split()[-2] + " " + line.split()[-1]
                
            if "number of atoms/cell" in line:
                QE_data["number of atoms/cell"] = float(line.split()[-1])
            
            if "number of atomic types" in line:
                QE_data["number of atomic types"] = float(line.split()[-1])
                
            if "number of electrons" in line:
                QE_data["number of electrons"] = float(line.split()[-1])
                
            if "number of Kohn-Sham states" in line:
                QE_data["number of Kohn-Sham states"] = float(line.split()[-1])
            
            if "kinetic-energy cutoff" in line:
                QE_data["kinetic-energy cutoff"] = line.split()[-2] + " " + line.split()[-1]
                
            if "charge density cutoff" in line:
                QE_data["charge density cutoff"] = line.split()[-2] + " " + line.split()[-1]
            
            if "convergence threshold" in line:
                QE_data["convergence threshold"] = float(line.split()[-1])
                
            if "mixing beta" in line:
                QE_data["mixing beta"] = float(line.split()[-1])
                
            if "number of iterations used" in line:
                if line.split()[-4] == "=":
                    try:
                        test = float(line.split()[-3])
                        QE_data["number of iterations used"] = [line.split()[-3] + " " + line.split()[-2],
                                                                "0 " + line.split()[-1]]
                    except:
                        pass
                    
                    try:
                        test = float(line.split()[-2])
                        QE_data["number of iterations used"] = ["0 " + line.split()[-3],
                                                                line.split()[-2] + " " + line.split()[-1]]
                    except:
                        pass
                else:
                    QE_data["number of iterations used"] = [line.split()[-4] + " " + line.split()[-3],
                                                            line.split()[-2] + " " + line.split()[-1]]
            if "Exchange-correlation" in line:
                index = line.split().index("=")
                QE_data["Exchange-correlation"] = line[28:]
                
            if "celldm(1)" in line:
                next_line = f[i + 1]
                celldm1 = float(line.split()[1])
                celldm2 = float(line.split()[3])
                celldm3 = float(line.split()[5])
                celldm4 = float(next_line.split()[1])
                celldm5 = float(next_line.split()[3])
                celldm6 = float(next_line.split()[5])
                
                QE_data["crystallographic constants"] = [celldm1, celldm2, celldm3,
                                                         celldm4, celldm5, celldm6]
            # The crystal axes are set as the columns of a numpy array.
            if "crystal axes" in line:
                line1 = f[i + 1].split()
                line2 = f[i + 2].split()
                line3 = f[i + 3].split()
                
                a1 = [float(line1[3]), float(line1[4]), float(line1[5])]
                a2 = [float(line2[3]), float(line2[4]), float(line2[5])]            
                a3 = [float(line3[3]), float(line3[4]), float(line3[5])]
                
                QE_data["crystal axes"] = np.transpose([a1, a2, a3])
                
            # The reciprocal axes are set as the columns of a numpy array.
            if "reciprocal axes" in line:
                line1 = f[i + 1].split()
                line2 = f[i + 2].split()
                line3 = f[i + 3].split()
                
                a1 = [float(line1[3]), float(line1[4]), float(line1[5])]
                a2 = [float(line2[3]), float(line2[4]), float(line2[5])]            
                a3 = [float(line3[3]), float(line3[4]), float(line3[5])]
                
                QE_data["reciprocal axes"] = np.transpose([a1, a2, a3])
                
            if "Sym. Ops." in line:
                QE_data["Sym. Ops."] = float(line.split()[0])
                
            if "number of k points" in line:
                QE_data["number of reduced k-points"] = float(line.split()[4])
                                
                index = i + 2
                kpt_list = []
                kpt_weights = []
                kpt = True
                k_line = f[index]
                if "Number of k-points >=" in k_line:
                    kpt = False
                k_line = k_line.split()
                while kpt:
                    kpt_list.append([float(k_line[4].strip(":")),
                                     float(k_line[5]),
                                     float(k_line[6].strip("),"))])
                    kpt_weights.append(float(k_line[9]))                                    
                    index += 1
                    k_line = f[index].split()
                    if k_line == []:
                        kpt = False
                QE_data["k-points"] = kpt_list
                QE_data["k-point weights"] = kpt_weights
            
            if "Dense  grid:" in line:
                QE_data["Dense grid"] = line.split()[2] + " " + line.split()[3]
                QE_data["FFT dimensions"] = (float(line.split()[7].strip(",")),
                                             float(line.split()[8].strip(",")),
                                             float(line.split()[9].strip(",)")))
                
            if "total cpu time spent up to now is" in line:
                QE_data["self-consistent calculation time"].append(line.split()[-2] + " " + line.split()[-1])
                
                
            if "End of self-consistent calculation" in line:                
                index = i + 2
                k_line = f[index]
                kpt = True
                kpt_energies = []
                kpt_plane_waves = []
                
                if "convergence NOT achieved" in k_line:
                    QE_data["k-point plane waves"] = []
                    QE_data["k-point energies"] = []
                else:
                    k_line = k_line.split()
                    while kpt:
                        if k_line == []:
                            pass
                        else:
                            k_line = k_line
                            if k_line[0] == "k":
                                pw_index = k_line.index("PWs)")
                                kpt_plane_waves.append(float(k_line[pw_index - 1]))
                                e_index = index + 2
                                e_line = f[e_index].split()
                                energies = [float(j) for j in e_line]
                                kpt_energies.append(energies)
                                en = True
                                
                                while en:
                                    try:                    
                                        e_index += 1
                                        e_line = f[e_index].split()
                                        energies = [float(j) for j in e_line]
                                        kpt_energies[-1] += energies
                                    except:
                                        en = False
                    
                        index += 1
                        k_line = f[index].split()
                        if k_line != []:
                            if k_line[0] == "the":
                                kpt = False
                    QE_data["k-point plane waves"] = kpt_plane_waves
                    QE_data["k-point energies"] = kpt_energies
            
            if "the Fermi energy is" in line:
                QE_data["Fermi energy"] = line.split()[4] + " " + line.split()[5]
                
            if "!    total energy" in line:
                QE_data["total energy"] = line.split()[4] + " " + line.split()[5]
                
            if "one-electron contribution" in line:
                QE_data["one-electron contribution"] = line.split()[3] + " " + line.split()[4]
                
            if "hartree contribution" in line:
                QE_data["hartree contribution"] = line.split()[3] + " " + line.split()[4]
                
            if "xc contribution" in line:
                QE_data["xc contribution"] = line.split()[3] + " " + line.split()[4]
                
            if "ewald contribution" in line:
                QE_data["ewald contribution"] = line.split()[3] + " " + line.split()[4]
                
            if "convergence has been achieved in" in line:
                QE_data["number of self-consistent iterations"] = float(line.split()[5])
                
    return QE_data


def remove_QE_save(element, parameters):
    """A function that will remove the save folder created during a Quantum Espresso
    run.
    
    Args:
        parameters (dict): a dictionary of adjustable parameters. The following
            key strings must be present: 'grid type list', 'offset list',
            'occupation list', 'smearing list', 'smearing value list',
            and 'k-points list'. Their corresponding values must be lists.
    """

    save_file_name = element + ".save"
    
    home_dir = os.getcwd()
    
    # Make and move into the element directory.
    element_dir = home_dir + "/" + element
    os.chdir(element_dir)
        
    # Move into the grid type directory.
    for grid_type in parameters["grid type list"]:
        grid_dir = element_dir + "/" + grid_type
        os.chdir(grid_dir)
        
        # Make and move into the offset directory
        for offset in parameters["offset list"]:
            offset_name = str(offset).strip("[").strip("]")
            offset_name = offset_name.replace(",", "")
            offset_dir = grid_dir + "/" + offset_name
            os.chdir(offset_dir)
            
            # Make and move into the occupation directory.
            for occupation in parameters["occupation list"]:
                occupation_dir = offset_dir + "/" + occupation
                os.chdir(occupation_dir)
                
                # Make and move into the smearing type directory.
                for smearing in parameters["smearing list"]:
                    smearing_dir = occupation_dir + "/" + smearing
                    os.chdir(smearing_dir)
                    
                    # Make and move into the smearing value directory.
                    for smearing_value in parameters["smearing value list"]:
                        smearing_value_dir = smearing_dir + "/" + str(smearing_value)
                        os.chdir(smearing_value_dir)
                            
                        # Make and move into the number of k-points directory.
                        for kpoints in parameters["k-point list"]:
                            nkpoints = np.prod(kpoints)
                            kpoints_dir = smearing_value_dir + "/" + str(nkpoints)
                            os.chdir(kpoints_dir)
                            
                            subprocess.call('rm -r ' + save_file_name, shell=True)
                            os.chdir(smearing_value_dir)
                        os.chdir(smearing_dir)
                    os.chdir(occupation_dir)
                os.chdir(offset_dir)
            os.chdir(grid_dir)
        os.chdir(element_dir)
    os.chdir(home_dir)
    

def plot_QE_data(home, grid_type_list, occupation_list, energy_list):
    """Create convergence plots of Quantum Espresso data. The file 
    structure must be grid_types/occupation_types/kpoint_runs/.

    Args:
        home (str): the location of the Quantum Espresso grid type
            directories.
        grid_type_list (list): a list of grid types, same as grid type
            directories.
        occupation_list (list): a list of occupation types, same as the
            occupation directory names.
        energy_list (list): a list of energies to be plotted, corresponds
            to the keys in the dictionary returned by `read_QE`.
    """

    # A dictionary of the different energy for each energy in energy_list.
    energy_units_dict = {}

    os.chdir(home)
    grid_integral_dict = {}
    
    for grid_type in grid_type_list:
        grid_dir = home + "/" + grid_type
        for occupation in occupation_list:
            occ_dir = grid_dir + "/" + occupation
            # Find all the k-point directories.
            kpoint_list = os.listdir(occ_dir)
            
            # Find the k-point run that contains the converged values.
            max_kpts = max([float(k.strip("k")) for k in kpoint_list])**3
            
            # Initialize a dictionary of series.
            df_dict = {}
            
            for energy in energy_list:
                df_dict[energy] = pd.Series()            
            total_kpoint_series = pd.Series()
            reduced_kpoint_series = pd.Series()
            
            for i, kpoint in enumerate(kpoint_list):
                kpt_dir = occ_dir + "/" + kpoint
                
                # Extract data from Quantum Espresso output file.
                qe_data = read_QE(kpt_dir, "Al")
                total_kpoints = qe_data["number of unreduced k-points"]
                total_kpoint_series.set_value(total_kpoints, qe_data["number of unreduced k-points"])
                reduced_kpoint_series.set_value(total_kpoints, qe_data["number of reduced k-points"])
                
                try:
                    for energy in energy_list:
                        energy_entry = float(qe_data[energy].split()[0])
                        energy_units_dict[energy] = qe_data[energy].split()[1]
                        df_dict[energy].set_value(total_kpoints, energy_entry)                    
                except:
                    for energy in energy_list:
                        df_dict[energy].set_value(total_kpoints, np.nan)

            df = pd.DataFrame(df_dict)        
            for energy in energy_list:
                df[energy + " error"] = abs(df[energy] - df[energy][max_kpts])                
                df[energy + " error"] = df[energy + " error"].drop(max_kpts)
            df["number of unreduced k-points"] = total_kpoint_series
            df["number of reduced k-points"] = reduced_kpoint_series
            df = df.sort_values(by=["number of unreduced k-points"], ascending=True)
            
            # The name of this database is determined by the type of grid and integration method
            df_name = grid_type + " " + occupation
            grid_integral_dict[df_name] = df
            
    grid_integral_panel = pd.Panel(grid_integral_dict)

    # Save the pandas panel.
    panel_file = open(home + "/panel.p", "wb")
    pickle.dump(grid_integral_panel, panel_file)
    panel_file.close()

    plots_dir = home + "/plots"
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
        
    xaxis_list = ["number of unreduced k-points", "number of reduced k-points"]

    # First plot the errors.
    for grid in grid_type_list:
        grid_dir = plots_dir + "/" + grid_type
        if not os.path.isdir(grid_dir):
            os.mkdir(grid_dir)
        for occupation in occupation_list:
            occ_dir = grid_dir + "/" + occupation
            if not os.path.isdir(occ_dir):
                os.mkdir(occ_dir)
            entry = grid + " " + occupation
            for energy in energy_list:

                # Plot the convergence of the error and the energy
                for err_or_not in ["error", ""]:
                    if err_or_not == "":
                        err_name = energy
                    else:
                        err_name = energy + " " + err_or_not
                        
                    for xaxis in xaxis_list:
                        if err_or_not == "":
                            ax = grid_integral_panel[entry].plot(x=xaxis, y=err_name, 
                                                                 kind = "scatter")
                        else:
                            ax = grid_integral_panel[entry].plot(x=xaxis, y=err_name,
                                                                 loglog=True, kind = "scatter")
                        ax.set_ylabel(err_name)
                        lgd = ax.legend(entry, loc='center left', bbox_to_anchor=(1, 0.5))

                        file_name = occ_dir + "/" + xaxis + " " + err_name + ".pdf"
                        fig = ax.get_figure()
                        fig.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")
    
    # Plot the energy convergence of the three different methods, both the energies
    # themselves and their errors.
    colors = ["black", "blue", "red", "purple", "orange", "violet"]
        
    compare_dir = plots_dir + "/compare_grid_integral"
    if not os.path.isdir(compare_dir):
        os.mkdir(compare_dir)
        
    for energy in energy_list:
        for err_or_not in ["error", ""]:    
            if err_or_not == "":
                err_name = energy            
            else:
                err_name = energy + " " + err_or_not
            for xaxis in xaxis_list:
                for grid_type in grid_type_list:
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    legend_names = []
                    for j, occupation in enumerate(occupation_list):
                        entry = grid_type + " " + occupation
                        legend_names.append(entry)
                        if err_or_not == "":
                            grid_integral_panel[entry].plot(x=xaxis, y=err_name, 
                                                            kind = "scatter", ax=ax,
                                                            color=colors[j])
                        else:
                            grid_integral_panel[entry].plot(x=xaxis, y=err_name,
                                                            loglog=True, kind = "scatter",
                                                            ax=ax, color=colors[j])
                ax.set_ylabel(err_name)
                lgd = ax.legend(legend_names, loc='center left', bbox_to_anchor=(1, 0.5))

                file_name = compare_dir + "/" + xaxis + " " + err_name + ".pdf"
                fig = ax.get_figure()
                fig.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches="tight")


def read_VASP(location):
    """Create a dictionary of most of the data created during a
    single VASP calculation.

    Args:
        location (str): the location of the VASP calculation.
    """

    VASP_data = {}
    incar_file = location + "/INCAR"
    kpoints_file = location + "/KPOINTS"
    poscar_file = location + "/POSCAR"
    ibzkpt_file = location + "/IBZKPT"
    eigenval_file = location + "/EIGENVAL"
    outcar_file = location + "/OUTCAR"
    potcar_file = location + "/POTCAR"
    
    # Get the number of unreduced k-points. This only works for one of 
    # the automatic k-mesh generation methods, where the number of k-points
    # and offset of the k-mesh are provided.
    with open(kpoints_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if "Gamma" in line:
                kpt_index = i + 1
                kpt_line = f[kpt_index].split()
                total_kpoints = np.prod([int(k) for k in kpt_line])
                
                VASP_data["number of unreduced k-points"] = total_kpoints
                
                offset_index = i + 2
                offset_line = f[offset_index].split()

                VASP_data["offset"] = [float(off) for off in offset_line]

    """
    The POSCAR should have the following format:

    Al_FCC # comment line
    1 # universal scaling factor
    4.00 0.00 0.00 # first Bravais lattice vector
    0.00 4.00 0.00 # second Bravais lattice vector
    0.00 0.00 4.00 # third Bravais lattice vector
    Al # 
    4 # number of atoms per species, be consistent:
    direct or cartesian, only first letter is significant
    0.00000000 0.00000000 0.0000000 positions
    0.50000000 0.00000000 0.5000000
    0.50000000 0.50000000 0.0000000
    0.00000000 0.50000000 0.5000000            
    """

    # This section creates a list of atomic basis dictionaries. These dictionaries
    # contain the atomic species, number of atoms, coordinates, and positions.
    with open(poscar_file, "r") as file:
        f = file.readlines()
        VASP_data["name of system"] = f[0].strip()

        # If negative, the scaling factor should be interpreted as the total volume
        # of the cell.
        VASP_data["scaling factor"] = float(f[1])

        a1 = [float(v) for v in f[2].strip().split()]
        a2 = [float(v) for v in f[3].strip().split()]            
        a3 = [float(v) for v in f[4].strip().split()]
        VASP_data["lattice vectors"] = np.transpose([a1,a2,a3])

        j = 5
        atomic_bases = []
        more = True
        while more:
            try:
                atomic_basis = {}
                species = f[j].strip()
                atomic_basis["atomic species"] = species
                natoms = int(f[j+1].strip())
                more = False
                atomic_basis["number of atoms"] = natoms
                atomic_basis["coordinates"] = f[j+2].strip()
                pos = []
                for n in range(natoms):
                    pos.append([float(p) for p in f[j + 3 + n].strip().split()[:3]])
                atomic_basis["positions"] = pos
                atomic_bases.append(atomic_basis)
            except:
                more = False
        VASP_data["atomic bases"] = atomic_bases


    # If there is any special formating, such as LREALS=.FALSE.,
    # then reading the INCAR won't work.
    with open(incar_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            vals = line.strip("").split()
            VASP_data[vals[0]] = vals[-1]

    # Read the eigenvalues, number of reduced k-points, and the
    # k-points weights from the EIGENVAL file.
    with open(eigenval_file, "r") as file:
        f = file.readlines()
        kpoints = []
        weights = []
        nkpts = 0
        degeneracy = []
        for i,line in enumerate(f):
            if line.strip() == "":
                nkpts += 1
                vals = [float(k) for k in f[i+1].strip().split()]
                kpoints.append(vals[:3])
                weights.append(vals[-1])
                degeneracy.append(vals[-1]*VASP_data["number of unreduced k-points"])
        VASP_data["number of reduced k-points"] = nkpts
        VASP_data["k-point weights"] = weights
        VASP_data["reduced k-points"] = kpoints
        VASP_data["k-point degeneracy"] = degeneracy
        
    with open(outcar_file, "r") as file:
        f = file.readlines()
        sym_group = []
        nplane_waves = []
        for i, line in enumerate(f):

            if 'isymop' in line:
                r1 = [float(q) for q in line.split()[1:]]
                r2 = [float(q) for q in f[i+1].split()]            
                r3 = [float(q) for q in f[i+2].split()]            
                op = np.array([r1, r2, r3])
                sym_group.append(op)

            if "NBANDS" in line:
                VASP_data["NBANDS"] = int(line.split()[-1])
                
                
            if "plane waves" in line:
                nplane_waves.append(int(line.split()[-1]))
                
            if " Free energy of the ion-electron system (eV)" in line:
                alpha_line = f[i+2].split()
                alpha_name = alpha_line[0] + " " + alpha_line[1]
                VASP_data[alpha_name] = float(alpha_line[-1])
                
                ewald_line = f[i+3].split()
                ewald_name = ewald_line[0] + " " + ewald_line[1]
                VASP_data[ewald_name] = float(ewald_line[-1])
                
                hartree_line = f[i+4].split()
                hartree_name = hartree_line[0] + " " + hartree_line[1]
                VASP_data[hartree_name] = float(hartree_line[-1])
                
                exchange_line = f[i+5].split()
                VASP_data[exchange_line[0]] = float(exchange_line[-1])
            
                vxc_exc_line = f[i+6].split()
                VASP_data[vxc_exc_line[0]] = float(vxc_exc_line[-1])
                
                paw_line = f[i+7].split()
                paw_name = paw_line[0] + " " + paw_line[1] + " " + paw_line[2]
                VASP_data[paw_name] = float(paw_line[-2])
                
                entropy_line = f[i+8].split()
                entropy_name = entropy_line[0] + " " + entropy_line[1]
                VASP_data[entropy_name] = float(entropy_line[-1])
                
                eigval_line = f[i+9].split()
                VASP_data[eigval_line[0]] = float(eigval_line[-1])
                
                atomic_line = f[i+10].split()
                atomic_name = atomic_line[0] + " " + atomic_line[1]
                VASP_data[atomic_name] = float(atomic_line[-1])
                
                free_line = f[i+12].split()
                free_name = free_line[0] + " " + free_line[1]
                VASP_data[free_name] = float(free_line[-2])
                
                no_entropy_line = f[i+14].split()
                no_entropy_name = (no_entropy_line[0] + " " + no_entropy_line[1] + " " +
                                   no_entropy_line[2])
                VASP_data[no_entropy_name] = float(no_entropy_line[4])
                
                sigma_line = no_entropy_line
                sigma_name = sigma_line[5]
                VASP_data[sigma_name] = float(sigma_line[7])

            if "VOLUME and BASIS-vectors are now" in line:
                VASP_data["final unit cell volume"] = float(f[i+3].split()[-1])

                a1 = [float(a) for a in f[i+5].split()[:3]]
                b1 = [float(b) for b in f[i+5].split()[3:]]
                
                a2 = [float(a) for a in f[i+6].split()[:3]]
                b2 = [float(b) for b in f[i+6].split()[3:]]

                a3 = [float(a) for a in f[i+6].split()[:3]]
                b3 = [float(b) for b in f[i+6].split()[3:]]
                
                VASP_data["final lattice vectors"] = np.transpose([a1, a2, a3])
                VASP_data["final reciprocal lattice vectors"] = np.transpose([b1, b2, b3])
                VASP_data["final reciprocal unit cell volume"] = np.linalg.det(
                    np.transpose([b1, b2, b3]))

            if "FORCES acting on ions" in line:
                forces = []
                forces.append({"electron-ion force": [float(fi) for fi in
                                                      f[i + natoms + 4].split()[:3]]})
                forces.append({"ewald-force": [float(fi) for fi in
                                               f[i + natoms + 4].split()[3:6]]})
                forces.append({"non-local-force": [float(fi) for fi in
                                                   f[i + natoms + 4].split()[6:]]})
                VASP_data["net forces acting on ions"] = forces

            if "Elapsed time" in line:
                VASP_data["Elapsed time"] = float(line.split()[-1])
                
        VASP_data["number of plane waves"] = nplane_waves
        VASP_data["symmetry operators"] = sym_group
        
    return VASP_data


def create_INCAR(location):
    """ Create a template INCAR energy convergence tests.
    
    Args:
        location (str): the location of the VASP input files
    """

    incar_file = os.path.join(location, "INCAR")
    system = "Cu"
    new_EAUG = 2*EAUG
    with open(incar_file, "w") as file:
        file.write("Determine the correct number of bands. \n \n")

        file.write("Start parameters \n")
        file.write("  SYSTEM = " + system + " \n \n")

        file.write("Read WAVECAR (default = 1 or 0) \n")
        file.write("  ISTART = 0 \n \n")

        file.write("Set up initial orbitals (default = 1) \n")
        file.write("  INIWAV = 1 \n \n")

        file.write("Type of charge mixing (default = 4)\n")
        file.write("  IMIX = 4 \n \n")
        
        file.write("The l-quantum number of the one-center PAW charge densities "
                   "that pass to charge density mixer (default = 2) \n")
        file.write("  LMAXMIX = 6 \n \n")

        file.write("Charge density extrapolation (default = 1)\n")
        file.write("  IWAVPR = 1 \n \n")

        file.write("Number of bands (default = (NELECT + NIONS)/2 \n")
        file.write("  NBANDS =  \n \n")

        file.write("Projection operators (default = .FALSE.) \n")
        file.write("  LREAL = .False. \n \n")    

        file.write("Spin polarization (default = 1) \n")
        file.write("  ISPIN = 1 \n \n")

        file.write("van der Waals corrections (default = 0) \n")
        file.write("  IVDW = 0 \n \n")

        file.write("Write wavefunctions (default = .TRUE. \n")
        file.write("  LWAVE = .False. \n \n")

        file.write("Aspherical charge distribution correction (default = .FALSE.) \n")
        file.write("  LASPH = True \n \n")

        file.write("Local spin density approximation with strong intra-atomic "
                   "interaction (default = .FALSE.) \n")
        file.write("  LDAU = .FALSE. \n \n")

        file.write("Subspace diagonalization within the main algorithm "
                   "(default = .TRUE.) \n")
        file.write("  LDIAG = .TRUE. \n \n")

        file.write("Symmetry (default = 1 for USPP) \n")
        file.write("  ISYM = 1 \n \n")

        file.write("Write charge densities (default = .TRUE.) \n")
        file.write("  LCHARG = False \n \n")

        file.write("Electronic relaxation algorithm (default = Normal) \n")
        file.write("  ALGO = Normal \n \n")

        file.write("Algorithm to optimize the orbitals (default = 38 or "
                   "blocked-Davison algorithm) \n")
        file.write("  IALGO = 38 \n \n")

        file.write("Number of bands optimized simultaneously (default = 4) \n")
        file.write("  NSIM = 1 \n \n")

        file.write("Global break condition for electronic self-consistency loop "
                   "(default = 1E-4) \n")
        file.write("  EDIFF = 1E-10 \n \n")

        file.write("The minimum number of electronic self-consistency steps "
                   "(default = 4) \n")
        file.write("  NELMIN = 6 \n \n")

        file.write("The maximum number of electronic self-consistency steps "
                   "(default = 60) \n")
        file.write("  NELM = 60 \n \n")

        file.write("The number of non-self-consistency steps taken at the beginning "
                   "(default = -12 for IALGO = 48) \n")
        file.write("  NELMDL = -12 \n \n")

        file.write("Algorithm for how ions are updated and moved \n")
        file.write("  IBRION = 2 \n \n")

        file.write("Determines which principal degrees of freedom are allowed to "
                   "change (default = 2) \n")
        file.write("  ISIF = 3 \n \n")

        file.write("Global break condition for the ionic relaxation "
                   "(default = EDIFF*10) \n")
        file.write("  EDIFFG = 1E-9 \n \n")

        file.write("The maximum number of ionic steps (default = 0) \n")
        file.write("  NSW = 0 \n \n")

        file.write("Determines how the partial occupancies are set for each orbital "
                   "(default = 1) \n")  
        file.write("  ISMEAR = 0 \n \n")

        file.write("The width of the smearing parameter in eV (default = 0.2) \n")
        file.write("  SIGMA = 1E-3")

        file.write("Add an support grid for the evaluation of augmentation charges "
                   "(default = .FALSE.) \n")
        file.write("  ADDGRID = .TRUE. \n \n")

        file.write("The cut-off energy of the plane wave representation of the "
                   "augmentation charges (default = EAUG) \n")
        file.write("  ENAUG = %f \n \n" %new_EAUG)

        file.write("Set the FFT grids used in the exact exchange routines "
                   "(default = Normal) \n")
        file.write("  PRECFOCK = Accurate \n \n")

        file.write("Number of grid points for density of states (default = 301) \n")
        file.write("  NEDOS = 2000 \n \n")

        file.write("A relative stopping criterion for the optimization of an eigenvalue "
                   "(default = 0.3) \n")
        file.write("  DEPER = 1E-2 \n \n")

        file.write("The maximum weight for a band to be considered empty "
                   "(default = 0.001) \n")
        file.write("  WEIMIN = 1E-6 \n \n")

        
def read_potcar(location):
    """Read values from the POTCAR

    Args:
        location (str): the file path to the POTCAR
    """

    # Get EAUG from POTCAR.
    potcar_file = os.path.join(location, "POTCAR")

    
    data = {}
    ZVAL_list = []
    EAUG_list = []

    with open(potcar_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if "EAUG" in line:
                EAUG_list.append(float(line.split()[-1]))

            if "ZVAL" in line:
                for j, elem in enumerate(line.split()):
                    if elem == "ZVAL":
                        ZVAL_list.append(float(line.split()[j+2]))

    data["ZVAL list"] = ZVAL_list
    data["EAUG list"] = EAUG_list
    return data


def run_VASP(element, parameters):
    """A function that will take the values contained in parameters, create the
    file structure, and submit a series of VASP jobs. The directory from which 
    this command is run must contain run.py and run.sh files. It must contain a 
    directory named 'element', and this directory must contain a directory named 
    'initial data' in which is located the template VASP input files. The run.sh
    file runs the run.py script.
    
    Args:
        parameters (dict): a dictionary of adjustable parameters. The following
            key strings must be present: 'grid type list', 'offset list',
            'smearing list', 'smearing value list', and 'k-points list'. 
            Their corresponding values must be lists.
    """

    # A dictionary to give the folders more meaningful names
    smearing_dict = {"-1": "Fermi-Dirac smearing", 
                     "0": "Gaussian smearing", 
                     "1": "1st order Methfessel-Paxton smearing", 
                     "2": "2nd order Methfessel-Paxton smearing", 
                     "3": "3rd order Methfessel-Paxton smearing", 
                     "4": "4th order Methfessel-Paxton smearing", 
                     "5": "5th order Methfessel-Paxton smearing", 
                     "-4": "Tetrahedron method without Blochl corrections",
                     "-5": "Tetrahedron method with Blochl corrections"}

    home_dir = os.getcwd()
    
    # Move into the element directory.
    element_dir = os.path.join(home_dir, element)
    os.chdir(element_dir)

    # Locate the inital data.
    idata_loc = os.path.join(element_dir, "initial_data")

    # Make and move into the grid type directory.
    for grid_type in parameters["grid type list"]:
        grid_dir = os.path.join(element_dir, grid_type)
        if not os.path.isdir(grid_dir):
            os.mkdir(grid_dir)
        os.chdir(grid_dir)

        # Make and move into the offset directory
        for offset in parameters["offset list"]:
            offset_name = str(offset).strip("[").strip("]")
            offset_name = offset_name.replace(",", "")

            offset_dir = os.path.join(grid_dir, offset_name)
            if not os.path.isdir(offset_dir):
                os.mkdir(offset_dir)
            os.chdir(offset_dir)

            # Make and move into the smearing type directory.
            for smearing in parameters["smearing list"]:
                smearing_dir = os.path.join(offset_dir, smearing_dict[smearing])
                if not os.path.isdir(smearing_dir):
                    os.mkdir(smearing_dir)
                os.chdir(smearing_dir)

                # Make and move into the smearing value directory.
                for smearing_value in parameters["smearing value list"]:
                    smearing_value_dir = os.path.join(smearing_dir, str(smearing_value))
                    if not os.path.isdir(smearing_value_dir):
                        os.mkdir(smearing_value_dir)
                    os.chdir(smearing_value_dir)

                    # Make and move into the number of k-points directory.
                    for kpoints in parameters["k-point list"]:
                        nkpoints = np.prod(kpoints)
                        kpoints_dir = os.path.join(smearing_value_dir, str(nkpoints))
                        if not os.path.isdir(kpoints_dir):
                            os.mkdir(kpoints_dir)
                        os.chdir(kpoints_dir)

                        # Copy initial_data, run.sh and run.py to current directory.
                        subprocess.call("cp " + idata_loc + "/* ./", shell=True)
                        subprocess.call("cp " + home_dir + "/run.sh ./", shell=True)
                        subprocess.call("cp " + home_dir + "/run.py ./", shell=True)
                        subprocess.call('chmod +x run.py', shell=True)

                        # Correctly label this job.
                        # Read in the file.
                        with open('run.sh', 'r') as file:
                            filedata = file.read()

                        # Replace the target string.
                        filedata = filedata.replace('ELEMENT', str(element))
                        filedata = filedata.replace('GRIDTYPE', grid_type)
                        filedata = filedata.replace('OFFSET', str(offset))
                        filedata = filedata.replace('SMEAR', smearing_dict[smearing])
                        filedata = filedata.replace('SMRVALUE', str(smearing_value))
                        filedata = filedata.replace('KPOINT', str(nkpoints))

                        # Write the file out again.
                        with open('run.sh', 'w') as file:
                            file.write(filedata)

                        incar_dir = os.path.join(kpoints_dir, "INCAR")
                        # Replace values in the INCAR.
                        with open(incar_dir, "r") as file:
                            filedata = file.read()

                        filedata = filedata.replace("smearing method", smearing)
                        filedata = filedata.replace("smearing value",
                                                    str(smearing_value))

                        with open(incar_dir, "w") as file:
                            file.write(filedata)

                        # Replace values in the KPOINTS file.
                        vkpts_dir = os.path.join(kpoints_dir, "KPOINTS")
                        with open(vkpts_dir, "r") as file:
                            filedata = file.read()

                        for i,kp in enumerate(kpoints):
                            kp_name = "kpoint" + str(i + 1)
                            filedata = filedata.replace(kp_name, str(kp))

                        for j,off in enumerate(offset):
                            off_name = "offset" + str(j + 1)
                            filedata = filedata.replace(off_name, str(off))

                        with open(vkpts_dir, "w") as file:
                            file.write(filedata)

                        # Adjust time to run and memory
                        with open("run.sh", "r") as file:
                            filedata = file.read()

                        if nkpoints <= 8000:
                            filedata = filedata.replace("12:00:00", "4:00:00")
                            filedata = filedata.replace("4096", "8192")

                        elif nkpoints > 8000 and nkpoints < 27000:
                            filedata = filedata.replace("12:00:00", "6:00:00")
                            filedata = filedata.replace("4096", "16384")                                

                        elif nkpoints >= 27000 and nkpoints < 64000:
                            filedata = filedata.replace("12:00:00", "12:00:00")
                            filedata = filedata.replace("4096", "32768")

                        elif nkpoints >= 64000:
                            filedata = filedata.replace("12:00:00", "24:00:00")
                            filedata = filedata.replace("4096", "65536")

                        subprocess.call('sbatch run.sh', shell=True)

                        os.chdir(smearing_value_dir)
                    os.chdir(smearing_dir)
                os.chdir(offset_dir)
            os.chdir(grid_dir)
        os.chdir(element_dir)
    os.chdir(home_dir)
