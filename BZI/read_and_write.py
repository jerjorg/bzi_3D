import os
import subprocess
import numpy as np
import pandas as pd
import pickle
import itertools
import matplotlib.pyplot as plt
import xarray as xr


#######
####### =========================================================================
####### You must already have the element directories with populated initial_data
####### =========================================================================
#######


# Plotting marker
marker = itertools.cycle(('1', '2', '3', '4'))

def run_QE(home_dir, system_name, parameters):
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


    
    # Move into the element directory.
    system_dir = home_dir + "/" + system_name
    os.chdir(system_dir)
    
    # Locate the inital data.
    idata_loc = system_dir + "/" + "initial_data"
    
    # Make and move into the grid type directory.
    for grid_type in parameters["grid type list"]:
        grid_dir = system_dir + "/" + grid_type
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

                            # Replace a few values in run.py
                            with open('run.py', 'r') as file :
                                filedata = file.read()
                                
                            filedata = filedata.replace("system_name", system_name)
                            
                            with open('run.py', 'w') as file:
                                file.write(filedata)
                                                        
                            # Correctly label this job.
                            # Read in the file.
                            with open('run.sh', 'r') as file :
                                filedata = file.read()

                            # Replace the target string.
                            filedata = filedata.replace('SYSTEM', str(system_name))
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
                            with open(system_name + '.in', 'r') as file:
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
                            with open(system_name + '.in', 'w') as file:
                                file.write(filedata)

                            # If using a generalized Monkhorst-Pack grid, create a
                            # PRECALC file.
                            kp = nkpoints**(1./3)
                            min_dist = int(2.8074*kp - 3.4008)
                            with open("PRECALC", "w") as file:
                                file.write("BETA_MODE=TRUE\n")
                                file.write("INCLUDEGAMMA=FALSE\n")
                                file.write("MINDISTANCE=%i\n"%min_dist)
                                file.write("OFFSET=AUTO\n")

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

                            with open("run.sh", "w") as file:
                                file.write(filedata)

                            if grid_type == "Generalized Monkhorst-Pack":
                                subprocess.call("getKPointsBeta -qe " + system_name + ".in", shell=True)

                            # Setting a larger stack size should keep some jobs from segfaulting.
                            subprocess.call("ulimit -s unlimited", shell=True)

                            # Submit the job.
                            subprocess.call('sbatch run.sh', shell=True)
                            
                            os.chdir(smearing_value_dir)
                        os.chdir(smearing_dir)
                    os.chdir(occupation_dir)
                os.chdir(offset_dir)
            os.chdir(grid_dir)
        os.chdir(system_dir)
    os.chdir(home_dir)


def read_QE(location, system_name):
    """Create a dictionary of most of the data created during a
    single Quantum Espresso calculation.

    Args:
        location (str): the location of the Quantum Espresso calculation.
        system_name (str): the prefix of the input and output files.
    """
    
    QE_data = {}
    QE_data["self-consistent calculation time"] = []
    output_file = location + "/" + system_name + ".out"
    input_file = location + "/" + system_name + ".in"

    with open(input_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if "K_POINTS" in line:
                try:
                    kpt_index = i + 1
                    print(f[kpt_index])
                    kpt_line = f[kpt_index].split()
                    total_kpoints = (float(kpt_line[0])*float(kpt_line[1])*
                                     float(kpt_line[2]))
                    QE_data["number of unreduced k-points"] = total_kpoints
                except:
                    continue
            if "Server version" in line:
                kpt_line = line.split()
                QE_data["number of unreduced k-points"] = int(kpt_line[kpt_line.index("total") - 1])
    
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


def remove_QE_save(home_dir, system_name, parameters):
    """A function that will remove the save folder created during a Quantum Espresso
    run.
    
    Args:
        parameters (dict): a dictionary of adjustable parameters. The following
            key strings must be present: 'grid type list', 'offset list',
            'occupation list', 'smearing list', 'smearing value list',
            and 'k-points list'. Their corresponding values must be lists.
        home_dir (str): the root directory that contains a filder 'system_name'.
        system_name (str): the name of the system being considered.
    """
    
    save_file_name = system_name + ".save"
    
    # Make and move into the system directory.
    system_dir = home_dir + "/" + system_name
    os.chdir(system_dir)
        
    # Move into the grid type directory.
    for grid_type in parameters["grid type list"]:
        grid_dir = system_dir + "/" + grid_type
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
        os.chdir(system_dir)
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


def read_vasp_input(location):
    """Read in VASP input parameters. This is used in creating the INCAR.

    Args:
        location (str): the file path to the input files.

    Returns:
        VASP_data (dict): a dictionary of input parameters.
    """

    VASP_data = {}
    kpoints_file = os.path.join(location, "KPOINTS")
    poscar_file = os.path.join(location, "POSCAR")
    potcar_file = os.path.join(location, "POTCAR")

    VASP_data = {"ZVAL list": [],
                 "EAUG list": [],
                 "ENMAX list": []}

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
        VASP_data["scaling factor"] = float(f[1].split()[0])

        a1 = [float(v) for v in f[2].strip().split()[:3]]
        a2 = [float(v) for v in f[3].strip().split()[:3]]            
        a3 = [float(v) for v in f[4].strip().split()[:3]]
        VASP_data["lattice vectors"] = np.transpose([a1,a2,a3])

        atomic_basis = {}
        atomic_species = []
        for elem in f[5].split():
            try:
                elem = int(elem)
                atomic_species.append(elem)
            except:
                continue            
        natoms = int(np.sum(atomic_species))
        atomic_basis["number of atoms per atomic species"] = atomic_species
        atomic_basis["number of atoms"] = natoms
        atomic_basis["coordinates"] = f[6].strip()

        atom_positions = []
        for k in range(natoms):
            atom_positions.append([float(k) for k in f[7 + k].split()[:3]])
        atomic_basis["atom positions"] = atom_positions
        VASP_data["atomic basis"] = atomic_basis
                                
        # more = True
        # while more:
        #     try:
        #         atomic_basis = {}
        #         # species = f[j].strip()
        #         # atomic_basis["atomic species"] = species
        #         natoms = int(f[j+1].strip())
        #         atomic_basis["number of atoms"] = natoms
        #         atomic_basis["coordinates"] = f[j+2].strip()
        #         pos = []
        #         for n in range(natoms):
        #             pos.append([float(p) for p in f[j + 3 + n].strip().split()[:3]])
        #         atomic_basis["positions"] = pos
        #         atomic_bases.append(atomic_basis)
        #     except:
        #         more = False
        # VASP_data["atomic bases"] = atomic_bases

         
    with open(potcar_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if "EAUG" in line:
                VASP_data["EAUG list"].append(float(line.split()[-1]))

            if "ZVAL" in line:
                for j, elem in enumerate(line.split()):
                    if elem == "ZVAL":
                        VASP_data["ZVAL list"].append(float(line.split()[j+2]))

            if "ENMAX" in line:
                VASP_data["ENMAX list"].append(float(line.split()[2].strip(";")))


    return VASP_data


def read_vasp(location):
    """Read in VASP output parameters.

    Args:
        location (str): the file path to the ouput files.

    Returns:
        VASP_data (dict): a dictionary of parameters.
    """

    job_finished = False
    VASP_data = read_vasp_input(location)
    natoms = len(VASP_data["atomic basis"]["atom positions"])
    outcar_file = os.path.join(location, "OUTCAR")
    eigenval_file = os.path.join(location, "EIGENVAL")
    oszicar_file = os.path.join(location, "OSZICAR")
    doscar_file = os.path.join(location, "DOSCAR")

    if not all([os.path.exists(oszicar_file),
                os.path.exists(outcar_file),
                os.path.exists(eigenval_file),
                os.path.exists(eigenval_file)]):
        return None

    # Right now it only counts the number of electronic self-consistency steps.
    with open(oszicar_file, "r") as file:
        f = file.readlines()
        niters = 0
        for i, line in enumerate(f):
            if "DAV" in line:
                niters += 1

        VASP_data["number of electronic iterations"] = niters
    
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


    energy_list = []
    dos_list = []
    idos_list = [] # integrated DOS
    
    with open(doscar_file, "r") as file:
        f = file.readlines()
        for i, line in enumerate(f):
            if i > 5:
                energy_list.append(float(line.split()[0]))
                dos_list.append(float(line.split()[1]))
                idos_list.append(float(line.split()[2]))
        VASP_data["density of states data"] = dos_list
        VASP_data["integrated density of states data"] = idos_list
        VASP_data["density of states energies"] = energy_list
    

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
                try:
                    VASP_data["NBANDS"] = int(line.split()[-1])
                except:
                    continue
                
            if "plane waves" in line:
                nplane_waves.append(int(line.split()[-1]))

            # if "the Fermi energy is" in line:
            #     VASP_data["Fermi level"] = line.split()[4] + " " + line.split()[5]
            if "E-fermi" in line:
                VASP_data["Fermi level"] = float(line.split()[2])
                
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

                # I ran into problems where VASP would calculate very large
                # values for the band energy that eventually prevented it from
                # converging.
                try:                    
                    VASP_data[eigval_line[0]] = float(eigval_line[-1])
                except:
                    return None
                
                atomic_line = f[i+10].split()
                atomic_name = atomic_line[0] + " " + atomic_line[1]
                VASP_data[atomic_name] = float(atomic_line[-1])
                
                free_line = f[i+12].split()
                free_name = free_line[0] + " " + free_line[1]
                
                VASP_data[free_name] = float(free_line[-2])
                no_entropy_line = f[i+14].split()
                no_entropy_name = (no_entropy_line[0] + " " + no_entropy_line[1] + " " +
                                   no_entropy_line[2])
                # For some reason splitting the lines isn't always consistent, and the "=" gets
                # attached to the value occasionally.
                try:
                    VASP_data[no_entropy_name] = float(no_entropy_line[4])
                    sigma_line = no_entropy_line
                    sigma_name = sigma_line[5]
                    VASP_data[sigma_name] = float(sigma_line[7])                    
                except:
                    VASP_data[no_entropy_name] = float(no_entropy_line[3].strip("="))
                    sigma_line = no_entropy_line
                    sigma_name = sigma_line[4]
                    VASP_data[sigma_name] = float(sigma_line[5].strip("="))

            if "VOLUME and BASIS-vectors are now" in line:
                VASP_data["Final unit cell volume"] = float(f[i+3].split()[-1])

                a1 = [float(a) for a in f[i+5].split()[:3]]
                b1 = [float(b) for b in f[i+5].split()[3:]]
                
                a2 = [float(a) for a in f[i+6].split()[:3]]
                b2 = [float(b) for b in f[i+6].split()[3:]]

                a3 = [float(a) for a in f[i+6].split()[:3]]
                b3 = [float(b) for b in f[i+6].split()[3:]]
                
                VASP_data["Final lattice vectors"] = np.transpose([a1, a2, a3])
                VASP_data["Final reciprocal lattice vectors"] = np.transpose([b1, b2, b3])
                VASP_data["Final reciprocal unit cell volume"] = np.linalg.det(
                    np.transpose([b1, b2, b3]))

            if "FORCES acting on ions" in line:
                forces = []
                forces.append({"Electron-ion force": [float(fi) for fi in
                                                      f[i + natoms + 4].split()[:3]]})
                forces.append({"Ewald-force": [float(fi) for fi in
                                               f[i + natoms + 4].split()[3:6]]})
                forces.append({"Non-local-force": [float(fi) for fi in
                                                   f[i + natoms + 4].split()[6:]]})
                VASP_data["Net forces acting on ions"] = forces

            if "Elapsed time" in line:
                VASP_data["Elapsed time"] = float(line.split()[-1])

            if "total charge-density along one line" in line:
                wrapped_charge_list = []
                directions = ["x", "y", "z"]
                
                for direct in directions:
                    too_far = False
                    k = i
                    wrapped_charge = None
                    while not too_far:
                        if direct in f[k].split():
                            wrapped_charge = float(f[k].split()[-2])
                        elif "pseudopotential strength for first" in f[k]:
                            too_far = True
                        k += 1
                    wrapped_charge_list.append(wrapped_charge)
                VASP_data["total wrapped charge"] = wrapped_charge_list

            # Locate where the wrap-around charge density is listed.
            if "soft charge-density along one line" in line:
                wrapped_charge_list = []                
                directions = ["x", "y", "z"]
                
                for j,direct in enumerate(directions):
                    too_far = False                
                    wrapped_charge = None
                    k = i
                    while not too_far:
                        if direct in f[k].split():
                            wrapped_charge = float(f[k].split()[-2])
                        elif "total charge-density along" in f[k]:
                            too_far = True
                        k += 1
                    wrapped_charge_list.append(wrapped_charge)
                VASP_data["total wrapped soft charge"] = wrapped_charge_list
                        
            if "total drift" in line:
                VASP_data["Drift force"] = np.linalg.norm([float(w) for w in
                                                           line.split()[2:]])

            if "TOTAL-FORCE" in line:
                forces = f[i-3].split()
                iforces = []
                for j in range(0,10,3):
                    iforces.append(np.linalg.norm([float(f) for f in forces[j:j+3]]))
                VASP_data["Electron-ion force"] = iforces[0]
                VASP_data["Ewald force"] = iforces[1]
                VASP_data["Non-local force"] = iforces[2]
                VASP_data["Convergence-correction force"] = iforces[3]

            if "General timing and accounting informations for this job:" in line:
                job_finished = True
                
        VASP_data["number of plane waves"] = nplane_waves
        VASP_data["symmetry operators"] = sym_group
        
    if job_finished == True:
        return VASP_data
    else:
        return None


# def read_VASP(location):
#     """Create a dictionary of most of the data created during a
#     single VASP calculation.

#     Args:
#         location (str): the location of the VASP calculation.
#     """

#     VASP_data = {}
#     incar_file = location + "/INCAR"
#     kpoints_file = location + "/KPOINTS"
#     poscar_file = location + "/POSCAR"
#     ibzkpt_file = location + "/IBZKPT"
#     eigenval_file = location + "/EIGENVAL"
#     outcar_file = location + "/OUTCAR"
#     potcar_file = location + "/POTCAR"
    
#     # # Get the number of unreduced k-points. This only works for one of 
#     # # the automatic k-mesh generation methods, where the number of k-points
#     # # and offset of the k-mesh are provided.
#     # with open(kpoints_file, "r") as file:
#     #     f = file.readlines()
#     #     for i, line in enumerate(f):
#     #         if "Gamma" in line:
#     #             kpt_index = i + 1
#     #             kpt_line = f[kpt_index].split()
#     #             total_kpoints = np.prod([int(k) for k in kpt_line])
                
#     #             VASP_data["number of unreduced k-points"] = total_kpoints
                
#     #             offset_index = i + 2
#     #             offset_line = f[offset_index].split()

#     #             VASP_data["offset"] = [float(off) for off in offset_line]

#     # """
#     # The POSCAR should have the following format:

#     # Al_FCC # comment line
#     # 1 # universal scaling factor
#     # 4.00 0.00 0.00 # first Bravais lattice vector
#     # 0.00 4.00 0.00 # second Bravais lattice vector
#     # 0.00 0.00 4.00 # third Bravais lattice vector
#     # Al # 
#     # 4 # number of atoms per species, be consistent:
#     # direct or cartesian, only first letter is significant
#     # 0.00000000 0.00000000 0.0000000 positions
#     # 0.50000000 0.00000000 0.5000000
#     # 0.50000000 0.50000000 0.0000000
#     # 0.00000000 0.50000000 0.5000000            
#     # """

#     # # This section creates a list of atomic basis dictionaries. These dictionaries
#     # # contain the atomic species, number of atoms, coordinates, and positions.
#     # with open(poscar_file, "r") as file:
#     #     f = file.readlines()
#     #     VASP_data["name of system"] = f[0].strip()

#     #     # If negative, the scaling factor should be interpreted as the total volume
#     #     # of the cell.
#     #     VASP_data["scaling factor"] = float(f[1].split()[0])

#     #     a1 = [float(v) for v in f[2].strip().split()]
#     #     a2 = [float(v) for v in f[3].strip().split()]            
#     #     a3 = [float(v) for v in f[4].strip().split()]
#     #     VASP_data["lattice vectors"] = np.transpose([a1,a2,a3])

#     #     j = 5
#     #     atomic_bases = []
#     #     more = True
#     #     while more:
#     #         try:
#     #             atomic_basis = {}
#     #             species = f[j].strip()
#     #             atomic_basis["atomic species"] = species
#     #             natoms = int(f[j+1].strip())
#     #             more = False
#     #             atomic_basis["number of atoms"] = natoms
#     #             atomic_basis["coordinates"] = f[j+2].strip()
#     #             pos = []
#     #             for n in range(natoms):
#     #                 pos.append([float(p) for p in f[j + 3 + n].strip().split()[:3]])
#     #             atomic_basis["positions"] = pos
#     #             atomic_bases.append(atomic_basis)
#     #         except:
#     #             more = False
#     #     VASP_data["atomic bases"] = atomic_bases


#     # # If there is any special formating, such as LREALS=.FALSE.,
#     # # then reading the INCAR won't work.
#     # with open(incar_file, "r") as file:
#     #     f = file.readlines()
#     #     for i, line in enumerate(f):
#     #         vals = line.strip("").split()
#     #         VASP_data[vals[0]] = vals[-1]

#     # Read the eigenvalues, number of reduced k-points, and the
#     # k-points weights from the EIGENVAL file.
#     with open(eigenval_file, "r") as file:
#         f = file.readlines()
#         kpoints = []
#         weights = []
#         nkpts = 0
#         degeneracy = []
#         for i,line in enumerate(f):
#             if line.strip() == "":
#                 nkpts += 1
#                 vals = [float(k) for k in f[i+1].strip().split()]
#                 kpoints.append(vals[:3])
#                 weights.append(vals[-1])
#                 degeneracy.append(vals[-1]*VASP_data["number of unreduced k-points"])
#         VASP_data["number of reduced k-points"] = nkpts
#         VASP_data["k-point weights"] = weights
#         VASP_data["reduced k-points"] = kpoints
#         VASP_data["k-point degeneracy"] = degeneracy
        
#     with open(outcar_file, "r") as file:
#         f = file.readlines()
#         sym_group = []
#         nplane_waves = []
#         for i, line in enumerate(f):

#             if 'isymop' in line:
#                 r1 = [float(q) for q in line.split()[1:]]
#                 r2 = [float(q) for q in f[i+1].split()]            
#                 r3 = [float(q) for q in f[i+2].split()]            
#                 op = np.array([r1, r2, r3])
#                 sym_group.append(op)

#             if "NBANDS" in line:
#                 VASP_data["NBANDS"] = int(line.split()[-1])
                
                
#             if "plane waves" in line:
#                 nplane_waves.append(int(line.split()[-1]))
                
#             if " Free energy of the ion-electron system (eV)" in line:
#                 alpha_line = f[i+2].split()
#                 alpha_name = alpha_line[0] + " " + alpha_line[1]
#                 VASP_data[alpha_name] = float(alpha_line[-1])
                
#                 ewald_line = f[i+3].split()
#                 ewald_name = ewald_line[0] + " " + ewald_line[1]
#                 VASP_data[ewald_name] = float(ewald_line[-1])
                
#                 hartree_line = f[i+4].split()
#                 hartree_name = hartree_line[0] + " " + hartree_line[1]
#                 VASP_data[hartree_name] = float(hartree_line[-1])
                
#                 exchange_line = f[i+5].split()
#                 VASP_data[exchange_line[0]] = float(exchange_line[-1])
            
#                 vxc_exc_line = f[i+6].split()
#                 VASP_data[vxc_exc_line[0]] = float(vxc_exc_line[-1])
                
#                 paw_line = f[i+7].split()
#                 paw_name = paw_line[0] + " " + paw_line[1] + " " + paw_line[2]
#                 VASP_data[paw_name] = float(paw_line[-2])
                
#                 entropy_line = f[i+8].split()
#                 entropy_name = entropy_line[0] + " " + entropy_line[1]
#                 VASP_data[entropy_name] = float(entropy_line[-1])
                
#                 eigval_line = f[i+9].split()
#                 VASP_data[eigval_line[0]] = float(eigval_line[-1])
                
#                 atomic_line = f[i+10].split()
#                 atomic_name = atomic_line[0] + " " + atomic_line[1]
#                 VASP_data[atomic_name] = float(atomic_line[-1])
                
#                 free_line = f[i+12].split()
#                 free_name = free_line[0] + " " + free_line[1]
#                 VASP_data[free_name] = float(free_line[-2])
                
#                 no_entropy_line = f[i+14].split()
#                 no_entropy_name = (no_entropy_line[0] + " " + no_entropy_line[1] + " " +
#                                    no_entropy_line[2])
#                 VASP_data[no_entropy_name] = float(no_entropy_line[4])
                
#                 sigma_line = no_entropy_line
#                 sigma_name = sigma_line[5]
#                 VASP_data[sigma_name] = float(sigma_line[7])

#             if "VOLUME and BASIS-vectors are now" in line:
#                 VASP_data["final unit cell volume"] = float(f[i+3].split()[-1])

#                 a1 = [float(a) for a in f[i+5].split()[:3]]
#                 b1 = [float(b) for b in f[i+5].split()[3:]]
                
#                 a2 = [float(a) for a in f[i+6].split()[:3]]
#                 b2 = [float(b) for b in f[i+6].split()[3:]]

#                 a3 = [float(a) for a in f[i+6].split()[:3]]
#                 b3 = [float(b) for b in f[i+6].split()[3:]]
                
#                 VASP_data["final lattice vectors"] = np.transpose([a1, a2, a3])
#                 VASP_data["final reciprocal lattice vectors"] = np.transpose([b1, b2, b3])
#                 VASP_data["final reciprocal unit cell volume"] = np.linalg.det(
#                     np.transpose([b1, b2, b3]))

#             if "FORCES acting on ions" in line:
#                 forces = []
#                 forces.append({"electron-ion force": [float(fi) for fi in
#                                                       f[i + natoms + 4].split()[:3]]})
#                 forces.append({"ewald-force": [float(fi) for fi in
#                                                f[i + natoms + 4].split()[3:6]]})
#                 forces.append({"non-local-force": [float(fi) for fi in
#                                                    f[i + natoms + 4].split()[6:]]})
#                 VASP_data["net forces acting on ions"] = forces

#             if "Elapsed time" in line:
#                 VASP_data["Elapsed time"] = float(line.split()[-1])
                
#         VASP_data["number of plane waves"] = nplane_waves
#         VASP_data["symmetry operators"] = sym_group
        
#     return VASP_data


def create_INCAR(location):
    """ Create a template INCAR energy convergence tests.
    
    Args:
        location (str): the location of the VASP input files
    """

    vasp_data = read_vasp_input(location)

    zval = np.sum(vasp_data["ZVAL list"])
    new_zval = 2*zval
    eaug = max(vasp_data["EAUG list"])
    new_eaug = 2*eaug

    enmax = max(vasp_data["ENMAX list"])
    new_enmax = 2*enmax
    
    incar_file = os.path.join(location, "INCAR")
    system = vasp_data["name of system"]
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
        file.write("  NBANDS = %i\n \n"%new_zval)

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
        file.write("  NSW = 20 \n \n")

        file.write("Determines how the partial occupancies are set for each orbital "
                   "(default = 1) \n")  
        file.write("  ISMEAR = 0 \n \n")

        file.write("The width of the smearing parameter in eV (default = 0.2) \n")
        file.write("  SIGMA = 1E-3 \n \n")

        file.write("Add an support grid for the evaluation of augmentation charges "
                   "(default = .FALSE.) \n")
        file.write("  ADDGRID = .TRUE. \n \n")

        file.write("The cut-off energy of the plane wave representation of the "
                   "augmentation charges (default = EAUG) \n")
        file.write("  ENAUG = %f \n \n" %new_eaug)

        file.write("The cut-off energy of the plane wave representation of the "
                   "charge density (default = ENMAX) \n")
        file.write("  ENCUT = %f \n \n" %new_enmax)
        
        file.write("Set the FFT grids used in the exact exchange routines "
                   "(default = Normal) \n")
        file.write("  PRECFOCK = Accurate \n \n")

        file.write("Set the size of the FFT grids (default = Normal) \n")
        file.write("  PREC = Accurate \n \n")        

        file.write("Number of grid points for density of states (default = 301) \n")
        file.write("  NEDOS = 2000 \n \n")

        file.write("A relative stopping criterion for the optimization of an eigenvalue "
                   "(default = 0.3) \n")
        file.write("  DEPER = 1E-2 \n \n")

        file.write("The maximum weight for a band to be considered empty "
                   "(default = 0.001) \n")
        file.write("  WEIMIN = 1E-6 \n \n")


def make_QE_input(location, pseudo_dir, system_params):
    """Make a Quantum espresso input file based on reasonable input parameters.

    Args:
        location (str): the file path to the directory where the input file will be saved.
        pseudo_file (str): the file path to the pseudopotential files.
        system_params (dict): a dictionary of system parameters. In particular, it
            should contain information about the atomic basis and crystal lattice.
    """

    input_file = os.path.join(location, system_params["system name"] + ".in")
    pseudo_files = []

    for file in os.listdir(pseudo_dir):
        if ".UPF" in file:
            pseudo_files.append(os.path.join(pseudo_dir, file))

    # Get the energy cutoffs and number of bands from the pseudopotential.
    QE_data = [read_QE_pseudopot(pf) for pf in pseudo_files]

    wfc_cutoff = max([qe["wavefunction cutoff"] for qe in QE_data])
    rho_cutoff = max([qe["charge density cutoff"] for qe in QE_data])
    nbands = np.sum([qe["valency"] for qe in QE_data])


    natoms = 0
    for atom in system_params["atomic species"]:
        for pos in system_params[atom]["positions"]:
            natoms += 1

    if np.isclose(wfc_cutoff, 0):
        wfc_cutoff = 100

    if np.isclose(rho_cutoff, 0):
        rho_cutoff = 12*wfc_cutoff

    with open(input_file, "w") as file:
        file.write("&CONTROL \n")
        file.write("  calculation = 'relax' \n")
        file.write("  verbosity = 'low' \n")
        file.write("  wf_collect = .false. \n") # don't write the wavefunctions to file
        file.write("  nstep = 100 \n")
        file.write("  tstress = .true. \n")
        file.write("  tprnfor = .true. \n")
        file.write("  prefix = '" + system_params["system name"] + "' \n")
        file.write("  etot_conv_thr = 1.0d-9 \n")
        file.write("  forc_conv_thr = 1.0d-8 \n")
        file.write("  disk_io = 'low' \n")
        file.write("  pseudo_dir = '" + system_params["pseudopotential directory"] + "' \n")
        file.write("/ \n \n")

        file.write("&SYSTEM \n")
        file.write("  ibrav = 0 \n")
        file.write("  nat = " + str(natoms)  + "\n")
        file.write("  ntyp = 1 \n")
        file.write("  nbnd = 20 \n")
        file.write("  ecutwfc = " + str(wfc_cutoff) + " \n")
        file.write("  ecutrho = " + str(rho_cutoff) + " \n")
        file.write("  ecutfock = " + str(rho_cutoff) + " \n")
        file.write("  nosym = .false. \n")
        file.write("  occupations = '" + system_params["occupations"] + "' \n")
        file.write("  degauss = " + system_params["smearing value"] + " \n")
        file.write("  smearing = '" + system_params["smearing method"] + "' \n")
        file.write("/ \n \n")

        file.write("&ELECTRONS \n")
        file.write("  electron_maxstep = 200 \n")
        file.write("  conv_thr = 1.0d-8 \n")
        file.write("  mixing_beta = 0.7 \n")
        file.write("/ \n \n")

        # The dictionary parameters contains a key 'atomic species' which contains a list
        # of all the elements in the system.
        
        # The dictionary parameters contains a key for each atomic species. The value
        # for each atomic species is another dictionary that contains the atomic mass,
        # positions, and the name of the pseudopotential file.
        file.write("ATOMIC_SPECIES \n")
        for atom in system_params["atomic species"]:
            file.write(" " + atom + " " + str(system_params[atom]["atomic mass"]) + " "
                       + system_params[atom]["pseudopotential file"] + " \n")
        file.write("\n")
        
        # file.write("Al 26.982 Al.pbe-high.UPF \n \n")
        
        file.write("ATOMIC_POSITIONS angstrom \n")
        for atom in system_params["atomic species"]:
            for pos in system_params[atom]["positions"]:
                atp = (" " + atom + " " + str(pos[0]) +
                       " " + str(pos[1]) + " " + str(pos[2]) + "\n")
                file.write(atp)
        file.write("\n")
        # file.write("Al 0.00000000 0.00000000 0.00000000 \n \n")
        file.write("K_POINTS automatic \n")
        file.write(" 20 20 20 1 1 1 \n \n")

        v11 = str(system_params["lattice vectors"][0,0])
        v12 = str(system_params["lattice vectors"][1,0])
        v13 = str(system_params["lattice vectors"][2,0])

        v21 = str(system_params["lattice vectors"][0,1])
        v22 = str(system_params["lattice vectors"][1,1])
        v23 = str(system_params["lattice vectors"][2,1])

        v31 = str(system_params["lattice vectors"][0,2])
        v32 = str(system_params["lattice vectors"][1,2])
        v33 = str(system_params["lattice vectors"][2,2])

        file.write("CELL_PARAMETERS angstrom \n")
        file.write(" " + v11 + " " + v12 + " " + v13 + "\n")
        file.write(" " + v21 + " " + v22 + " " + v23 + "\n")
        file.write(" " + v31 + " " + v32 + " " + v33 + "\n\n")

        
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

                        # Setting a larger stack size should keep some jobs from segfaulting.
                        subprocess.call("ulimit -s unlimited", shell=True)

                        # Submit the job.
                        subprocess.call('sbatch run.sh', shell=True)

                        os.chdir(smearing_value_dir)
                    os.chdir(smearing_dir)
                os.chdir(offset_dir)
            os.chdir(grid_dir)
        os.chdir(element_dir)
    os.chdir(home_dir)

def setup_nbands_tests(location, system_name, nbands_list):
    """Create the file tree and submit jobs on the supercomputer for
    testing the number of bands in the INCAR.

    Args:
        location (str): the root directory. It must have a folder named
            'system_name' and this folder must contain a folder named 'initial_data',
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        nbands_list (list): a list of the number of bands for the simulations.
    """

    system_dir = os.path.join(location, system_name)
    data_dir = os.path.join(system_dir, "initial_data")

    vasp_data = read_vasp_input(data_dir)
    zval = np.sum(vasp_data["ZVAL list"])
    new_zval = 2*zval
    eaug = max(vasp_data["EAUG list"])
    new_eaug = 2*eaug

    # Make a directory where testing the number of bands will be performed.
    nbands_dir = os.path.join(system_dir, "nbands")
    if not os.path.isdir(nbands_dir):
        os.mkdir(nbands_dir)
    os.chdir(nbands_dir)

    for band_n in nbands_list:
        band_dir = os.path.join(nbands_dir, str(band_n))
        if not os.path.isdir(band_dir):
            os.mkdir(band_dir)
        os.chdir(band_dir)
        subprocess.call("cp " + data_dir + "/* .", shell=True)
        subprocess.call("cp " + location + "/run.py .", shell=True)
        subprocess.call("cp " + location + "/run.sh .", shell=True)
            
        create_INCAR(band_dir)
        
        incar_dir = os.path.join(band_dir, "INCAR")
        run_dir = os.path.join(band_dir, "run.sh")

        # Correctly label this job and adjust runtime if necessary.
        with open(run_dir, 'r') as file:
            filedata = file.read()
        
        job_name = system_name + " nbands " + str(band_n)
        filedata = filedata.replace("JOB NAME", job_name)
        filedata = filedata.replace("12:00:00", "4:00:00")
        filedata = filedata.replace("4096", "8192")

        with open(run_dir, 'w') as file:
            file.write(filedata)

        # Replace the number of bands and change the initial charge density.
        # The manual says that 1E-6 accuracy should be obtained within 10-15 iterations.
        # I'll use a stricter energy tolerance.
        with open(incar_dir, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("NBANDS = %i"%new_zval, "NBANDS = " + str(band_n))
        filedata = filedata.replace("ICHARG = 2", "ICHARG = 12")

        # When I kept the convergence tolerance very small (1e-10, 1e-9) the energy convergence        
        filedata = filedata.replace("EDIFF = 1E-10", "EDIFF = 1E-6")
        filedata = filedata.replace("EDIFFG = 1E-9", "EDIFF = 1E-5")
        filedata = filedata.replace("PREC = Accurate", "PREC = Normal")

        with open(incar_dir, "w") as file:
            file.write(filedata)
        
        # Setting a larger stack size should keep some jobs from segfaulting.
        subprocess.call("ulimit -s unlimited", shell=True)

        # Submit the job.        
        subprocess.call('sbatch run.sh', shell=True)


def gen_nbands_plots(system_dir):
    """Generate a plot of the number of bands against the number of
    iterations. Increasing the number of bands should reduce the number of 
    self-consistency iterations. The number of bands at which the number of 
    iterations doesn't reduce is the value that should be put in the INCAR.
    If it never stops reducing, or doesn't converge, than any value that requires
    fewer than 10-15 iterations will suffice.

    Args:
        system_name (str): the file path to the system being simulated. This folder
            must contain subfolders that contain VASP simulations with varying number of 
            bands.
    """
        
    # Initialize quantities that will be plotted.
    nbands_list = []
    iterations_list = []

    system_name = system_dir.split(os.sep)[-1]
    nbands_test_dir = os.path.join(system_dir, "nbands")
    bands_list = os.listdir(nbands_test_dir)
    count = 0
    
    try:
        bands_list.remove("plots")
    except:
        None
    bands_list = [int(b) for b in bands_list]

    # It's possible some of the runs failed. This list will contain the
    # number of bands for those that didn't.
    plotting_bands_list = []
    
    for bands in bands_list:
        band_dir = os.path.join(nbands_test_dir, str(bands))
        # oszicar_dir = os.path.join(band_dir, "OSZICAR")
        vasp_data = read_vasp(band_dir)
        
        if vasp_data != None:
            iterations_list.append(vasp_data["number of electronic iterations"])
            plotting_bands_list.append(bands)
        else:
            count += 1
            bands_list.remove(bands)
        
        # with open(oszicar_dir, "r") as file:
        #     f = file.readlines()
        #     for line in f:
        #         if "DAV" in line:
        #             niters += 1

        # iterations_list.append(niters)
        
    fig, ax = plt.subplots()
    ax.axhline(y=10 , color="black", linestyle="-")
    ax.scatter(plotting_bands_list, iterations_list)
    ax.set_title("Number of iterations required for 1E-6 accuracy")
    ax.set_xlabel("Number of bands")
    ax.set_ylabel("Number of iterations")
    plot_dir = os.path.join(nbands_test_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plot_name = os.path.join(plot_dir, "nbands_convergence.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

    
def setup_encut_tests(location, system_name, encut_list):
    """Create the file structure and submit jobs that will compare various
    values of the energy cutoff to find one that is adequate.

    Args:
        location (str): the root directory. It must have a folder named
            'system_name' and this folder must contain a folder named 'initial_data',
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        encut_list (list): a list of energy cutoffs that are fractions of the 
            default energy cutoff.
    """
    
    system_dir = os.path.join(location, system_name)
    data_location = os.path.join(system_dir, "initial_data")

    # Get the energy cutoff and number of bands from the POTCAR.
    # These are relevant for building the INCAR.
    vasp_data = read_vasp_input(data_location)

    enmax = max(vasp_data["ENMAX list"])
    new_enmax = 2*enmax

    # Make a directory where testing the energy cutoff will be performed.
    encut_test_dir = os.path.join(system_dir, "encut")
    if not os.path.isdir(encut_test_dir):
        os.mkdir(encut_test_dir)
    os.chdir(encut_test_dir)

    encut_list = [i*enmax for i in encut_list]

    # One of the tests will compare the total energy of two VASP calculations
    # with the atoms of one run shifted with respect to the other.
    shift_list = [[0, 0, 0], [.5, .4, .3]]
    symmetry_list = [0, 1]

    for symmetry in symmetry_list:
        sym_dir = os.path.join(encut_test_dir, str(symmetry))
        if not os.path.isdir(sym_dir):
            os.mkdir(sym_dir)
        os.chdir(sym_dir)
        
        for shift in shift_list:
            shift_dir = os.path.join(sym_dir, str(shift))
            if not os.path.isdir(shift_dir):
                os.mkdir(shift_dir)
            os.chdir(shift_dir)
            
            for encut in encut_list:
                encut_dir = os.path.join(shift_dir, str(encut))
                if not os.path.isdir(encut_dir):
                    os.mkdir(encut_dir)
                os.chdir(encut_dir)

                # Copy the required VASP input files into the energy cutoff directory.
                # Also copy the batch job and python script.
                subprocess.call("cp " + data_location + "/* .", shell=True)
                subprocess.call("cp " + location + "/run.py .", shell=True)
                subprocess.call("cp " + location + "/run.sh .", shell=True)
                    
                create_INCAR(encut_dir)
                incar_dir = os.path.join(encut_dir, "INCAR")
                run_dir = os.path.join(encut_dir, "run.sh")

                # Correctly label this job, and adjust runtime and memory if necessary.
                with open(run_dir, 'r') as file:
                    filedata = file.read()
                    
                filedata = filedata.replace("ELEMENT", system_name)
                filedata = filedata.replace("GRIDTYPE", str(shift[0]))
                filedata = filedata.replace("SMEAR KPOINT", "encut " + str(encut))
                filedata = filedata.replace("12:00:00", "4:00:00")
                filedata = filedata.replace("4096", "8192")

                with open(run_dir, 'w') as file:
                    file.write(filedata)
                    
                # Change the energy cutoff, and decrease the precision since it affects
                # the energy cutoff, and it isn't clear whether the energy cutoff prescribed
                # by PREC will overwrite that of ENCUT. Also turn off or leave on symmetry.
                with open(incar_dir, "r") as file:
                    filedata = file.read()

                filedata = filedata.replace("ENCUT = %f"%new_enmax, "ENCUT = %f"%encut)
                filedata = filedata.replace("PREC = Accurate", "PREC = Normal")
                filedata = filedata.replace("ISYM = 1", "ISYM = %i"%symmetry)

                with open(incar_dir, "w") as file:
                    file.write(filedata)


                # Setting a larger stack size should keep some jobs from segfaulting.
                subprocess.call("ulimit -s unlimited", shell=True)
                    
                # Submit the job.
                subprocess.call('sbatch run.sh', shell=True)


def gen_encut_plots(system_dir):
    """Generate plots that help identify the appropriate value for the energy encut.

    Args:
        system_name (str): the file path to the system being simulated. This folder
            must contain subfolders that contain VASP simulations with varying energy
            cutoffs.
    """

    # Initialize all quantities that'll be plotted.
    cutoff_plot_list = []
    cutoff_lists = []
    encut_list = []
    tot_wrapped_charge_list = [[],[],[]]
    drift_force_list = []
    cell_volume_list = []
    lat_vec_norm_list = []
    force_list = [[] for _ in range(5)]
    directions = ["x", "y", "z"]
    force_type_list = ["Electron-ion force", "Ewald force", "Non-local force",
                       "Convergence-correction force", "Drift force"]
    shift_list = [[0, 0, 0], [.5, .4, .3]]
    symmetry_list = [0, 1]

    system_name = system_dir.split(os.sep)[-1]
    elem_dir = os.path.join(system_dir, system_name)
    encut_test_dir = os.path.join(system_dir, "encut")
    os.chdir(encut_test_dir)

    # These tests don't depend on the different atomic shifts or turning off symmetry.
    tests1_dir = os.path.join(encut_test_dir, str(symmetry_list[0]), str(shift_list[1]))
    energy_cutoff_list = os.listdir(tests1_dir)
    for encut in energy_cutoff_list:
        cell_vol_bool = False
        contained = False
        encut_dir = os.path.join(tests1_dir, encut)
        os.chdir(encut_dir)
        outcar_dir = os.path.join(encut_dir, "OUTCAR")

        vasp_data = read_vasp(encut_dir)
        if vasp_data != None:
            # Get the cutoff energy for this run, and add it to the list of energies.
            cutoff_plot_list.append(float(encut))

            # Find the wrapped charge density in each of the directions.
            for i in range(3):
                tot_wrapped_charge_list[i].append(vasp_data["total wrapped charge"][i])

            # Find all the forces, and the change in cell volume and shape.
            drift_force_list.append(vasp_data["Drift force"])
            cell_volume_list.append(vasp_data["Final unit cell volume"])
            lat_vec_norm_list.append(np.linalg.norm(vasp_data["Final lattice vectors"]))
            for j in range(5):
                force_list[j].append(vasp_data[force_type_list[j]])

    # A dictionary that goes between the VASP parameter and it's meaning, useful
    # for labeling plots and files.
    sym_label_dict = {0: "no symmetry", 1: "symmetry"}

    # Another dictionary useful for labeling plots and files.
    shift_label_dict = {str([0,0,0]): "no shift", str([.5, .4, .3]): "shift"}

    # A list of lists that contains all the energies for every run.
    # The first nested loop contains the different energies (Ewald, Hartree, ...)
    # The second nested list four elements for each combination of shift/no shift
    # and symmmetry reduction/no symmetry reduction.
    vasp_energies = [[] for _ in range(12)]
    cutoff_energies = [[] for _ in range(12)]
    vasp_energy_names  = ['alpha Z', 'Ewald energy', '-1/2 Hartree', '-exchange',
                          '-V(xc)+E(xc)', 'PAW double counting', 'entropy T*S',
                          'eigenvalues', 'atomic energy', 'free energy',
                          'energy without entropy', 'energy(sigma->0)']

    # A dictionary used for labels in plots.
    vasp_energy_names_dict = {'alpha Z': "alpha",
                              'Ewald energy': "Ewald",
                              '-1/2 Hartree': "Hartree",
                              '-exchange': "Hartree exchange",
                              '-V(xc)+E(xc)': "total exchange",
                              'PAW double counting': "PAW energy",
                              'entropy T*S': "entropy",
                              'eigenvalues': "band energy",
                              'atomic energy': "atomic energy",
                              'free energy': "free energy",
                              'energy without entropy': "energy without entropy",
                              'energy(sigma->0)': "extrapolated free energy"}

    shift_sym_labels = []

    # Collect data from symmetry and atom offset runs.
    for symmetry in symmetry_list:
        print("symmetry: ", symmetry)
        sym_dir = os.path.join(encut_test_dir, str(symmetry))
        
        for shift in shift_list:
            print("shift: ", shift)
            shift_dir = os.path.join(sym_dir, str(shift))
            slabel = sym_label_dict[symmetry] + ", " + shift_label_dict[str(shift)]
            shift_sym_labels.append(slabel)
            
            for i in range(len(vasp_energy_names)):
                vasp_energies[i].append([])
                cutoff_energies[i].append([])

            for encut in energy_cutoff_list:
                print("encut: ", encut)
                finished = False
                encut_dir = os.path.join(shift_dir, str(encut))
                vasp_data = read_vasp(encut_dir)

                if vasp_data != None:
                    for i, en in enumerate(vasp_energy_names):
                        cutoff_energies[i][-1].append(float(encut))
                        vasp_energies[i][-1].append(float(vasp_data[en]))
                else:
                    continue
            print(cutoff_energies)

    # Make a directory to store plots
    plot_dir = os.path.join(encut_test_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    for e in cutoff_energies:
        print(e)
    xlims = [min(min(min(cutoff_energies))) - 10,  max(max(max(cutoff_energies))) + 10]
    print(xlims)
    
    # Plot the individual energy convergences with/without symmetry and atomic shift.
    for i, sym_energy in enumerate(vasp_energies):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ylims = [0,0]
        
        for j, energy in enumerate(sym_energy):
            ax1.scatter(cutoff_energies[i][j], energy, label=shift_sym_labels[j],
                        marker = next(marker))
            
            ind = np.argsort(cutoff_energies)
            error = abs(np.array(energy) - energy[-1])
            ylims = [min(error)/2, max(error)*2]            
            ax2.scatter(cutoff_energies[i][j][:-1], error[:-1], label=shift_sym_labels[j],
                        marker = next(marker))
        print(ylims)
        if np.isclose(ylims[0], 0):
            ylims[0] = 1e-10

        if np.isclose(ylims[1], 0):
            ylims[1] = 1
        
        # Plot the energy convergences.
        plot_label_i = vasp_energy_names_dict[vasp_energy_names[i]]
        ax1.set_title("Comparing " + plot_label_i + " without symmetry "
                      "and with shifted atoms")
        ax1.set_xlabel("Energy cutoff (eV)")
        ax1.set_ylabel("Energy (eV/Ang)")
        ax1.set_xticks(ax1.get_xticks()[::2])
        ax1.legend()
        plot_name = os.path.join(plot_dir, plot_label_i + ".png")
        fig1.savefig(plot_name, bbox_inches="tight")
        plt.close(fig1)

        # Plot the error relative to the highest energy cutoff.
        plot_label_i = vasp_energy_names_dict[vasp_energy_names[i]] + " error"
        ax2.set_title("Comparing " + plot_label_i + " without symmetry "
                      "and with shifted atoms")
        ax2.set_xlabel("Energy cutoff (eV)")
        ax2.set_ylabel("Energy error (eV)")
        ax2.set_xticks(ax2.get_xticks()[::2])
        ax2.set_yscale("log")
        ax2.set_ylim(ylims[0], ylims[1])
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.legend()
        plot_name = os.path.join(plot_dir, plot_label_i + ".png")
        fig2.savefig(plot_name, bbox_inches="tight")
        plt.close(fig2)

    # Plot the total charge wrap-around vs. energy cutoff.
    fig, ax = plt.subplots()

    for i,direct in enumerate(directions):
        ax.scatter(cutoff_plot_list, tot_wrapped_charge_list[i],
                   label="%s-direction"%directions[i], marker = next(marker))

    ax.set_title("Number of iterations required for 1E-10 accuracy")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_ylabel("Total charge density wrap-around")
    ax.legend()
    plot_name = os.path.join(plot_dir, "total_charge_wraparound.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

    # Plot drift forces.
    fig, ax = plt.subplots()
    ax.scatter(cutoff_plot_list, drift_force_list)
    ax.set_title("Change in drift force")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_ylabel("Drift force (eV/Ang)")
    ax.set_xticks(ax.get_xticks()[::2])
    plot_name = os.path.join(plot_dir, "drift_force.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

    # Plot cell volumes.
    fig, ax = plt.subplots()
    ax.scatter(cutoff_plot_list, cell_volume_list)
    ax.set_title("Change in cell volume")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_ylabel("Cell volume ($\mathrm{Ang}^3$)")
    ax.set_xticks(ax.get_xticks()[::2])
    plot_name = os.path.join(plot_dir, "cell_volume.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

    # Plot cell norms.
    fig, ax = plt.subplots()
    ax.scatter(cutoff_plot_list, lat_vec_norm_list)
    ax.set_title("Change in basis vectors")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_ylabel("Frobenius norm of basis")
    ax.set_xticks(ax.get_xticks()[::2])
    plot_name = os.path.join(plot_dir, "basis_norm.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

    # Plot all forces.
    fig, ax = plt.subplots()
    for i,force_type in enumerate(force_type_list):
        ax.scatter(cutoff_plot_list, force_list[i], label=force_type,
                   marker = next(marker))

    # ax.scatter(cutoff_plot_list, drift_force_list, label="drift")
    ax.set_title("Forces acting on ions")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_ylabel("Force (eV/Ang)")
    ax.set_xticks(ax.get_xticks()[::2])
    ax.legend()
    plot_name = os.path.join(plot_dir, "all_forces.png")
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)
    
    return None

def setup_enaug_tests(location, system_name, enaug_list):
    """Create the file structure and submit jobs that will compare various
    values of the energy cutoff for the augmentation chargens to find one that 
    is adequate.

    Args:
        location (str): the root directory. It must have a folder named
            system_name and this folder must contain a folder named initial_data,
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        enaug_list (list): a list of energy cutoffs that are fractions of the 
            default energy cutoff.
    """

    # The location is the file path of the element directory.
    system_dir = os.path.join(location, system_name)
    data_dir = os.path.join(system_dir, "initial_data")

    # Get the energy cutoff and number of bands from the POTCAR.
    # These are relevant for building the INCAR.
    vasp_data = read_vasp_input(data_dir)
    eaug = np.max(vasp_data["EAUG list"])
    new_eaug = 2*eaug

    # Make a directory where testing the augmentation energy cutoff will be performed.
    enaug_test_dir = os.path.join(system_dir, "enaug")
    if not os.path.isdir(enaug_test_dir):
        os.mkdir(enaug_test_dir)
    os.chdir(enaug_test_dir)

    enaug_list = [i*eaug for i in enaug_list]

    for enaugcut in enaug_list:
        enaugcut_dir = os.path.join(enaug_test_dir, str(enaugcut))
        if not os.path.isdir(enaugcut_dir):
            os.mkdir(enaugcut_dir)
        
        os.chdir(enaugcut_dir)

        subprocess.call("cp " + data_dir + "/* .", shell=True)
        subprocess.call("cp " + location + "/run.py .", shell=True)
        subprocess.call("cp " + location + "/run.sh .", shell=True)

        create_INCAR(enaugcut_dir)
        incar_dir = os.path.join(enaugcut_dir, "INCAR")
        run_dir = os.path.join(enaugcut_dir, "run.sh")

        # Correctly label this job and adjust runtime if necessary.
        with open(run_dir, 'r') as file:
            filedata = file.read()

        job_name = "enaug " + str(enaugcut)
        filedata = filedata.replace("JOB NAME", job_name)
        filedata = filedata.replace("12:00:00", "4:00:00")
        filedata = filedata.replace("4096", "8192")

        with open(run_dir, 'w') as file:
            file.write(filedata)

        # Replace the number of bands and change the initial charge density.
        # The manual says that 1E-6 accuracy should be obtained within 10-15 iterations.
        with open(incar_dir, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("ENAUG = %f"%new_eaug, "ENAUG = %f"%enaugcut)
        filedata = filedata.replace("PREC = Accurate", "PREC = Normal")

        with open(incar_dir, "w") as file:
            file.write(filedata)

        # Setting a larger stack size should keep some jobs from segfaulting.
        subprocess.call("ulimit -s unlimited", shell=True)
            
        # Submit the job.
        subprocess.call('sbatch run.sh', shell=True)


def gen_enaug_plots(system_dir):
    """Generate plots that help identify the appropriate value for the energy cutoff
    for the augmentation charges.

    Args:
        system_name (str): the file path to the system being simulated. This folder
            must contain subfolders that contain VASP simulations with varying energy
            cutoffs.
    """


    system_name = system_dir.split(os.sep)[-1]    
    enaug_test_dir = os.path.join(system_dir, "enaug")
    
    # Initialize all quantities that'll be plotted.
    enaug_list = []
    wrapped_charge_list = [[],[],[]]
    directions = ["x", "y", "z"]

    enaug_cutoff_list = os.listdir(enaug_test_dir)
    try:
        enaug_cutoff_list.remove("plots")
    except:
        None

    for enaug_cutoff in enaug_cutoff_list:
        enaug_dir = os.path.join(enaug_test_dir, str(enaug_cutoff))
        outcar_dir = os.path.join(enaug_dir, "OUTCAR")

        vasp_data = read_vasp(enaug_dir)

        if vasp_data != None:
            enaug_list.append(float(enaug_cutoff))
            for j in range(len(directions)):
                wrapped_charge_list[j].append(vasp_data["total wrapped soft charge"][j])    

    plot_dir = os.path.join(enaug_test_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # Plot the soft charge wrap-around vs. energy cutoff.
    fig, ax = plt.subplots()

    for i,direct in enumerate(directions):
        ax.scatter(enaug_list, wrapped_charge_list[i],
                   label="%s-direction"%directions[i],
                   marker = next(marker))

    ax.set_title("Number of iterations required for 1E-10 accuracy")
    ax.set_xlabel("Energy cutoff (eV)")
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_ylabel("Soft charge density wrap-around")
    ax.legend()
    plot_name = os.path.join(plot_dir, "soft_charge_wraparound.png")
    fig.savefig(plot_name)
    plt.close(fig)


def setup_ndos_tests(location, system_name, ndos_list):
    """Create the file tree and submit jobs on the supercomputer for
    testing the sampling of the density of states (NEDOS) in the INCAR.

    Args:
        location (str): the root directory. It must have a folder named
            'system_name' and this folder must contain a folder named 'initial_data',
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        ndos_list (list): a list of the number of density of states sampling points,
            in fractions of the default value (NEDOS = 301).
    """

    ## Everything below will be creating runs for testing the sampling
    ## for the density of states.

    # The location is the file path of the element directory.
    system_dir = os.path.join(location, system_name)
    data_location = os.path.join(system_dir, "initial_data")

    # Make a directory where testing the DOS sampling will be performed.
    ndos_test_dir = os.path.join(system_dir, "ndos")
    if not os.path.isdir(ndos_test_dir):
        os.mkdir(ndos_test_dir)
    os.chdir(ndos_test_dir)

    default_ndos = 301
    ndos_list = [int(default_ndos*i) for i in ndos_list]

    for ndos in ndos_list:
        ndos_dir = os.path.join(ndos_test_dir, str(ndos))
        if not os.path.isdir(ndos_dir):
            os.mkdir(ndos_dir)
        os.chdir(ndos_dir)
        
        subprocess.call("cp " + data_location + "/* .", shell=True)
        subprocess.call("cp " + location + "/run.py .", shell=True)
        subprocess.call("cp " + location + "/run.sh .", shell=True)
            
        create_INCAR(ndos_dir)
        incar_dir = os.path.join(ndos_dir, "INCAR")
        run_dir = os.path.join(ndos_dir, "run.sh")

        # Correctly label this job and adjust runtime if necessary.
        with open(run_dir, 'r') as file:
            filedata = file.read()

        job_name = "ndos " + str(ndos)
        filedata = filedata.replace("JOB NAME", job_name)
        filedata = filedata.replace("12:00:00", "4:00:00")
        filedata = filedata.replace("4096", "8192")

        with open(run_dir, 'w') as file:
            file.write(filedata)

        # Replace the number of bands and change the initial charge density.
        # The manual says that 1E-6 accuracy should be obtained within 10-15 iterations.
        with open(incar_dir, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("NEDOS = 2000", "NEDOS = %i"%ndos)
        
        with open(incar_dir, "w") as file:
            file.write(filedata)


        # Setting a larger stack size should keep some jobs from segfaulting.
        subprocess.call("ulimit -s unlimited", shell=True)
            
        # Submit the job.
        subprocess.call('sbatch run.sh', shell=True)


def gen_ndos_plots(system_dir):
    """Generate plots that help identify the appropriate value for the sampling
    of the density of states.

    Args:
        system_dir (str): the file path to the system being simulated. This folder
            must contain subfolders that contain VASP simulations with varying density
            of states samplings.
    """

    system_name = system_dir.split(os.sep)[-1]    
    ndos_test_dir = os.path.join(system_dir, "ndos")
    
    # Initialize all quantities that'll be plotted.
    cutoff_energies = []
    vasp_energies = [[] for _ in range(12)]
    ndos_values = [[] for _ in range(12)]
    VASP_data = {}

    vasp_energy_names  = ['alpha Z', 'Ewald energy', '-1/2 Hartree', '-exchange',
                          '-V(xc)+E(xc)', 'PAW double counting', 'entropy T*S',
                          'eigenvalues', 'atomic energy', 'free energy',
                          'energy without entropy', 'energy(sigma->0)']

    # A dictionary used for labels in plots.
    vasp_energy_names_dict = {'alpha Z': "alpha",
                              'Ewald energy': "Ewald",
                              '-1/2 Hartree': "Hartree",
                              '-exchange': "Hartree exchange",
                              '-V(xc)+E(xc)': "total exchange",
                              'PAW double counting': "PAW energy",
                              'entropy T*S': "entropy",
                              'eigenvalues': "band energy",
                              'atomic energy': "atomic energy",
                              'free energy': "free energy",
                              'energy without entropy': "energy without entropy",
                              'energy(sigma->0)': "extrapolated free energy"}
    
    ndos_list = os.listdir(ndos_test_dir)
    try:
        ndos_list.remove("plots")
    except:
        None

    for ndos in ndos_list:
        ndos_dir = os.path.join(ndos_test_dir, ndos)
        vasp_data = read_vasp(ndos_dir)

        if vasp_data != None:
            for i, en in enumerate(vasp_energy_names):
                ndos_values[i].append(int(ndos))
                vasp_energies[i].append(float(vasp_data[en]))
        else:
            continue

    # Make a directory to store plots.
    plot_dir = os.path.join(ndos_test_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # Plot energy convergence against density of state sampling.
    fig1, ax1 = plt.subplots()
    for ndos_list, vasp_energy, energy_name in zip(ndos_values, vasp_energies, vasp_energy_names):
        fig2,ax2 = plt.subplots()
        ax1.scatter(ndos_list, vasp_energy, label=vasp_energy_names_dict[energy_name],
                    marker = next(marker))
        ax2.scatter(ndos_list, vasp_energy, label=vasp_energy_names_dict[energy_name])
        ax2.set_title("Energy convergence with greater DOS accuracy")
        ax2.set_xlabel("Number of grid points on which the DOS is evaluated")
        ax2.set_ylabel("Energy (eV/Ang)")
        ax2.set_xticks(ax2.get_xticks()[::2])
        ax2.legend()
        plot_name = os.path.join(plot_dir, vasp_energy_names_dict[energy_name] + "_dos.png")
        fig2.savefig(plot_name, bbox_inches="tight")
        plt.close(fig2)
        
    ax1.set_title("Energy convergence with greater DOS accuracy")
    ax1.set_xlabel("Number of grid points on which the DOS is evaluated")
    ax1.set_ylabel("Energy (eV/Ang)")
    ax1.set_xticks(ax1.get_xticks()[::2])
    lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plot_name = os.path.join(plot_dir, "dos_convergence.png")
    fig1.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig1)

    # List the density of states sampling directories again.
    ndos_list = os.listdir(ndos_test_dir)
    try:
        ndos_list.remove("plots")
    except:
        None

    # Make a directory to save density of state and integrated density of state plots.
    dos_plot_dir = os.path.join(plot_dir, "DOS")
    if not os.path.isdir(dos_plot_dir):
        os.mkdir(dos_plot_dir)

    # Collect and plot density of states data.
    for ndos in ndos_list:
        ndos_dir = os.path.join(ndos_test_dir, ndos)
        fermi_level = 0
        vasp_data = read_vasp(ndos_dir)

        if vasp_data != None:
            energy_list = vasp_data["density of states energies"]
            dos_list = vasp_data["density of states data"]
            idos_list = vasp_data["integrated density of states data"]

            fermi_level = vasp_data["Fermi level"]
        else:
            continue

        plot_dir = os.path.join(dos_plot_dir, ndos)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
            
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        ax1.plot(energy_list, dos_list)
        ax1.set_title("Density of states with %s energy bins"%ndos)
        ax1.set_xlim(min(energy_list), fermi_level + 2)
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Density of states")

        plot_name = os.path.join(plot_dir, "dos.png")
        fig1.savefig(plot_name, bbox_inches="tight")
        plt.close(fig1)
        
        ax2.plot(energy_list, idos_list)
        ax2.set_title("Integrated density of states with %s energy bins"%ndos)
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Integrated density of states")
        ax2.set_xlim(min(energy_list), fermi_level + 1)
        
        plot_name = os.path.join(plot_dir, "idos.png")
        fig2.savefig(plot_name, bbox_inches="tight")
        plt.close(fig2)

    return None


def pickle_QE_data(home_dir, system_name, parameters):
    """Write a pickle file of the data from the Quantum Espresso runs of a system.

    Args:
        home_dir (str): the root directory. It must have a folder named
            'system_name' and this folder must contain a folder named 'initial_data',
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        parameters (dict): a dictionary of adjustable parameters. The following
            key strings must be present: 'grid type list', 'offset list',
            'occupation list', 'smearing list', 'smearing value list',
            and 'k-points list'. Their corresponding values must be lists.
    """
    
    data = np.empty([len(parameters[i]) for i in parameters.keys()], dtype=float)

    # Make and move into the system directory.
    system_dir = home_dir + "/" + system_name

    # Find the grid type directory.
    for i, grid_type in enumerate(parameters["grid type list"]):
        grid_dir = system_dir + "/" + grid_type

        # Find the offset directory.
        for j, shift in enumerate(parameters["offset list"]):
            offset_name = str(shift).strip("[").strip("]")
            offset_name = offset_name.replace(",", "")
            offset_dir = grid_dir + "/" + offset_name

            # Find the occupation type directory.
            for k,occupation in enumerate(parameters["occupation list"]):
                occupation_dir = offset_dir + "/" + occupation

                # Find the smearing type directory.
                for l,smearing in enumerate(parameters["smearing list"]):
                    smearing_dir = occupation_dir + "/" + smearing

                    # Find the smearing value directory.
                    for m,smearing_value in enumerate(parameters["smearing value list"]):
                        smearing_value_dir = smearing_dir + "/" + str(smearing_value)

                        # Find the k-points directory.
                        kpoint_list = os.listdir(smearing_value_dir)

                        # Find the number of reduced and unreduced k-points.
                        nkpoints_list = []
                        for n,nkpoints in enumerate(kpoint_list):
                        # for n,kpoints in enumerate(parameters["k-point list"]):
                            # nkpoints = np.prod(kpoints)
                            kpoint_dir = smearing_value_dir + "/" + str(nkpoints)
                            qe_data = read_QE(kpoint_dir, system_name)

                            nkpoints_list.append(str([int(qe_data["number of unreduced k-points"]), (
                                int(qe_data["number of reduced k-points"]))]))
                            for o,energy in enumerate(parameters["energy list"]):
                                data[i,j,k,l,m,n,o] = float(qe_data[energy].split()[0])    
                                
                                
    # A list of quantities to include in the data frame.
    energy_list = ["total energy", "Fermi energy",
                     "one-electron contribution", "ewald contribution",
                     "xc contribution", "hartree contribution"]    

    offset_names = [str(o) for o in parameters["offset list"]]
    kpoint_names = [str(k).strip("[]").replace(",","") for k in nkpoints_list]
    coordinates = [parameters["grid type list"],
                   offset_names,
                   parameters["occupation list"],
                   parameters["smearing list"],
                   parameters["smearing value list"],
                   nkpoints_list,
                   energy_list]

    dimensions = ("grid", "offset", "occupation", "smearing", "smearing_value",
                  "kpoints", "energy")
    
    coordinates = {i:j for i,j in zip(dimensions, coordinates)}
    data = xr.DataArray(data, coords=coordinates, dims=dimensions)
    data.name = system_name + " Quantum Espresso Data"
    
    pickle.dump(data, open(system_name + ".p", "wb"))

def read_QE_pseudopot(pseudo_file):
    """Read relevant parameters from the pseudopotential file.

    Args:
        location (str): the file path to the pseudopotential.

    Returns:
        parameters (dict): a dictionary of pseudopotential parameters.
    """
    
    parameters = {}
    
    with open(pseudo_file, 'r') as file :
        for i,line in enumerate(file.readlines()):
            if "Z valence" in line:
                parameters["valency"] = int(line.split()[0])
                
            if "z_valence" in line:
                parameters["valency"] = int(float(line.split()[0].strip("z_valence=").replace('"', '')))

            if "wfc_cutoff" in line:
                parameters["wavefunction cutoff"] = float(
                    line.split()[0].strip("wfc_cutoff=").replace('"', ''))

            if "rho_cutoff" in line:
                parameters["charge density cutoff"] = float(
                    line.split()[0].strip("rho_cutoff=").replace('"', ''))

            if "Suggested cutoff for wfc and rho" in line:
                parameters["wavefunction cutoff"] = float(line.split()[0])
                parameters["charge density cutoff"] = float(line.split()[0])                

    return parameters

def test_QE_inputs(location, system_name, system_parameters,
                   ecutwfc_list, ecutrho_list, nbands_list):
    """Create the file structure and submit jobs that will compare various
    values of the wavefunction cutoff in order to find one that is adequate.

    Args:
        location (str): the root directory. It must have a folder named
            'system_name' and this folder must contain a folder named 'initial_data',
            where the VASP input files (POTCAR, POSCAR, and KPOINTS) are located.
        system_name (str): the name of the system being simulated.
        system_parameters (dict): a dictionary that contains the geometric properties of
            the system, including the atomic types and positions, as well as the lattice
            basis.
        ecutwfc_list (list): a list of wavefunction cutoffs that are in fractions of the
            default cutoff.
        ecutrho_list (list): a list of charge density cutoffs that are in fractions of the
            default cutoff.
        nbands_list (list): a list of the number of bands included in the calculation in
            multiples of the default cutoff.
    """
    
    system_dir = os.path.join(location, system_name)
    data_location = os.path.join(system_dir, "initial_data")

    pseudo_files = []
    for file in os.listdir(data_location):
        if ".UPF" in file:
            pseudo_files.append(os.path.join(data_location, file))
    
    # Get the energy cutoffs and number of bands from the pseudopotential.
    # These are relevant for building the QE input file.
    QE_data = [read_QE_pseudopot(pf) for pf in pseudo_files]

    wfc_cutoff = max([qe["wavefunction cutoff"] for qe in QE_data])
    rho_cutoff = max([qe["charge density cutoff"] for qe in QE_data])
    nbands = np.sum([qe["valency"] for qe in QE_data])

    if np.isclose(wfc_cutoff, 0):
        wfc_cutoff = 100

    if np.isclose(rho_cutoff, 0):
        rho_cutoff = 4*wfc_cutoff

    # Make a directory where testing the wavefunction cutoff will be performed.
    wfc_dir = os.path.join(system_dir, "ecutwfc")
    if not os.path.isdir(wfc_dir):
        os.mkdir(wfc_dir)
    os.chdir(wfc_dir)

    ecutwfc_list = [i*wfc_cutoff for i in ecutwfc_list]

    # One of the tests will compare the total energy of two QE calculations
    # with the atoms of one run shifted with respect to the other.
    shift_list = [[0, 0, 0], [.2, .4, .3]]
    symmetry_list = [0, 1]

    for symmetry in symmetry_list:
        sym_dir = os.path.join(encut_test_dir, str(symmetry))
        if not os.path.isdir(sym_dir):
            os.mkdir(sym_dir)
        os.chdir(sym_dir)
        
        for shift in shift_list:
            shift_dir = os.path.join(sym_dir, str(shift))
            if not os.path.isdir(shift_dir):
                os.mkdir(shift_dir)
            os.chdir(shift_dir)
            
            for ecutwfc in ecutwfc_list:
                ecutwfc_dir = os.path.join(shift_dir, str(ecutwfc))
                if not os.path.isdir(ecutwfc_dir):
                    os.mkdir(ecutwfc_dir)
                os.chdir(ecutwfc_dir)

                # Copy the required VASP input files into the energy cutoff directory.
                # Also copy the batch job and python script.
                subprocess.call("cp " + data_location + "/* .", shell=True)
                subprocess.call("cp " + location + "/run.py .", shell=True)
                subprocess.call("cp " + location + "/run.sh .", shell=True)
                    
                # create_input(ecutwfc_dir)
                input_dir = os.path.join(ecutwfc_dir, system_name + ".in")
                run_dir = os.path.join(ecutwfc_dir, "run.sh")

                # Correctly label this job, and adjust runtime and memory if necessary.
                with open(run_dir, 'r') as file:
                    filedata = file.read()
                    
                filedata = filedata.replace("ELEMENT", system_name)
                filedata = filedata.replace("GRIDTYPE", str(shift[0]))
                filedata = filedata.replace("SMEAR KPOINT", "ecutwfc " + str(ecutwfc))
                filedata = filedata.replace("12:00:00", "4:00:00")
                filedata = filedata.replace("4096", "8192")

                with open(run_dir, 'w') as file:
                    file.write(filedata)
                    
                # Change the energy cutoff, and decrease the precision since it affects
                # the energy cutoff, and it isn't clear whether the energy cutoff prescribed
                # by PREC will overwrite that of ENCUT. Also turn off or leave on symmetry.
                with open(input_dir, "r") as file:
                    filedata = file.read()

                if symmetry == 0:
                    filedata = filedata.replace("nosym = .false.", "nosym = .true.")

                filedata = filedata.replace("20 20 20 1 1 1", "20 20 20 " +
                                            str(shift[0]) + " " +
                                            str(shift[1]) + " " +
                                            str(shift[2]))

                filedata = filedata.replace("ecutwfc = " + wfc_cutoff, "ecutwfc = " + str(ecutwfc))

                with open(input_dir, "w") as file:
                    file.write(filedata)

                # Setting a larger stack size should keep some jobs from segfaulting.
                subprocess.call("ulimit -s unlimited", shell=True)

                # Submit the job.
                subprocess.call('sbatch run.sh', shell=True)

    # Make a directory where testing the charge density cutoff will be performed.
    rho_dir = os.path.join(system_dir, "ecutrho")
    if not os.path.isdir(rho_dir):
        os.mkdir(rho_dir)
    os.chdir(rho_dir)

    ecutrho_list = [i*rho_cutoff for i in ecutrho_list]

    ## Charge density cutoff tests
    for ecutrho in ecutrho_list:
        ecutrho_dir = os.path.join(shift_dir, str(ecutrho))
        if not os.path.isdir(ecutrho_dir):
            os.mkdir(ecutrho_dir)
        os.chdir(ecutrho_dir)

        # Copy the required VASP input files into the energy cutoff directory.
        # Also copy the batch job and python script.
        subprocess.call("cp " + data_location + "/* .", shell=True)
        subprocess.call("cp " + location + "/run.py .", shell=True)
        subprocess.call("cp " + location + "/run.sh .", shell=True)
            
        # create_input(ecutwfc_dir)
        input_dir = os.path.join(ecutrho_dir, system_name + ".in")
        run_dir = os.path.join(ecutrho_dir, "run.sh")

        # Correctly label this job, and adjust runtime and memory if necessary.
        with open(run_dir, 'r') as file:
            filedata = file.read()

        filedata = filedata.replace("ELEMENT", system_name)
        filedata = filedata.replace("GRIDTYPE", str(shift[0]))
        filedata = filedata.replace("SMEAR KPOINT", "ecutrho " + str(ecutrho))
        filedata = filedata.replace("12:00:00", "4:00:00")
        filedata = filedata.replace("4096", "8192")

        with open(run_dir, 'w') as file:
            file.write(filedata)

        # Change the energy cutoff, and decrease the precision since it affects
        # the energy cutoff, and it isn't clear whether the energy cutoff prescribed
        # by PREC will overwrite that of ENCUT. Also turn off or leave on symmetry.
        with open(input_dir, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("ecutrho = " + rho_cutoff, "ecutrho = " + str(ecutrho))

        with open(input_dir, "w") as file:
            file.write(filedata)

        # Setting a larger stack size should keep some jobs from segfaulting.
        subprocess.call("ulimit -s unlimited", shell=True)

        # Submit the job.
        subprocess.call('sbatch run.sh', shell=True)


    # Make a directory where testing the number of bands will be performed.
    nbands_dir = os.path.join(system_dir, "nbnd")
    if not os.path.isdir(nbands_dir):
        os.mkdir(nbands_dir)
    os.chdir(nbands_dir)
                
    nbands_list = [i*nbands for i in range(20)]

    ## Number of bands tests 
    for nbands in nbands_list:
        nbands_dir = os.path.join(shift_dir, str(nbands))
        if not os.path.isdir(nbands_dir):
            os.mkdir(nbands_dir)
        os.chdir(nbands_dir)

        # Copy the required VASP input files into the energy cutoff directory.
        # Also copy the batch job and python script.
        subprocess.call("cp " + data_location + "/* .", shell=True)
        subprocess.call("cp " + location + "/run.py .", shell=True)
        subprocess.call("cp " + location + "/run.sh .", shell=True)
            
        # create_input(ecutwfc_dir)
        input_dir = os.path.join(nbands_dir, system_name + ".in")
        run_dir = os.path.join(nbands_dir, "run.sh")

        # Correctly label this job, and adjust runtime and memory if necessary.
        with open(run_dir, 'r') as file:
            filedata = file.read()

        filedata = filedata.replace("ELEMENT", system_name)
        filedata = filedata.replace("GRIDTYPE", str(shift[0]))
        filedata = filedata.replace("SMEAR KPOINT", "nbands " + str(nbands))
        filedata = filedata.replace("12:00:00", "4:00:00")
        filedata = filedata.replace("4096", "8192")

        with open(run_dir, 'w') as file:
            file.write(filedata)

        # Change the energy cutoff, and decrease the precision since it affects
        # the energy cutoff, and it isn't clear whether the energy cutoff prescribed
        # by PREC will overwrite that of ENCUT. Also turn off or leave on symmetry.
        with open(input_dir, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("nbnd = 20", "nbnd = " + nbands)

        with open(input_dir, "w") as file:
            file.write(filedata)

        # Setting a larger stack size should keep some jobs from segfaulting.
        subprocess.call("ulimit -s unlimited", shell=True)

        # Submit the job.
        subprocess.call('sbatch run.sh', shell=True)
