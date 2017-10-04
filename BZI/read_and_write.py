import os
import numpy as np
import pandas as pd
import pickle

def read_QE(location, file_prefix):
    """
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
    panel_file = open(home + "/panel.p", "w")
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
