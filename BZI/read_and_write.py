import numpy as np

def read_QE(output_file):
    QE_data = {}
    QE_data["self-consistent calculation time"] = []    
                                       
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
                QE_data["number of k-points"] = float(line.split()[4])
                
                kpt = True
                index = i + 2
                k_line = f[index].split()
                kpt_list = []
                kpt_weights = []
                while kpt:
                    kpt_list.append([float(k_line[4]), float(k_line[5]), float(k_line[6].strip("),"))])
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
                k_line = f[index].split()
                kpt = True
                kpt_energies = []
                kpt_plane_waves = []
                
                while kpt:
                    if k_line == []:
                        pass
                    else:
                        
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
