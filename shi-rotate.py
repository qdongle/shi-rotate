#!/usr/bin/env python3
import numpy as np
import argparse
import sys
from functools import reduce
from scipy.optimize import minimize
# from scipy.linalg import expm
import os
import subprocess
import glob
import time
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import config


start_time = time.time()

parser = argparse.ArgumentParser(description="SHI-rotate: A tool for SOMO-HOMO inversion.")
# Add required arguments
parser.add_argument("file47",default=None,help='.gen (FILE47) from NWChem or Gaussian output')

# Add optional arguments
parser.add_argument('--overlap-threshold',"-t",default=0.00, type=float, help='Overlap threshold between alpha and beta')
parser.add_argument('--clean',"-c",default=False, help='Remove all cube files, rotated.movecs.')

# TO-DOs or improve. These options help user to generate cube files
parser.add_argument('--movecs-nwchem',"-movecs",dest='movecs',default=None, help='NWChem movecs file: to restart NWChem to generate cube files.')
parser.add_argument('--fch-gaussian',"-fch",dest='fch',default=None, help='Gaussian fch file: to generate cube files.')

args = parser.parse_args()


# JAX setup
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# Create a log file and print the header
shirotate_log= open('shi_rotate.log', 'w')
shirotate_log.write(r'''
  ____  ____    _______
 6MMMMb\`MM'    `MM`MM'
6M'    ` MM      MM MM                          /               /
MM       MM      MM MM       ___  __   _____   /M        ___   /M      ____
YM.      MM      MM MM       `MM 6MM  6MMMMMb /MMMMM   6MMMMb /MMMMM  6MMMMb
 YMMMMb  MMMMMMMMMM MM        MM69 " 6M'   `Mb MM     8M'  `Mb MM    6M'  `Mb
     `Mb MM      MM MM        MM'    MM     MM MM         ,oMM MM    MM    MM
      MM MM      MM MM        MM     MM     MM MM     ,6MM9'MM MM    MMMMMMMM
      MM MM      MM MM        MM     MM     MM MM     MM'   MM MM    MM
L    ,M9 MM      MM MM        MM     YM.   ,M9 YM.  , MM.  ,MM YM.  ,YM    d9
MYMMMM9 _MM_    _MM_MM_      _MM_     YMMMMM9   YMMM9 `YMMM9'Yb.YMMM9 YMMMM9
-----------------------------------------------------------------------------''')




# Process input files---------------------------------------------------------
current_directory = os.getcwd()
file47=args.file47
if args.file47==None:
    file47s=glob.glob(current_directory+'/*.gen')
    if len(file47s)>1:
        print("Please specify which file47 file will be used:")
        for gen in file47s:
            print(f'''     {gen.split("/")[-1]}''')
        sys.exit()
    elif len(file47s)==0:
        print("There is no file47 in the current directory.")
        sys.exit()
    else:
        file47=file47s[0]
shirotate_log.write(f"\nFILE47   : {file47.split('/')[-1]}\n")
shirotate_log.write(f"Work directory: {current_directory} \n")
#-----------------------------------------------------------------------------


toEV=27.2113245702

def custom_format(num):
    if num >= 0:
        return format(num, "17.15E")  # 17 characters for positive mantissa
    else:
        return format(num, "18.15E")  # 18 characters for negative mantissa

def formatted_overlap(number):
    if abs(number)<0.01:
        return "       "
    else:
        return f"{number: >+7.4f}".replace("+", " ")

def print_overlap(smat,a_homo_idx,numb):
    header="      |"
    for i in range(-numb,0):
        header+=f"  b_{a_homo_idx+i+1:04d}"
    header+="\n"
    shirotate_log.write(header)
    shirotate_log.write("---------------------------------------------------------------\n")
    for a in range(-numb,0):
        row=f"a_{a+a_homo_idx+1:04d}|"
        for i in range(-numb,0):
            row+=f" {formatted_overlap(smat[a][i])}"
        shirotate_log.write(row+"\n")

def write_ascii_array(file,array):
    for i in range(len(array)):
        if (i+1)%3!=0:
            file.write("   "+custom_format(array[i]))
            if (i+1)==len(array):
                file.write('\n')
        else:
            file.write("   "+custom_format(array[i]))
            file.write("\n")

def write_ascii_movec(ascii_movecs_name,num_elec_alpha,num_elec_beta,alpha_energies,beta_energies,c_a,c_b):
    new_ascii_path=current_directory+f"/{ascii_movecs_name}.ascii"
    nbas=c_a.shape[0]
    with open(current_directory+"/canonical.ascii","r") as f:
        with open(new_ascii_path,"w") as g:
            for i in range(14):
                line=f.readline()
                g.writelines(line)
            # Write alpha occupation
            occ=np.concatenate((np.ones(num_elec_alpha),np.zeros(nbas-num_elec_alpha)))
            write_ascii_array(g,occ)
            # Write alpha energy
            write_ascii_array(g,alpha_energies)
            # Write alpha coeff
            for i in range(nbas):
                write_ascii_array(g,c_a[i])
            # Write beta occupation
            occ=np.concatenate((np.ones(num_elec_beta),np.zeros(nbas-num_elec_beta)))
            write_ascii_array(g,occ)
            # Write beta energy
            write_ascii_array(g,beta_energies)
            # Write beta coeff
            for i in range(nbas):
                write_ascii_array(g,c_b[i])
            g.writelines(f.readlines()[-1])

def excute_bash_command(command):
    bash_result=subprocess.run(command.split(),capture_output=True,text=True)
    if bash_result.returncode!=0:
        shirotate_log.write(f"Error executing command: {command}\n")
        shirotate_log.write(bash_result.stderr)
        sys.exit()
    else:
        shirotate_log.write(command)
        shirotate_log.write(bash_result.stdout)

def get_number(string):
    # Purpose: Sometimes, Fortran code prints a wrong number: e.g 9.1-300 = 9.1E-300
    if "E" in string:
        return float(string)
    else:
        if "+" in string:
            return float(string.split("+")[0])*10**(int(string.split("+")[1]))
        elif "-" in string:
            return float(string.split("-")[0])*10**(-int(string.split("-")[1]))

def get_enviroment_variables():
    bash_result=subprocess.run("which nwchem".split(),capture_output=True,text=True)
    if bash_result.stdout=="":
        print("$NWCHEM is not set. Please specify the folder of NWChem")
        sys.exit()
    nwchem_top = "/".join(bash_result.stdout.split("/")[:-3])
    mov2asc=nwchem_top+"/contrib/mov2asc/mov2asc"
    asc2mov=nwchem_top+"/contrib/mov2asc/asc2mov"
    nwchem=nwchem_top+"/bin/LINUX64/nwchem"
    bash_result= subprocess.run('which dplot.py',shell=True,capture_output=True,text=True)
    dplot=bash_result.stdout.split('\n')[0]
    shirotate_log.write("----------------------------\n")
    shirotate_log.write(f"NWChem: {nwchem_top}\n")
    shirotate_log.write(f"mov2asc: {mov2asc}\n")
    shirotate_log.write(f"asc2mov: {asc2mov}\n")
    shirotate_log.write(f"dplot: {dplot}")
    return nwchem_top,mov2asc,asc2mov,nwchem,dplot

def get_upper_triangular_matrix_FILE47(line,f,number_of_matrices,nbas):
    temp=[]
    line = f.readline() # Skip the header line
    alpha_matrix =[]
    while(("$END" not in line)):
        numbers=line.split()
        for number in numbers:
            temp.append(get_number(number))
        line=f.readline()
        if ("END" in line):
            break
    if number_of_matrices==2:
        temp_a=temp[:int(nbas*(nbas-1)/2+nbas)]
        temp_b=temp[int(nbas*(nbas-1)/2+nbas):]
        alpha_matrix=np.zeros((nbas,nbas), dtype=np.float64)
        beta_matrix=np.zeros((nbas,nbas), dtype=np.float64)
        k=-1
        for i in range(nbas):
            for j in range(i+1):
                k=k+1
                alpha_matrix[i][j]=temp_a[k]
                alpha_matrix[j][i]=temp_a[k]
        k=-1
        for i in range(nbas):
            for j in range(i+1):
                k=k+1
                beta_matrix[i][j]=temp_b[k]
                beta_matrix[j][i]=temp_b[k]
        return alpha_matrix, beta_matrix , line
    else:
        matrix=np.zeros((nbas,nbas), dtype=np.float64)
        k=-1
        for i in range(nbas):
            for j in range(i+1):
                k=k+1
                matrix[i][j]=temp[k]
                matrix[j][i]=temp[k]
        return matrix , line

def get_square_matrix_FILE47(line,f,number_of_matrices,nbas):
    temp=[]
    line = f.readline() # Skip the header line
    while(("$END" not in line)):
        numbers=line.split()
        for number in numbers:
            temp.append(get_number(number))
        line=f.readline()
    if number_of_matrices==2:
        alpha_matrix=np.array(temp[:nbas*nbas], dtype=np.float64).reshape(nbas,nbas)
        beta_matrix=np.array(temp[nbas*nbas:], dtype=np.float64).reshape(nbas,nbas)
        return alpha_matrix, beta_matrix , line
    else:
        matrix=np.array(temp, dtype=np.float64).reshape(nbas,nbas)
        return matrix , line


def get_data_from_gen(genfile):
    shirotate_log.write(f"Reading data from gen file: {genfile}")
    alpha_coef_matrix = np.array([], dtype=np.float64)
    beta_coef_matrix = np.array([], dtype=np.float64)
    overlap_matrix = np.array([], dtype=np.float64)
    alpha_density_matrix = np.array([], dtype=np.float64)
    beta_density_matrix = np.array([], dtype=np.float64)
    alpha_fock_matrix = np.array([], dtype=np.float64)
    beta_fock_matrix = np.array([], dtype=np.float64)
    with open(genfile) as f:
        line=f.readline()
        if "OPEN" in line:
            open_shell= True
        else:
            open_shell=False
            shirotate_log.write("Please check your molecular system (geometry, charge, ab initio input)")
            sys.exit("FILE47 contains the closed shell calculation. SHI-rotate needs open shell calculation.")

        # Get number of basis functions: NBAS
        # The first has a format like that: $GENNBO  UPPER  BODM  BOHR  NATOMS= 3  NBAS=  24 $END
        #  Next to the string "NBAS=" is the number of basis functions
        if "NBAS" in line:
            # If nbas<100, Fortran developer leaves a space after "NBAS= 99"
            # this is different with the case "NBAS=100"
            for token in line.split():
                if token.startswith("NBAS="):
                    nbas_str = token[len("NBAS="):]
                    # There is a space if nbas_str=="", because line was split by space
                    if nbas_str == "":
                        idx = line.split().index(token)
                        nbas = int(line.split()[idx + 1])
                    # This is the case when nbas>100
                    else:
                        nbas = int(nbas_str)
                    break
            else:
                sys.exit("Failed to parse NBAS from FILE47")
        else:
            sys.exit("NBAS is not found in the first line of FILE47. Please check the FILE47.")
        shirotate_log.write(f"Number of basis functions (NBAS): {nbas}\n")


        # Parse the rest of the file
        while line:
            if "$LCAOMO" in line:
                alpha_coef_matrix, beta_coef_matrix ,line = get_square_matrix_FILE47(line, f, 2, nbas)
            elif "$DENSITY" in line:
                alpha_density_matrix, beta_density_matrix ,line = get_upper_triangular_matrix_FILE47(line, f, 2, nbas)
            elif "$FOCK" in line:
                alpha_fock_matrix, beta_fock_matrix ,line = get_upper_triangular_matrix_FILE47(line, f, 2, nbas)
            elif "$OVERLAP" in line:
                    overlap_matrix ,line = get_upper_triangular_matrix_FILE47(line, f, 1, nbas)
            line = f.readline()

        # Calculate numbers of alpha, beta electrons by density matrix.
        alpha_elec=int(np.trace(alpha_density_matrix@overlap_matrix))
        shirotate_log.write("\n")
        shirotate_log.write("\n")
        shirotate_log.write(f"Alpha electrons: {alpha_elec}\n")
        beta_elec=int(np.trace(beta_density_matrix@overlap_matrix))
        shirotate_log.write(f"Beta electrons: {beta_elec}\n" )

        return alpha_coef_matrix, beta_coef_matrix, overlap_matrix, alpha_density_matrix, beta_density_matrix, alpha_fock_matrix, beta_fock_matrix, nbas, alpha_elec, beta_elec



def shi_rotate():
    # Get C, S_ao, DM, Fock matrices, nbas, number of alpha-beta electrons from file47
    # density matrix is not used, they are discarded.
    c_a,c_b,S_basis,_,_,F_alfa,F_beta,nbas,alpha_homo_idx,beta_homo_idx=get_data_from_gen(file47)
    beta_sumo_idx=beta_homo_idx+1
    shirotate_log.write("Alpha HOMO index is: "+str(alpha_homo_idx)+"    Beta SUMO index is: "+str(beta_sumo_idx)+"\n")
    # Get the MO coefs
    # alpha_coef and beta_coef are NOT full coefs matrix. It only include alpha occupied  (all occ)
    # and beta occupied + beta SUMO (all occ+1virt)
    alpha_coef=c_a[0:alpha_homo_idx,:]  # All occupied alpha MOs only
    alpha_coef=alpha_coef.T             # Transpose due to row-vector in file47
    beta_coef=c_b[0:beta_sumo_idx,:]    # All occupied beta MOs + SUMO
    beta_coef=beta_coef.T               # Transpose due to row-vector in file47
    initial_S_ab=reduce(np.dot, (alpha_coef.T, S_basis, beta_coef))


    # Alpha set contains orbitals that will be used in rotation
    alpha_set=[]
    beta_set=[]
    recur_step=0
    max_depth=10
    def check_overlap(a,b):
        if a=="all":
            ib=b-1
            for ia in range(beta_sumo_idx):
                if abs(initial_S_ab[ia][ib])>args.overlap_threshold:
                    if not(ia+1 in alpha_set):
                        alpha_set.append(ia+1)
        if b=="all":
            ia=a-1
            for ib in range(beta_sumo_idx):
                if abs(initial_S_ab[ia][ib])>args.overlap_threshold:
                    if not(ib+1 in beta_set):
                        beta_set.append(ib+1)

    def recursive_check(current_depth=0):
        if current_depth >= max_depth:
            shirotate_log.write("Maximum recursion depth reached!\n")
            return

        if(len(alpha_set)==0):
            check_overlap("all",beta_sumo_idx)
            for a in alpha_set:
                check_overlap(a,"all")
        if len(alpha_set) > len(beta_set):
            # Find more beta orbitals for existing alpha orbitals
            for a in alpha_set:
                check_overlap(a, "all")
        elif len(beta_set) > len(alpha_set):
            # Find more alpha orbitals for existing beta orbitals
            for b in beta_set:
                check_overlap("all", b)

        # Check if sets are equal in length
        if len(alpha_set) != len(beta_set):
            current_depth=current_depth+1
            recursive_check(current_depth)  # Recurse until sets are equal
        else:
            shirotate_log.write("OVERLAP THRESHOLD: "+str(args.overlap_threshold)+"\n")

    if args.overlap_threshold == 0.0:
        alpha_set=[i+1 for i in range(alpha_homo_idx)]
        beta_set=[i+1 for i in range(alpha_homo_idx)]
    else:
        recursive_check(0)

    alpha_set=sorted(alpha_set)
    beta_set=sorted(beta_set)
    shirotate_log.write("\nINITIAL OVERLAP\n")
    shirotate_log.write("=====================\n")
    print_overlap(initial_S_ab,alpha_homo_idx,7)

    shirotate_log.write("\n")
    shirotate_log.write(f"Alpha set used in rotation {len(alpha_set)} MOs: ")
    if args.overlap_threshold == 0.0:
        shirotate_log.write(f"[All] = {alpha_homo_idx} occ.")
    else:
        h=0
        shirotate_log.write("\n")
        for orb in alpha_set:
            h=h+1
            shirotate_log.write(f"{orb}  ")
            if h%10==0:
                shirotate_log.write(f"\n")

    shirotate_log.write("\nBeta reference set: ")
    if args.overlap_threshold==0.0:
        shirotate_log.write(f"[All] = {beta_sumo_idx-1} occ + 1 vir.")
    else:
        h=0
        shirotate_log.write("\n")
        for orb in beta_set:
            h=h+1
            shirotate_log.write(f"{orb}  ")
            if h%10==0:
                shirotate_log.write(f"\n")
    shirotate_log.write("\n")
    #Define the shape of rotation matrix
    N = len(alpha_set)
    num_elements = N * (N - 1) // 2 # Number of unique elements in X


    # Select alpha and beta orbitals by sets determined by overlap threshold
    A=alpha_coef[:,np.array(alpha_set)-1]
    B=beta_coef[:,np.array(beta_set)-1]
    s_ab_matrix=jnp.dot(A.T,jnp.dot(S_basis, B))


    # Get the initial guess for vector X to form antisymmetric matrix
    intial_guess=np.zeros(num_elements)
    row_indices, col_indices = jnp.triu_indices(N, k=1)
    l=0
    for i, j in zip(row_indices, col_indices):
        abs_overl=abs(s_ab_matrix[i,j])
        if abs_overl>0.90 or abs_overl<0.10:
            intial_guess[l] = 0.0
        else:
            intial_guess[l]=-s_ab_matrix[i,j]+np.random.uniform(-0.1,0.1)
        l =l+ 1

    def antisymm_mat_from_vec(vec):
        X = jnp.zeros((N, N))
        triu_indices = jnp.triu_indices(N, k=1)
        X = X.at[triu_indices].set(vec)
        X = X - X.T
        return X

    def objective_function(vec, s_ab_matrix):
        # Form antisymmetric matrix from vec
        X=antisymm_mat_from_vec(vec)
        # Form rotation matrix from X
        R=expm(X)
        # Compute S' = C'_alpha.T S_ao C_beta
        #            = (C_alpha * R).T S_ao C_beta
        #            = R.T C_alpha.T S_ao C_beta
        overlap = jnp.dot(R.T, s_ab_matrix)
        # Compute penalty by Frobenius Norm of ABS(overlap) subtract I (identity matrix)
        obj_matrix=jnp.abs(overlap)-jnp.eye(N)
        J2=jnp.sum(jnp.square(obj_matrix))
        # Return the sum of squared differences
        return J2
    grad_fn = jax.grad(objective_function)

    def callback_func(xk):
        J2 = objective_function(xk, s_ab_matrix)
        shirotate_log.write(f"{callback_func.step:5d}   {J2:10.8f} \n")
        callback_func.step += 1
    callback_func.step = 1

    shirotate_log.write("=============================\n")
    shirotate_log.write("         CONVERGENCE\n")
    shirotate_log.write("=============================\n")
    shirotate_log.write(" step   objective function  \n")
    shirotate_log.write("-----------------------------\n")

    result = minimize(lambda v, s_ab_matrix: np.asarray(objective_function(v, s_ab_matrix)),
                  intial_guess, args=(s_ab_matrix),
                  jac=lambda v, s_ab_matrix: np.asarray(grad_fn(v, s_ab_matrix)),
                  method='CG',
                  callback=callback_func)


   # Extract the final result
    final_rotation_matrix=expm(antisymm_mat_from_vec(result.x))
    shirotate_log.write("--------------------\n")
    final_J2=objective_function(result.x,s_ab_matrix)


    shirotate_log.write("\n")
    shirotate_log.write(f"The minimization is done after {result.nit} steps.\n")
    shirotate_log.write(f"Squared Frobenius norm     ||abs(S)-1||2F  =  {final_J2:.5f} \n")
    numb_steps=result.nit

    rotated_alpha=np.dot(A,final_rotation_matrix)
    final_occ_alpha_coef=np.delete(alpha_coef,np.array(alpha_set)-1,axis=1)
    final_occ_alpha_coef=np.hstack([final_occ_alpha_coef,rotated_alpha])
    if np.sum((final_occ_alpha_coef.T@S_basis@final_occ_alpha_coef).round(decimals=0)==np.eye(final_occ_alpha_coef.shape[1]))==final_occ_alpha_coef.shape[1]*final_occ_alpha_coef.shape[1]:
        shirotate_log.write("\n")
    else:
        shirotate_log.write("WARNING:C.T @ S @ C is different from I\n")
    final_occ_alpha_energies=(final_occ_alpha_coef.T@F_alfa@final_occ_alpha_coef).diagonal()
    old_alpha_energies=(c_a@F_alfa@c_a.T).diagonal() # This is NOT wrong. Just because NWChem print each MO line by line

    a_SOMO_energy=final_occ_alpha_energies[-1]


    ### new_alpha_energies: new alpha_occupied energies
    argsort_energy=np.argsort(final_occ_alpha_energies) # Get the index of the sorted energies
    final_occ_alpha_energies=final_occ_alpha_energies[argsort_energy]
    S_aSOMO_bSUMO=final_occ_alpha_coef[:,-1].T@S_basis@beta_coef[:,beta_sumo_idx-1]
    shirotate_log.write(f"Overlap between (rotated) a_SOMO and beta_SUMO = {S_aSOMO_bSUMO:.4f}\n\n")
    final_occ_alpha_coef=final_occ_alpha_coef[:,argsort_energy]
    ###################################################
    # PRINT LAST TEN OVERLAP
    ##################################################
    last_step_overlap=final_occ_alpha_coef.T@S_basis@beta_coef
    shirotate_log.write("\nFINAL OVERLAP\n")
    shirotate_log.write("=====================\n")
    print_overlap(last_step_overlap,alpha_homo_idx,7)

    shirotate_log.write("\n")
    if final_J2>0.5:
        shirotate_log.write("WARNING: There might some orbital swappings.\n")

    shirotate_log.write("\n")

    if abs(S_aSOMO_bSUMO)<0.97:
        shirotate_log.write("WARNING: Frobenius norm is too high.\n")
        shirotate_log.write("JOB FAILED\n")
        return False,"failed",final_J2
    print("\n")

    indices=[]
    for i in range(alpha_homo_idx):
        if (i+1) not in alpha_set:
            indices.append("\n")

    for i in range(len(alpha_set)-1):
        indices.append("\n")
    indices.append(" matched with beta SUMO\n")
    indices=np.array(indices)
    indices=indices[argsort_energy]
    shirotate_log.write("==============================\n")
    shirotate_log.write("       Alpha MO energies   eV\n")
    shirotate_log.write("==============================\n")
    shirotate_log.write(" #        scf       rotated  \n")
    shirotate_log.write("------------------------------\n")
    for idx in range(alpha_homo_idx):
        shirotate_log.write(f'''{idx+1:3d}     {old_alpha_energies[idx]* toEV:7.3f}     {final_occ_alpha_energies[idx]* toEV:7.3f}     {indices[idx]}''')
    shirotate_log.write("\n")
    shirotate_log.write("\n")

    canonical_beta_energies=(c_b@F_beta@c_b.T).diagonal()
    shirotate_log.write("==============================\n")
    shirotate_log.write("Alpha and Beta MO energies eV\n")
    shirotate_log.write("==============================\n")
    shirotate_log.write(" #       alpha        beta   \n")
    shirotate_log.write("------------------------------\n")
    for idx in range(beta_sumo_idx):
        shirotate_log.write(f'''{idx+1:3d}     {final_occ_alpha_energies[idx]* toEV:7.3f}     {canonical_beta_energies[idx]* toEV:7.3f}   \n''')
    shirotate_log.write("\n")
    shirotate_log.write("\n")
    b_HOMO_energy=canonical_beta_energies[beta_sumo_idx-2] #python start at 0
    a_HOMO_energy=np.max(final_occ_alpha_energies)
    shi_gap=(a_HOMO_energy-a_SOMO_energy)* toEV
    shirotate_log.write("==============================\n")
    shirotate_log.write("           SHI GAP          eV\n")
    shirotate_log.write("==============================\n")
    shirotate_log.write(f"a HOMO: {a_HOMO_energy* toEV:.4f} eV\n")
    shirotate_log.write(f"a SOMO: {a_SOMO_energy* toEV:.4f} eV\n")
    shirotate_log.write(f"SHI gap: {shi_gap:.4f} eV\n")
    full_rotated_alpha_energies=np.concatenate((final_occ_alpha_energies,old_alpha_energies[alpha_homo_idx:]))
    full_rotated_alpha_coef=np.vstack((final_occ_alpha_coef.T,c_a[alpha_homo_idx:,:]))
    shirotate_log.write("\n")

    if abs(a_HOMO_energy-old_alpha_energies[alpha_homo_idx-1])<1e-6:
        shirotate_log.write("\nWARNING: Alpha canonical HOMO is changed!\n")
        print(a_HOMO_energy)
        print(old_alpha_energies[alpha_homo_idx-1])

    if shi_gap == 0.0 and b_HOMO_energy < a_HOMO_energy:
        shirotate_log.write("The molecule does NOT exhibit SHI characteristics. (non SHI)\n")
    elif shi_gap == 0.0 and b_HOMO_energy > a_HOMO_energy:
        shirotate_log.write("The molecule has paritial SHI characteristics. (partial SHI)")
    elif shi_gap > 0.0 and b_HOMO_energy > a_HOMO_energy:
        shirotate_log.write("The molecule has SHI characteristics. (SHI)")

    if args.movecs:
        numb_occ_cubes=4
        numb_vir_cubes=1
        bash_result=subprocess.run(["bash",f'''cd {current_directory}'''],capture_output=True,text=True)
        nwchem_top,mov2asc,asc2mov,nwchem,dplot = get_enviroment_variables()
        nwchem_inputs=glob.glob(current_directory+'/*.nw')
        nw_file="input.nw"
        shirotate_log.write(nw_file)

        def generate_nwchem_cube(file_name,nw_file,movecs_file,list_mo,grid,cube_type):
            copy_command=f'''cp {nw_file} {file_name}.nw'''
            excute_bash_command(copy_command)
            dplot_command=f'''{dplot} -i {file_name}.nw -m {movecs_file} {list_mo} {cube_type} -g {grid}'''
            print(dplot_command)
            excute_bash_command(dplot_command)
            nwchem_command=f'''{nwchem} dplot.nw > dplot.out'''
            excute_bash_command(nwchem_command)
            clean_command=f'''rm {file_name}.nw'''
            excute_bash_command(clean_command)

        shirotate_log.write("----------------------------")
        shirotate_log.write("Generation cube files by NWChem and dplot\n")
        mov2asc_command=f'''{mov2asc} {nbas} molecule.movecs canonical.ascii'''
        excute_bash_command(mov2asc_command)

        # CREATE ROTATED MOVECS
        write_ascii_movec("rotated",alpha_homo_idx,beta_homo_idx,full_rotated_alpha_energies,canonical_beta_energies,full_rotated_alpha_coef,c_b)
        asc2mov_command=f'''{asc2mov} {nbas} rotated.ascii rotated.movecs'''
        excute_bash_command(asc2mov_command)

        # Canonical cubes
        generate_nwchem_cube(f"canonical",nw_file,"molecule.movecs",f"-l {alpha_homo_idx-numb_occ_cubes}-{alpha_homo_idx+numb_vir_cubes}",100,"-a")
        # Rotated cubes
        generate_nwchem_cube(f"rotated",nw_file,"rotated.movecs",f"-l {alpha_homo_idx-numb_occ_cubes}-{alpha_homo_idx+numb_vir_cubes}",100,"")
        # Spin density
        generate_nwchem_cube(f"canonical",nw_file,"molecule.movecs","",100,"-d spin")

    return True, shi_gap,final_J2

for i in range(4):
    if i!=0:
        shirotate_log.write('======================================================================\n')
        shirotate_log.write('======================================================================\n')
        shirotate_log.write("\n")
        shirotate_log.write("########################\n")
        shirotate_log.write("  RESTART CALCULATION\n")
        shirotate_log.write("########################\n")
    job_done,shi_gap,final_J2=shi_rotate()
    if job_done == True:
        print(f'''SHI gap: {shi_gap:.4f} J^2: {final_J2:.4e}''')
        break
    elif i==3 and job_done == False:
        shirotate_log.write("Job failed after 4 attempts. \n")
        shirotate_log.write("########################\n")
        shirotate_log.write("        FAILED\n")
        shirotate_log.write("########################\n")
        print(f'''FAILED J^2: {final_J2:.4e}''')

end_time = time.time()
elapsed_time = end_time - start_time
shirotate_log.write(f"\n\nTotal elapsed time: {elapsed_time:.1f} seconds\n")
shirotate_log.close()
