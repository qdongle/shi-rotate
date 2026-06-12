#!/usr/bin/env python3
import argparse
import glob

# from scipy.linalg import expm
import os
import subprocess
import sys
import time
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from jax.scipy.linalg import expm
from scipy.optimize import minimize

start_time = time.time()

parser = argparse.ArgumentParser(
    description="SHI-rotate: A tool for SOMO-HOMO inversion."
)
# Add required arguments
parser.add_argument(
    "file47", default=None, help=".gen (FILE47) from NWChem or Gaussian output"
)

# Add optional arguments
parser.add_argument(
    "--overlap-threshold",
    "-t",
    default=0.00,
    type=float,
    help="Overlap threshold between alpha and beta",
)
parser.add_argument(
    "--clean", "-c", default=False, help="Remove all cube files, rotated.movecs."
)

# TO-DOs or improve. These options help user to generate cube files
parser.add_argument(
    "--movecs-nwchem",
    "-movecs",
    dest="movecs",
    default=None,
    help="NWChem movecs file: to restart NWChem to generate cube files.",
)
parser.add_argument(
    "--fch-gaussian",
    "-fch",
    dest="fch",
    default=None,
    help="Gaussian fch file: to generate cube files.",
)

args = parser.parse_args()


# JAX setup
config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Create a log file and print the header
shirotate_log = open("shi_rotate.log", "w")
shirotate_log.write(r"""
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
-----------------------------------------------------------------------------""")


# Process input files---------------------------------------------------------
current_directory = os.getcwd()
file47 = args.file47
if args.file47 == None:
    file47s = glob.glob(current_directory + "/*.gen")
    if len(file47s) > 1:
        print("Please specify which file47 file will be used:")
        for gen in file47s:
            print(f"""     {gen.split("/")[-1]}""")
        sys.exit()
    elif len(file47s) == 0:
        print("There is no file47 in the current directory.")
        sys.exit()
    else:
        file47 = file47s[0]
shirotate_log.write(f"\nFILE47   : {file47.split('/')[-1]}\n")
shirotate_log.write(f"Work directory: {current_directory} \n")
# -----------------------------------------------------------------------------


toEV = 27.2113245702


def custom_format(num):
    if num >= 0:
        return format(num, "17.15E")  # 17 characters for positive mantissa
    else:
        return format(num, "18.15E")  # 18 characters for negative mantissa


def formatted_overlap(number):
    if abs(number) < 0.01:
        return "       "
    else:
        return f"{number: >+7.4f}".replace("+", " ")


def print_overlap(smat, a_homo_idx, numb):
    header = "      |"
    for i in range(-numb, 0):
        header += f"  b_{a_homo_idx + i + 1:04d}"
    header += "\n"
    shirotate_log.write(header)
    shirotate_log.write(
        "---------------------------------------------------------------\n"
    )
    for a in range(-numb, 0):
        row = f"a_{a + a_homo_idx + 1:04d}|"
        for i in range(-numb, 0):
            row += f" {formatted_overlap(smat[a][i])}"
        shirotate_log.write(row + "\n")


def write_ascii_array(file, array):
    for i in range(len(array)):
        if (i + 1) % 3 != 0:
            file.write("   " + custom_format(array[i]))
            if (i + 1) == len(array):
                file.write("\n")
        else:
            file.write("   " + custom_format(array[i]))
            file.write("\n")


def write_ascii_movec(
    ascii_movecs_name,
    num_elec_alpha,
    num_elec_beta,
    alpha_energies,
    beta_energies,
    c_a,
    c_b,
):
    new_ascii_path = current_directory + f"/{ascii_movecs_name}.ascii"
    nbas = c_a.shape[0]
    with open(current_directory + "/canonical.ascii", "r") as f:
        with open(new_ascii_path, "w") as g:
            for i in range(14):
                line = f.readline()
                g.writelines(line)
            # Write alpha occupation
            occ = np.concatenate(
                (np.ones(num_elec_alpha), np.zeros(nbas - num_elec_alpha))
            )
            write_ascii_array(g, occ)
            # Write alpha energy
            write_ascii_array(g, alpha_energies)
            # Write alpha coeff
            for i in range(nbas):
                write_ascii_array(g, c_a[i])
            # Write beta occupation
            occ = np.concatenate(
                (np.ones(num_elec_beta), np.zeros(nbas - num_elec_beta))
            )
            write_ascii_array(g, occ)
            # Write beta energy
            write_ascii_array(g, beta_energies)
            # Write beta coeff
            for i in range(nbas):
                write_ascii_array(g, c_b[i])
            g.writelines(f.readlines()[-1])


def execute_bash(command):
    bash_result = subprocess.run(command.split(), capture_output=True, text=True)
    if bash_result.returncode != 0:
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
            return float(string.split("+")[0]) * 10 ** (int(string.split("+")[1]))
        elif "-" in string:
            return float(string.split("-")[0]) * 10 ** (-int(string.split("-")[1]))


def get_enviroment_variables():
    bash_result = subprocess.run("which nwchem".split(), capture_output=True, text=True)
    if bash_result.stdout == "":
        print("$NWCHEM is not set. Please specify the folder of NWChem")
        sys.exit()
    nwchem_top = "/".join(bash_result.stdout.split("/")[:-3])
    mov2asc = nwchem_top + "/contrib/mov2asc/mov2asc"
    asc2mov = nwchem_top + "/contrib/mov2asc/asc2mov"
    nwchem = nwchem_top + "/bin/LINUX64/nwchem"
    bash_result = subprocess.run(
        "which dplot.py", shell=True, capture_output=True, text=True
    )
    dplot = bash_result.stdout.split("\n")[0]
    shirotate_log.write("----------------------------\n")
    shirotate_log.write(f"NWChem: {nwchem_top}\n")
    shirotate_log.write(f"mov2asc: {mov2asc}\n")
    shirotate_log.write(f"asc2mov: {asc2mov}\n")
    shirotate_log.write(f"dplot: {dplot}")
    return nwchem_top, mov2asc, asc2mov, nwchem, dplot


def get_upper_triangular_matrix_FILE47(line, f, number_of_matrices, nbas):
    temp = []
    line = f.readline()  # Skip the header line
    alpha_matrix = []
    while "$END" not in line:
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = f.readline()
        if "END" in line:
            break
    if number_of_matrices == 2:
        temp_a = temp[: int(nbas * (nbas - 1) / 2 + nbas)]
        temp_b = temp[int(nbas * (nbas - 1) / 2 + nbas) :]
        alpha_matrix = np.zeros((nbas, nbas), dtype=np.float64)
        beta_matrix = np.zeros((nbas, nbas), dtype=np.float64)
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                alpha_matrix[i][j] = temp_a[k]
                alpha_matrix[j][i] = temp_a[k]
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                beta_matrix[i][j] = temp_b[k]
                beta_matrix[j][i] = temp_b[k]
        return alpha_matrix, beta_matrix, line
    else:
        matrix = np.zeros((nbas, nbas), dtype=np.float64)
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                matrix[i][j] = temp[k]
                matrix[j][i] = temp[k]
        return matrix, line


def get_square_matrix_FILE47(line, f, number_of_matrices, nbas):
    temp = []
    line = f.readline()  # Skip the header line
    while "$END" not in line:
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = f.readline()
    if number_of_matrices == 2:
        alpha_matrix = np.array(temp[: nbas * nbas], dtype=np.float64).reshape(
            nbas, nbas
        )
        beta_matrix = np.array(temp[nbas * nbas :], dtype=np.float64).reshape(
            nbas, nbas
        )
        return alpha_matrix, beta_matrix, line
    else:
        matrix = np.array(temp, dtype=np.float64).reshape(nbas, nbas)
        return matrix, line


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
        line = f.readline()
        if "OPEN" in line:
            open_shell = True
        else:
            open_shell = False
            shirotate_log.write(
                "Please check your molecular system (geometry, charge, ab initio input)"
            )
            sys.exit(
                "FILE47 contains the closed shell calculation. SHI-rotate needs open shell calculation."
            )

        # Get number of basis functions: NBAS
        # The first has a format like that: $GENNBO  UPPER  BODM  BOHR  NATOMS= 3  NBAS=  24 $END
        #  Next to the string "NBAS=" is the number of basis functions
        if "NBAS" in line:
            # If nbas<100, Fortran developer leaves a space after "NBAS= 99"
            # this is different with the case "NBAS=100"
            for token in line.split():
                if token.startswith("NBAS="):
                    nbas_str = token[len("NBAS=") :]
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
            sys.exit(
                "NBAS is not found in the first line of FILE47. Please check the FILE47."
            )
        shirotate_log.write(f"\n\n\nNumber of basis functions (NBAS): {nbas}\n")

        # Parse the rest of the file
        while line:
            if "$LCAOMO" in line:
                alpha_coef_matrix, beta_coef_matrix, line = get_square_matrix_FILE47(
                    line, f, 2, nbas
                )
            elif "$DENSITY" in line:
                alpha_density_matrix, beta_density_matrix, line = (
                    get_upper_triangular_matrix_FILE47(line, f, 2, nbas)
                )
            elif "$FOCK" in line:
                alpha_fock_matrix, beta_fock_matrix, line = (
                    get_upper_triangular_matrix_FILE47(line, f, 2, nbas)
                )
            elif "$OVERLAP" in line:
                overlap_matrix, line = get_upper_triangular_matrix_FILE47(
                    line, f, 1, nbas
                )
            line = f.readline()

        # Calculate numbers of alpha, beta electrons by density matrix.
        alpha_elec = int(round(np.trace(alpha_density_matrix @ overlap_matrix)))
        shirotate_log.write(f"                 Alpha electrons: {alpha_elec}\n")
        beta_elec = int(round(np.trace(beta_density_matrix @ overlap_matrix)))
        shirotate_log.write(f"                  Beta electrons: {beta_elec}\n")

        return (
            alpha_coef_matrix,
            beta_coef_matrix,
            overlap_matrix,
            alpha_density_matrix,
            beta_density_matrix,
            alpha_fock_matrix,
            beta_fock_matrix,
            nbas,
            alpha_elec,
            beta_elec,
        )


def matching_orbital(ovlp_ab, threshold=0.97):
    max_idx = np.argmax(abs(ovlp_ab))
    max_ovlp = ovlp_ab[max_idx]
    if abs(max_ovlp) >= threshold:
        return max_idx, max_ovlp
    return None, None


# Get C, S_ao, DM, Fock matrices, nbas, number of alpha-beta electrons from file47
# density matrix is not used, they are discarded.
c_a, c_b, S_bf, _, _, F_alfa, F_beta, nbas, n_alpha_elec, n_beta_elec = (
    get_data_from_gen(file47)
)
if n_beta_elec != n_alpha_elec - 1:
    shirotate_log.write(
        f"NOTE: This molecule is not a doublet radical. # alpha electrons = {n_alpha_elec}  ,  # beta electrons = {n_beta_elec}"
    )
    shirotate_log.write(
        "      SHI-rotate is epxerimental for triplet, quartet radicals,..."
    )

beta_sumo_idx = n_beta_elec + 1
shirotate_log.write("\n")
shirotate_log.write(f"                Alpha HOMO index: {n_alpha_elec}\n")
shirotate_log.write(f"                 Beta SUMO index: {beta_sumo_idx}\n")
# Get the MO coefs
# alpha_coef and beta_coef are NOT full coefs matrix. It only include alpha occupied  (all occ)
# and beta occupied + beta SUMO (all occ+1virt)
initial_S_ab = reduce(np.dot, (c_a, S_bf, c_b.T))


# Alpha set contains orbitals that will be used in rotation
alpha_set = []
beta_set = []
recur_step = 0
max_depth = 10


def check_overlap(a, b):
    if a == "all":
        ib = b - 1
        for ia in range(beta_sumo_idx):
            if abs(initial_S_ab[ia][ib]) > args.overlap_threshold:
                if not (ia + 1 in alpha_set):
                    alpha_set.append(ia + 1)
    if b == "all":
        ia = a - 1
        for ib in range(beta_sumo_idx):
            if abs(initial_S_ab[ia][ib]) > args.overlap_threshold:
                if not (ib + 1 in beta_set):
                    beta_set.append(ib + 1)


def recursive_check(current_depth=0):
    if current_depth >= max_depth:
        shirotate_log.write("Maximum recursion depth reached!\n")
        return

    if len(alpha_set) == 0:
        check_overlap("all", beta_sumo_idx)
        for a in alpha_set:
            check_overlap(a, "all")
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
        current_depth = current_depth + 1
        recursive_check(current_depth)  # Recurse until sets are equal
    else:
        shirotate_log.write("OVERLAP THRESHOLD: " + str(args.overlap_threshold) + "\n")


if args.overlap_threshold == 0.0:
    alpha_set = [i + 1 for i in range(n_alpha_elec)]
    beta_set = [i + 1 for i in range(n_alpha_elec)]
else:
    recursive_check(0)

alpha_set = sorted(alpha_set)
beta_set = sorted(beta_set)
shirotate_log.write("\nINITIAL OVERLAP\n")
shirotate_log.write("=====================\n")
print_overlap(initial_S_ab[np.ix_(alpha_set, beta_set)], n_alpha_elec, 7)


shirotate_log.write("\n")
shirotate_log.write(f"Alpha set used in rotation {len(alpha_set)} MOs: ")
if args.overlap_threshold == 0.0:
    shirotate_log.write(f"[All] = {n_alpha_elec} occ.")
else:
    h = 0
    shirotate_log.write("\n")
    for orb in alpha_set:
        h = h + 1
        shirotate_log.write(f"{orb}  ")
        if h % 10 == 0:
            shirotate_log.write("\n")

shirotate_log.write("\nBeta reference set: ")
if args.overlap_threshold == 0.0:
    shirotate_log.write(f"[All] = {beta_sumo_idx - 1} occ + 1 vir.")
else:
    h = 0
    shirotate_log.write("\n")
    for orb in beta_set:
        h = h + 1
        shirotate_log.write(f"{orb}  ")
        shirotate_log.write("\n") if h % 10 == 0 else None
shirotate_log.write("\n")
# Define the shape of rotation matrix
N = len(alpha_set)
n_uniq_elem = N * (N - 1) // 2  # Number of unique elements in X


# Select alpha and beta orbitals by sets determined by overlap threshold
A = c_a.T[:, np.array(alpha_set) - 1]
B = c_b.T[:, np.array(beta_set) - 1]
S_ab = jnp.dot(A.T, jnp.dot(S_bf, B))


def permuted_identity(overlap, threshold=0.05):
    abs_overlap = abs(overlap)
    n_orbital = abs_overlap.shape[0]
    strong_ovlp = []
    for orb_idx in range(n_orbital):
        row = abs_overlap[orb_idx]
        # Find index of max element
        max_idx = np.argmax(row)
        max_ovlp = row[max_idx]
        if max_ovlp < (1.0 - threshold):
            return False  # No element close to 1
        strong_ovlp.append(max_idx)

    # Check if all strong overlap are unique [set in Python will remove duplicates]
    if len(set(strong_ovlp)) != n_orbital:
        return False

    # Sort rows by their strong overlap index --> Create a identity permuted matrix
    sorted_rows = sorted(
        range(n_orbital), key=lambda orb_idx: np.argmax(abs_overlap[orb_idx])
    )
    permuted_mat = abs_overlap[sorted_rows]

    # Verify
    for orb_idx in range(n_orbital):
        if permuted_mat[orb_idx, orb_idx] < (1.0 - threshold):
            return False
        for j in range(n_orbital):
            if orb_idx != j and permuted_mat[orb_idx, j] > threshold:
                return False
    return True


def antisymm_mat_from_vec(vec):
    X = jnp.zeros((N, N))
    triu_indices = jnp.triu_indices(N, k=1)
    X = X.at[triu_indices].set(vec)
    X = X - X.T
    return X


def objective_func(vec, s_ab_matrix):
    # Form antisymmetric matrix from vec
    X = antisymm_mat_from_vec(vec)
    # Form rotation matrix from X
    R = expm(X)
    # Compute S' = C'_alpha.T S_ao C_beta
    #            = (C_alpha * R).T S_ao C_beta
    #            = R.T C_alpha.T S_ao C_beta
    overlap = jnp.dot(R.T, s_ab_matrix)
    # Compute penalty by Frobenius Norm of ABS(overlap) subtract I (identity matrix)
    obj_matrix = overlap**2 - jnp.eye(N)
    J2 = jnp.sum(jnp.square(obj_matrix))

    return J2


def shi_rotate():

    # Get the initial guess for vector X to form antisymmetric matrix
    init_guess = np.zeros(n_uniq_elem)
    row_indices, col_indices = jnp.triu_indices(N, k=1)
    a_idx = 0
    for i, j in zip(row_indices, col_indices):
        abs_overl = abs(S_ab[i, j])
        if abs_overl > 0.90 or abs_overl < 0.10:
            init_guess[a_idx] = 0.0
        else:
            init_guess[a_idx] = -S_ab[i, j] + np.random.uniform(-0.1, 0.1)
        a_idx += 1

    grad_fn = jax.grad(objective_func)

    step = 0

    def callback_func(xk):
        nonlocal step
        J2 = objective_func(xk, S_ab)
        shirotate_log.write(f"{step:<4d}   {J2:>13.8f} \n")
        step += 1

    shirotate_log.write("====================\n")
    shirotate_log.write("    CONVERGENCE\n")
    shirotate_log.write("====================\n")
    shirotate_log.write("step      obj. func.\n")
    shirotate_log.write("--------------------\n")

    result = minimize(
        lambda v, s_ab_matrix: np.asarray(objective_func(v, s_ab_matrix)),
        init_guess,
        args=(S_ab),
        jac=lambda v, s_ab_matrix: np.asarray(grad_fn(v, s_ab_matrix)),
        method="CG",
        callback=callback_func,
    )

    shirotate_log.write("\n")
    shirotate_log.write(f"The minimization is done after {result.nit} steps.\n")
    final_J2 = objective_func(result.x, S_ab)
    shirotate_log.write(f"Squared Frobenius norm     ||S^2-1||2F  =  {final_J2:.5f} \n")

    # Extract the final result
    final_rotation_matrix = expm(antisymm_mat_from_vec(result.x))
    shirotate_log.write("--------------------\n")

    rotated_A = np.dot(A, final_rotation_matrix)

    rotated_alpha_ener = (rotated_A.T @ F_alfa @ rotated_A).diagonal()
    cmo_alpha_ener = (
        c_a @ F_alfa @ c_a.T
    ).diagonal()  # This is NOT wrong. Just because NWChem print each MO line by line

    a_SOMO_energy = rotated_alpha_ener[-1]

    ### new_alpha_energies: new alpha_occupied energies
    argsort_ener = np.argsort(
        rotated_alpha_ener
    )  # Get the index of the sorted energies
    rotated_alpha_ener = rotated_alpha_ener[argsort_ener]
    # -------------
    rotated_A = rotated_A[:, argsort_ener]
    ###################################################
    # PRINT LAST TEN OVERLAP
    ##################################################
    final_overlap = rotated_A.T @ S_bf @ B
    shirotate_log.write("\nFINAL OVERLAP\n")
    shirotate_log.write("=====================\n")
    print_overlap(final_overlap, n_alpha_elec, 7)

    shirotate_log.write("\n")
    if final_J2 > 0.5:
        shirotate_log.write("WARNING: There might some orbital swappings.\n")

    shirotate_log.write("\n")

    if not permuted_identity(final_overlap):
        shirotate_log.write("WARNING: Overlap matrix is not permuted identity.\n")
        shirotate_log.write("JOB FAILED\n")
        return False, "failed", final_J2
    print("\n")

    cmo_beta_ener = (c_b @ F_beta @ c_b.T).diagonal()
    shirotate_log.write("================================\n")
    shirotate_log.write("    Alpha MO energies (eV)\n")
    shirotate_log.write("================================\n")
    shirotate_log.write("i       canonical        rotated  \n")
    shirotate_log.write("--------------------------------\n")
    for a_idx in range(n_alpha_elec):
        shirotate_log.write(
            f"""{a_idx + 1:<4d} {cmo_alpha_ener[a_idx] * toEV:>12.3f}   {rotated_alpha_ener[a_idx] * toEV:>12.3f}   \n"""
        )
    shirotate_log.write("\n")
    shirotate_log.write("\n")

    shirotate_log.write("==========================================\n")
    shirotate_log.write("            MO energies eV\n")
    shirotate_log.write("==========================================\n")
    shirotate_log.write("i    rotated ALPHA  canonical BETA    S_ii\n")
    shirotate_log.write("------------------------------------------\n")
    for a_idx in range(beta_sumo_idx):
        ovlp_str = (
            f"{final_overlap[a_idx, a_idx]:>5.2f}"
            if abs(final_overlap[a_idx, a_idx]) > 0.01
            else "  0  "
        )
        shirotate_log.write(
            f"""{a_idx + 1:<4d} {rotated_alpha_ener[a_idx] * toEV:>13.3f}   {cmo_beta_ener[a_idx] * toEV:>13.3f}   {ovlp_str}\n"""
        )

    shirotate_log.write("===============================\n")
    shirotate_log.write("Matching rotated alpha and beta\n")
    shirotate_log.write("===============================\n")
    shirotate_log.write("alpha(i)  beta(j)  S_ij    swap\n")
    shirotate_log.write("-------------------------------\n")
    for a_idx in range(n_alpha_elec):
        b_idx, ovlp = matching_orbital(final_overlap[a_idx, :])
        note = "   " if a_idx == b_idx else "yes"
        shirotate_log.write(
            f"""   {a_idx + 1:>5d}  {b_idx + 1:<5d}    {ovlp:>5.2f}    {note} \n"""
        )
    shirotate_log.write("\n")
    shirotate_log.write("\n")
    b_HOMO_energy = cmo_beta_ener[beta_sumo_idx - 2]  # python start at 0
    a_HOMO_energy = np.max(rotated_alpha_ener)
    shi_gap = (a_HOMO_energy - a_SOMO_energy) * toEV
    shirotate_log.write("==============================\n")
    shirotate_log.write("           SHI GAP          eV\n")
    shirotate_log.write("==============================\n")
    shirotate_log.write(f"a HOMO: {a_HOMO_energy * toEV:.4f} eV\n")
    shirotate_log.write(f"a SOMO: {a_SOMO_energy * toEV:.4f} eV\n")
    shirotate_log.write(f"SHI gap: {shi_gap:.4f} eV\n")
    full_rotated_alpha_energies = np.concatenate(
        (rotated_alpha_ener, cmo_alpha_ener[n_alpha_elec:])
    )
    full_A = np.vstack((rotated_A.T, c_a[n_alpha_elec:, :]))
    shirotate_log.write("\n")

    if abs(a_HOMO_energy - cmo_alpha_ener[n_alpha_elec - 1]) < 1e-6:
        shirotate_log.write("\nWARNING: Alpha canonical HOMO is changed!\n")
        print(a_HOMO_energy)
        print(cmo_alpha_ener[n_alpha_elec - 1])

    if shi_gap == 0.0 and b_HOMO_energy < a_HOMO_energy:
        shirotate_log.write("Classification: (non SHI)\n")
    elif shi_gap == 0.0 and b_HOMO_energy > a_HOMO_energy:
        shirotate_log.write("Classification: (partial SHI)")
    elif shi_gap > 0.0 and b_HOMO_energy > a_HOMO_energy:
        shirotate_log.write("Classification: (SHI)")

    if args.movecs:
        numb_occ_cubes = 4
        numb_vir_cubes = 1
        bash_result = subprocess.run(
            ["bash", f"""cd {current_directory}"""], capture_output=True, text=True
        )
        nwchem_top, mov2asc, asc2mov, nwchem, dplot = get_enviroment_variables()
        nwchem_inputs = glob.glob(current_directory + "/*.nw")
        nw_file = "input.nw"
        shirotate_log.write(nw_file)

        def gen_nwchem_cube(file_name, nw_file, movecs_file, list_mo, grid, cube_type):
            copy_command = f"""cp {nw_file} {file_name}.nw"""
            execute_bash(copy_command)
            dplot_command = f"""{dplot} -i {file_name}.nw -m {movecs_file} {list_mo} {cube_type} -g {grid}"""
            print(dplot_command)
            execute_bash(dplot_command)
            execute_bash(f"mv dplot.nw {file_name}_dplot.nw")

        shirotate_log.write("----------------------------")
        shirotate_log.write("Generation cube files by NWChem and dplot\n")
        mov2asc_command = f"""{mov2asc} {nbas} molecule.movecs canonical.ascii"""
        execute_bash(mov2asc_command)

        # CREATE ROTATED MOVECS
        write_ascii_movec(
            "rotated",
            n_alpha_elec,
            n_beta_elec,
            full_rotated_alpha_energies,
            cmo_beta_ener,
            full_A,
            c_b,
        )
        asc2mov_command = f"""{asc2mov} {nbas} rotated.ascii rotated.movecs"""
        execute_bash(asc2mov_command)

        # Canonical cubes
        gen_nwchem_cube(
            "canonical",
            nw_file,
            "molecule.movecs",
            f"-l {n_alpha_elec - numb_occ_cubes}-{n_alpha_elec + numb_vir_cubes}",
            100,
            "-a",
        )
        # Rotated cubes
        gen_nwchem_cube(
            "rotated",
            nw_file,
            "rotated.movecs",
            f"-l {n_alpha_elec - numb_occ_cubes}-{n_alpha_elec + numb_vir_cubes}",
            100,
            "",
        )
        # Spin density
        gen_nwchem_cube("canonical", nw_file, "molecule.movecs", "", 100, "-d spin")

    return True, shi_gap, final_J2


for i in range(4):
    if i != 0:
        shirotate_log.write(
            "======================================================================\n"
        )
        shirotate_log.write(
            "======================================================================\n"
        )
        shirotate_log.write("\n")
        shirotate_log.write("########################\n")
        shirotate_log.write("  RESTART CALCULATION\n")
        shirotate_log.write("########################\n")
    job_done, shi_gap, final_J2 = shi_rotate()
    if job_done == True:
        print(f"""SHI gap: {shi_gap:.4f} J^2: {final_J2:.4e}""")
        break
    elif i == 3 and job_done == False:
        shirotate_log.write("Job failed after 4 attempts. \n")
        shirotate_log.write("########################\n")
        shirotate_log.write("        FAILED\n")
        shirotate_log.write("########################\n")
        print(f"""FAILED J^2: {final_J2:.4e}""")

end_time = time.time()
elapsed_time = end_time - start_time
shirotate_log.write(f"\n\nTotal elapsed time: {elapsed_time:.1f} seconds\n")
shirotate_log.close()
