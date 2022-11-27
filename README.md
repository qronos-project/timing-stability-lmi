# Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing

## Code
This folder contains the code for the following publications:

- Doctoral thesis "Safety Verification of Real-Time Control Systems with Flexible Timing" (M. Gaukler, 2023)
  See sourceLinksAppendix.pdf for the mapping of thesis contents to source files.

- "Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing" (Gaukler et al., 2019/2020; IFAC World Congress 2020).
   An extended preprint including details and proofs is available at https://arxiv.org/abs/1911.02537

- "Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing (Benchmark Proposal)" (Gaukler, Ulbrich 2019, ARCH - Workshop on Applied Verification for Continuous and Hybrid Systems).
   The original code corresponding to that publication was published on https://github.com/qronos-project/arch19-benchmark-iotiming . It has been merged into this codebase.

- "Analysis of Real-Time Control Systems using First-Order Continuization" (Gaukler, 2020, ARCH - Workshop on Applied Verification for Continuous and Hybrid Systems)
  https://doi.org/10.29007/8nq6
  Examples C3-C5 are in src/qronos/reachability/experiments/continuization.py, see below on how to run.
  Examples F1 and F2 are given as MATLAB/Simulink model in notes/continuization-counterexample-f1-f2/. Just start the provided MATLAB script.

## Precomputed Output

An "output.tar.gz" archive file is avaiiable that contains the source and all outputs, so you don't have to run it yourself.
https://doi.org/10.5281/zenodo.6373637

See sourceLinksAppendix.pdf for the mapping of thesis contents (Gaukler, 2023) to output files.

# Running with docker (recommended)

The run_* scripts only require docker, docker-compose, and appropriate permissions (docker group) for your user.
They are made for a standard Linux environment, e.g., Ubuntu 18.04. At the time of writing, setting up Docker in Microsoft "Windows Subsystem for Linux" (WSL) was not possible easily, so please use a Linux VM instead.

For example, you can use it the following way:

## Setup docker environment
Install Ubuntu 18.04 (or similar) in a virtual machine (VirtualBox, VMWare, etc.) with sufficient RAM. (For building the official output archive, Debian 11 with 16GB RAM was used. If you are experiencing "hangups" or other issues with less RAM, reduce the RAM limit in docker-compose.yml).
```
sudo apt install docker.io docker-compose git ssh
sudo adduser $USER docker
```
Restart (or at least log out and back in).

## Get the source
Open a terminal in the desired working folder.
```
git clone --recurse-submodules https://github.com/qronos-project/timing-stability-lmi mg-diss
```
Note: in this example, the 'mg-diss' branch is used, which corresponds to the doctoral thesis Gaukler (2023).
Or unpack the output.tar.gz archive somewhere and open a terminal there.

## Run
```
# Quick check if everything works:
# Run a small subset of the experiments, just to see if there is no error
./run_experiments_docker.sh ALL --fast

# To run ALL experiments,
# (write all results into logfile.txt; and save outputs to output/,
# Re-create "output.tar.gz" archive with source and outputs):
# This will take about 2 days.
./make_full_archive.sh

# To run a specific experiment only:
./run_experiments_docker.sh arch19 # run ARCH 2019 experiments
./run_experiments_docker.sh ifac20 # run IFAC 2020 experiments
./run_experiments_docker.sh cont # run ARCH 2020 Continuization experiments
./run_experiments_docker.sh abs # run Abstraction / Dynamic Resource Management experiments
./run_experiments_docker.sh --help # compiles everything and then shows help on further options
```
# Contact
If anything does not work or you have questions, please write an email to the mail address linked on that page: https://arxiv.org/abs/1911.02537 

# Running locally (advanced users)

Have a look at the `Dockerfile` for the requirements of the code.

Run via `cd src; python3 -m qronos.run_experiments`.

