# Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing

This is the code for the following publications:

- "Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing" (Gaukler et al., 2019/2020; IFAC World Congress 2020).
   An extended preprint including details and proofs is available at https://arxiv.org/abs/1911.02537

- "Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing (Benchmark Proposal)" (Gaukler, Ulbrich 2019, ARCH - Workshop on Applied Verification for Continuous and Hybrid Systems).
   The original code corresponding to that publication was published on https://github.com/qronos-project/arch19-benchmark-iotiming . It has been merged into this codebase.

# Running with docker (recommended)

The run_* scripts only require docker, docker-compose, and appropriate permissions (docker group) for your user. No other dependencies are required.

For example, you can use it the following way:

Install Ubuntu 18.04 in a virtual machine.
```
sudo apt install docker.io docker-compose git ssh
sudo adduser $USER docker
git clone --recurse-submodules https://github.com/qronos-project/timing-stability-lmi
./run_experiments_docker.sh arch19 # run ARCH 2019 experiments, write all results into logfile.txt; the table from the paper can be found at the very end.
./run_experiments_docker.sh ifac20 # run IFAC 2020 experiments, write all results into logfile.txt; the table from the paper can be found at the very end.
./run_experiments_docker.sh --help # compiles everything and then shows help on further options
./run_webserver_docker.sh # Run a webserver with Jupyter Notebook for playing around interactively.
```

If anything does not work or you have questions, please write an email or open an issue on GitHub.

# Running locally (advanced users)

Have a look at the `Dockerfile` for the requirements of the code.

Run via `cd src; python3 -m qronos.run_experiments`.

