# Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing

This is the code for the publication "Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing" (Gaukler et al., 2019/2020; Submitted for publication).

It shares some code with "Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing (Benchmark Proposal)" (Gaukler, Ulbrich 2019, ARCH - Workshop on Applied Verification for Continuous and Hybrid Systems). The code corresponding to that publication is published on https://github.com/qronos-project/arch19-benchmark-iotiming . In the future it is planned to merge the functionality of that codebase into here; meanwhile only the examples are used.

# Running with docker (recommended)

The run_* scripts only require docker, docker-compose, and appropriate permissions (docker group) for your user. No other dependencies are required.

For example, you can use it the following way:

Install Ubuntu 18.04 in a virtual machine.
```
sudo apt install docker.io docker-compose git ssh
sudo adduser $USER docker
git clone https://github.com/qronos-project/timing-stability-lmi
./run_experiments_docker.sh # will take a few hours and write all results into logfile.txt; the table from the paper can be found at the very end.
./run_webserver_docker.sh # Run a webserver with Jupyter Notebook for playing around interactively.
```

# Running locally (advanced users)

Have a look at the `Dockerfile` for the requirements of the code.

Run via `cd src; python3 -m qronos.lis.analyze`.
