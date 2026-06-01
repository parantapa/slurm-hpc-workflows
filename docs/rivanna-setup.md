# How to setup slurm-workflows on Rivanna

## Setup your personal Miniforge environment

For this setup we shall use the Conda package manager.
We shall not be using the conda modules from Rivanna's module system.
They are frequently out of date.

Execute the following commands once you are logged into Rivanna.

Download Miniforge.

```sh
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
```

Install Miniforge. During installation:
* accept license terms
* accept the default installation directory (`$HOME/miniforge3`)
* allow conda to update your shell profile

```sh
bash Miniforge3-Linux-x86_64.sh
```

Execute bash to allow the new shell config to be used.

```sh
exec bash
```

Conda environments can get pretty big.
We shall now configure conda to use `/project/bii_nssac` for environments.
Note, we don't want to use `/scratch` due to rolling 90 day deletion policy.

```sh
mkdir -p "/project/bii_nssac/people/$USER/conda-envs"
mkdir -p "/project/bii_nssac/people/$USER/conda-pkgs-cache"
```

Next create/update your conda config (`$HOME/.condarc`)
Replace `<USERNAME>` below with your Rivanna username.

```
channels:
  - conda-forge
auto_activate_base: false
auto_update_conda: false
add_pip_as_python_dependency: true
envs_dirs:
  - /project/bii_nssac/people/<USERNAME>/conda-envs
pkgs_dirs:
  - /project/bii_nssac/people/<USERNAME>/conda-pkgs-cache
```

Update conda with the new config.

```sh
conda update -n base --all --yes
```

## Setup Cross-Desktop Group (XDG) user directories

Slurm workflows uses the XDG user directories
for storing temporary and runtime files.
On Rivanna we use `/scratch` for this.
Add the following lines at the end of your `$HOME/.bashrc`

```sh
export XDG_CACHE_HOME="/scratch/$USER/var/cache"
export XDG_DATA_HOME="/scratch/$USER/var/data"
export XDG_STATE_HOME="/scratch/$USER/var/state"
export XDG_RUNTIME_DIR="/scratch/$USER/var/runtime"
```

Execute bash to allow the new shell config to be used.

```sh
exec bash
```

Next we create the directories.

```sh
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$XDG_DATA_HOME"
mkdir -p "$XDG_STATE_HOME"
mkdir -p "$XDG_RUNTIME_DIR"
```

## Create a conda environment for slurm-workflows

We shall create a separate environment to install slurm-workflows.
Create and activate a conda environment called `workflow-env`.

```sh
conda create -n workflow-env --yes python=3.12 nodejs=22 nb_conda_kernels
conda activate workflow-env
pip install -v jupyterlab
```

Clone slurm-workflows repository in your home directory and install it.
```sh
cd $HOME
git clone https://github.com/parantapa/slurm-workflows.git
cd slurm-workflows
pip install -ve .
```

## Create default setup script for slurm-workflows

We shall create a setup script for slur-workflows.
This is used to setup environments when they workflow system creates slurm jobs.
Create the file `$HOME/default-env.sh` with the following content.

```sh
module load gcc/14.2.0

CONDA_ENVS_DIR="/project/bii_nssac/people/$USER/conda-envs"
```

## Setup password for Jupyter Lab

Ensure that `workflow-env` is still activated.

```sh
jupyter lab password
```

## Configure Jupyter server to wait longer for restarted kernels

Create the file `$HOME/.jupyter/jupyter_server_config.py` with the following
content.

```python
# give kernels 60 seconds to shutdown cleanly before we terminate them
c.KernelManager.shutdown_wait_time = 60
```

## Set up Foxy Proxy on your local computer to access Jupyter

Install FoxyProxy Standard for your browser.

* [FoxyProxy Standard for Google Chrome](https://chromewebstore.google.com/detail/foxyproxy/gcknhkkoolaabfmlnjonogaaifnjlfnp?pli=1)
* [FoxyProxy Standard for Firefox](https://addons.mozilla.org/en-US/firefox/addon/foxyproxy-standard/)
* [FoxyProxy Standard for Edge](https://microsoftedge.microsoft.com/addons/detail/foxyproxy/flcnoalcefgkhkinjkffipfdhglnpnem)

Download the FoxyProxy configuration for Rivanna.

* [FoxyProxy Configuration for Rivanna](../extra/Rivanna-FoxyProxy_2024-06-06.json)

From your browser's addon section to configure configure FoxyProxy.
From FoxyProxy's settings go to Options tab and import the configuration file.
Close the FoxyProxy settings tab.
Go back to FoxyProxy's settings and ensure that FoxyProxy is configured to enable proxy by patterns.

## Start jupyter from slurm-workflows on Rivanna

Ensure you are logged into Rivanna via ssh.
Also ensure that `workflow-env` is still activated.

```sh
run-jupyter -- -A bii_nssac -p bii --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --mem=0 -t 1-00:00:00
```

Wait for the job to start.
You can monitor it using the `squeue` command.

```sh
squeue -u $USER
```

Once the job has started, note the hostname assigned to the job.
This should be under the NODELIST column of squeue's output.
This should be of the form `udc-XXX-XXX`

## Setup a socks proxy over ssh on your local computer to enable your browser to connect to Rivanna

On Linux and MacOS systems the following command should work.
Run the command in a new terminal.

```sh
ssh -o ExitOnForwardFailure=yes -N -D 127.0.0.1:12001 <username>@rivanna.hpc.virginia.edu
```

Note the program would not return a shell prompt.
This only runs the proxy.
Leave the command running.

## Connect to the Jupyter notebook on Rivanna from your local browser

Use the following url: `http://udc-XXX-XXX:8888`
Replace the hostname part `udc-XXX-XXX` with the actual hostname where
Jupyter notebook is running.

## Stop Jupyter Lab on Rivanna when no longer in use

Once your session is over ensure that you cancel any running Jupyter Lab and
other slurm jobs.

Use `squeue` to see if any are running and use `scancel` to stop those jobs.
