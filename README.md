# PI-AGenT: Personalised Integrated Atria Generation Tool

We propose a personalized integrated atria generation tool (PI-AGenT) as a highly automated framework which, starting from clinical geometrical data, provides ready-to-use atrial personalized computational models. 
PI-AGenT consists firstly of a pre-processing step applied to the input geometry. Secondly, the atrial orifices are identified and labelled using only one reference point per atrium. If the workflow includes fitting a statistical shape model (SSM) to the input geometry, this is first rigidly aligned with the given mean shape and finally a non-rigid fitting procedure is applied. PI-AGenT automatically generates the fiber orientation using a Laplace-Dirichlet-Rule-based-Method.

![Pipeline](/images/pipeline.png)

## Files and Folders

- **main.py:** main script. Don't forget to re-install carputils with ```pip3 install .```
- **stimulation_LA.par:** parameters file containing all ionic models to assign to each region
- **endo/:** contains the endocardium surface mesh in carp format
- **mesh/:** contains the bilayer mesh in both vtk/carp format
- **results/**: all simulation results will be put here (will be created on first usage)
- **prepace/**: all steady-states of the PSD rotors will be put here (will be created on first usage)
- **run_files/**: contains multiple json files that specify all parameters and ablation primitives for a series of simulations.
- **tissue_ablation_3d.py:** script that performs one job. Direct use is discouraged as there are a lot of paramters
- **tissue_ablation_3d_job_runner.py:** main script that executes a series of jobs as specified by a given json run file (see below)

## Setup

Create a python virtual environment to install the current requirements after installing the requirements of carputils: 
```
python -m venv ~/myEnv
source ~/myEnv/bin/activate
pip install -r requirements.txt

# Go to the carputils folder and re-install carputils' requirements
pip install -r requirements.txt
```
## Usage

Remember to source to myEnv before using the pipeline:
```
source ~/myEnv/bin/activate
```
