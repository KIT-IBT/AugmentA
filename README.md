# PI-AGenT: Personalised Integrated Atria Generation Tool

We propose a personalized integrated atria generation tool (PI-AGenT) as a highly automated framework which, starting from clinical geometrical data, provides ready-to-use atrial personalized computational models. 
PI-AGenT consists firstly of a pre-processing step applied to the input geometry. Secondly, the atrial orifices are identified and labelled using only one reference point per atrium. If the workflow includes fitting a statistical shape model (SSM) to the input geometry, this is first rigidly aligned with the given mean shape and finally a non-rigid fitting procedure is applied. PI-AGenT automatically generates the fiber orientation using a Laplace-Dirichlet-Rule-based-Method.

![Pipeline](/images/pipeline.png)

## Files and Folders

- **main.py:** PI-AGenT's main script
- **mesh/:** contains the exemplary mesh and the statistical shape model
- **standalones/:** standalone tools used in the pipeline
- **template/**: template for non-rigid fitting process
- **Atrial_LDRBM/**: Laplace-Dirichlet-Rule-based-Method to annotate anatomical regions and generate atrial fiber orientation in the atria

## Setup

Create a python virtual environment to install the current requirements after installing the requirements of carputils: 
```
python -m venv ~/myEnv
source ~/myEnv/bin/activate
pip install -r requirements.txt

```
Go to the carputils folder and re-install carputils' requirements (assuming that carputils was installed in the home folder):
```
cd ~/carputils
pip install -r requirements.txt
```
## Usage

Remember to source to myEnv before using the pipeline:
```
source ~/myEnv/bin/activate
python main.py --MRI 1 --mesh mesh/LA_map.vtk
```
Show all options:
```
source ~/myEnv/bin/activate
python main.py --help
```