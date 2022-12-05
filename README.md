# AugmentA: Patient-specific Augmented Atrial model Generation Tool

We propose a patient-specific Augmented Atrial model Generation Tool (AugmentA) as a highly automated framework which, starting from clinical geometrical data, provides ready-to-use atrial personalized computational models. 
AugmentA consists firstly of a pre-processing step applied to the input geometry. Secondly, the atrial orifices are identified and labelled using only one reference point per atrium. If the workflow includes fitting a statistical shape model (SSM) to the input geometry, this is first rigidly aligned with the given mean shape and finally a non-rigid fitting procedure is applied. AugmentA automatically generates the fiber orientation using a Laplace-Dirichlet-Rule-based-Method.

![Pipeline](/images/pipeline.png)

## Files and Folders

- **main.py:** AugmentA's main script
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
Install [PyMesh](https://pymesh.readthedocs.io/en/latest/installation.html)

Go to the carputils folder and re-install carputils' requirements (assuming that carputils was installed in the home folder):
```
cd ~/carputils
pip install -r requirements.txt
```
## Usage

Remember to source to myEnv before using the pipeline:
```
source ~/myEnv/bin/activate
```
Show all options:
```
python main.py --help
```
Example using an MRI segmentation to produce a bilayer atrial model:
```
python main.py --mesh mesh/LA_MRI.vtp --closed_surface 0 --use_curvature_to_open 1 --atrium LA --open_orifices 1 --MRI 1
```
Example opening the atrial orifices using the surface curvature to identify the veins of a closed geometry, it expects the valve region to be tagged on the atrial surface with a value > 0.5 (see the scalar "valve" in mesh/LA_EAM.vtp):
```
python main.py --mesh mesh/LA_EAM.vtp --open_orifices 1 --MRI 0
```
Example manually opening the atrial orifices of a closed geometry:
```
python main.py --mesh mesh/LA_EAM.vtp --open_orifices 1 --MRI 0 --use_curvature_to_open 0
```
Example using a closed surface derived from a MRI segmentation to produce a volumetric atrial model:
```
python main.py --mesh mesh/mwk05_bi.vtp --closed_surface 1 --use_curvature_to_open 0 --atrium LA_RA
```
## Q&A

- Selection of appendage apex: the selected point will be used as boundary condition for a Laplacian problem. Therefore, the point at the center of the appendage is the most suitable to identify the whole appendage body
- Fiber_LA: LAA labeling (check LPVs identification functions distinguish_Pvs and optimize_PVs in la_generate_fiber.py)
- Fiber_RA: PMs (check step in function Method.downsample_path in ra_generate_fiber.py), bridges (boolean operations and normal directions of original mesh)

## Citation
When using this work, please cite
> *AugmentA: Patient-specific Augmented Atrial model Generation Tool*
>
> Luca Azzolin, Martin Eichenlaub, Claudia Nagel, Deborah Nairn, Jorge Sánchez, Laura Unger, Olaf Dössel, Amir Jadidi, Axel Loewe
> [doi:10.1101/2022.02.13.22270835](https://doi.org/10.1101/2022.02.13.22270835)


## License

All source code is subject to the terms of the Academic Public License.
Copyright 2021 Luca Azzolin, Karlsruhe Institute of Technology.

## Contact

Luca Azzolin  
Institute of Biomedical Engineering  
Karlsruhe Institute of Technology  
www.ibt.kit.edu
