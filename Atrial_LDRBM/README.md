# AtrialLDRBM

Create a python virtual environment to install the current requirements after installing the requirements of carputils:
python -m venv myEnv
source myEnv/bin/activate
pip install requirements.txt
##Left Atrium

### [step1]Generate Boundaries
Identifies and labels atrial openings. Run the file Generate_Boundaries/extract_rings.py

### [step2]Generate Fibers
Warning: the algorithm expects endocardium mesh with point normals pointing inside!!!
after generating boundaries
1. input the coordinate of appendage in LDRBM/Fiber_LA/la_main.py
2. run LDRBM/Fiber_LA/la_main.py

##Right Atrium
same




## Start form blood pool
if you have already have separated Epi- and Endocarium surface.
you can:

1.save the epicardium surface in: result/la_epi_surface.vtk
2.save the endocardium surface in: result/la_endo_surface.vtk
3.save the closed the surface(endo+epi) in: model/LA.vtk
4.comment the first 3 steps in Generate_Boundaries/LA/la_main.py (skip the separating step)
5.do the steps above
same for right atrium