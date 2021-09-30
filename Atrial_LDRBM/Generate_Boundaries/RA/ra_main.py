import os
import subprocess as sp
import datetime
from ra_ring_center_points import run as ring_center_points
from ra_extract_epi_endo import run as extract_epi_endo
from ra_extract_rings import run as extract_rings
from ra_generate_mesh import run as generate_mesh
from ra_separate_rings import run as separate_rings
from ra_generate_surf_id import run as generate_surf_id
from ra_gamma_top import run as gamma_top

def run():
    # example:
    appen_point = [2.4696, -78.0776, 43.5639]

    start_time = datetime.datetime.now()
    print('[Step 1] Detecting the rings... ' + str(start_time))
    ring_center_points()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 1] Detecting the rings...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 2] Using faraday.m to calculate the contour... ' + str(start_time))
    # Ideal way: but not there is not matlab.engine in library
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # eng.faraday(nargout=0)
    # eng.quit()
    current_path = os.getcwd()
    command = """/Applications/MATLAB_R2020b.app/Contents/MacOS/MATLAB_maci64  -r "cd(fullfile('"""+current_path+"""/MATLAB')), faradatryum" """
    #print(current_path)
    #print(command)
    print('MATLAB program running...')
    sh=sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    sh.wait()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 2] Using faraday.m to calculate the contour...done! ' + str(end_time) + '\nRunning time: ' + str(
        running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 3] Extracting Epi, Endo and rings... ' + str(start_time))
    extract_epi_endo()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 3] Extracting Epi, Endo and rings...done! ' + str(end_time) + '\nRunning time: ' + str(
        running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 4] Extracting rings... ' + str(start_time))
    extract_rings()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 4] Extracting rings...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 5] Separating rings... ' + str(start_time))
    separate_rings(appen_point)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 5] Separating rings...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 6] Separating gamma top... ' + str(start_time))
    gamma_top()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 6] Separating gamma top...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 7] Generating surface ids... ' + str(start_time))
    generate_surf_id(appen_point)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 7] Generating surface ids...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 8] Generating the mesh... ' + str(start_time))
    generate_mesh()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 8] Generating the mesh...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')


if __name__ == '__main__':
    run()
