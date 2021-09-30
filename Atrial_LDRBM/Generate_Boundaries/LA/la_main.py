import os
import subprocess as sp
import datetime
import la_ring_center_points
import la_extract_epi_endo
import la_extract_rings 
import la_generate_mesh
import la_separate_rings 
import la_generate_surf_id 
import argparse

parser = argparse.ArgumentParser(description='Generate boundaries.')
parser.add_argument('--mesh',
                    type=str,
                    default='LA',
                    help='path to meshname')
parser.add_argument('-ap', action='append',
                    type=float,
                    dest='apex_point',
                    default=[],
                    help='Add repeated values to a list')

args = parser.parse_args()

def jobID(args):
    """
    Generate name of top level output directory.
    """
    today = date.today()
    return '{}_generate_bd_{}'.format(today.isoformat(), args.mesh)

def run(args):
    # example :
    appen_point = [22.561380859375, -35.339421875, 34.9769375]
    
    start_time = datetime.datetime.now()
    print('[Step 1] Detecting the rings... ' + str(start_time))
    ring_center_points()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 1] Detecting the rings...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 2] Using faraday.m to calculate the contour... ' + str(start_time))
    # Ideal way: but there is not matlab.engine in library of IBT computer
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # eng.faraday(nargout=0)
    # eng.quit()
    current_path = os.getcwd()
    command = """/Applications/MATLAB_R2020b.app/Contents/MacOS/MATLAB_maci64  -r "cd(fullfile('"""+current_path+"""/MATLAB')), faradatryum.m" """
    #print(current_path)
    #print(command)
    print('MATLAB program running...')
    sh=sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    sh.wait()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 2] Using faraday.m to calculate the contour...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 3] Extracting Epi, Endo and rings... ' + str(start_time))
    extract_epi_endo()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 3] Extracting Epi, Endo and rings...done! ' + str(end_time) + '\nRunning time: ' + str(
        running_time) + '\n')
    # if you separated Epi- and Endocarium surface. please check README.md
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
    print('[Step 6] Generating surface ids... ' + str(start_time))
    generate_surf_id(appen_point)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 6] Generating surface ids...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 7] Generating the mesh... ' + str(start_time))
    generate_mesh()
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 7] Generating the mesh...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')


if __name__ == '__main__':
    run()
