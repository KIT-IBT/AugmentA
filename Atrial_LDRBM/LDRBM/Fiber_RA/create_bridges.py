#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin

Copyright 2021 Luca Azzolin

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.  
"""
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy
import datetime
import Methods_RA as Method
import csv
import os
import subprocess
import pymesh
import pymeshlab
import pickle
from numpy.linalg import norm

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

def add_free_bridge(args, la_epi, ra_epi, CS_p, df, job):

    ########################################
    with open(os.path.join(EXAMPLE_DIR,'../../element_tag.csv')) as f:
        tag_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            tag_dict[row['name']] = row['tag']
    
    bridge_radius = 1.65*args.scale
    
    bachmann_bundel_left = int(tag_dict['bachmann_bundel_left'])      
    bachmann_bundel_right = int(tag_dict['bachmann_bundel_right'])
    bachmann_bundel_internal = int(tag_dict['bachmann_bundel_internal'])
    middle_posterior_bridge_left = int(tag_dict['middle_posterior_bridge_left'])
    middle_posterior_bridge_right = int(tag_dict['middle_posterior_bridge_right'])
    upper_posterior_bridge_left = int(tag_dict['upper_posterior_bridge_left'])
    upper_posterior_bridge_right = int(tag_dict['upper_posterior_bridge_right'])
    coronary_sinus_bridge_left = int(tag_dict['coronary_sinus_bridge_left'])
    coronary_sinus_bridge_right = int(tag_dict['coronary_sinus_bridge_right'])
    right_atrial_septum_epi = int(tag_dict['right_atrial_septum_epi'])
    left_atrial_wall_epi = int(tag_dict["left_atrial_wall_epi"])
    mitral_valve_epi = int(tag_dict["mitral_valve_epi"])
    tricuspid_valve_epi = int(tag_dict["tricuspid_valve_epi"])
    
    #la_epi = Method.vtk_thr(la, 0 "CELLS", "elemTag", left_atrial_wall_epi)
    geo_filter_la = vtk.vtkGeometryFilter()
    geo_filter_la.SetInputData(la_epi)
    geo_filter_la.Update()
    la_epi_surface = geo_filter_la.GetOutput()
    
    #ra_epi = Method.vtk_thr(la, 0 "CELLS", "elemTag", left_atrial_wall_epi)

    geo_filter_ra = vtk.vtkGeometryFilter()
    geo_filter_ra.SetInputData(ra_epi)
    geo_filter_ra.Update()
    ra_epi_surface = geo_filter_ra.GetOutput()
    
    SVC_p = np.array(df["SVC"])
    IVC_p = np.array(df["IVC"])
    TV_p = np.array(df["TV"])
    
    ra_septum = Method.vtk_thr(ra_epi, 2, "CELLS", "elemTag", right_atrial_septum_epi,right_atrial_septum_epi)
    la_wall = Method.vtk_thr(la_epi, 2, "CELLS", "elemTag", left_atrial_wall_epi,left_atrial_wall_epi)
    mv_la = Method.vtk_thr(la_epi, 2, "CELLS", "elemTag", mitral_valve_epi,mitral_valve_epi)
    tv_ra = Method.vtk_thr(ra_epi, 2, "CELLS", "elemTag", tricuspid_valve_epi,tricuspid_valve_epi)
    
    # Find middle and upper posterior bridges points
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(ra_septum)
    loc.BuildLocator()
    point_septum_SVC = ra_septum.GetPoint(loc.FindClosestPoint(SVC_p))
    point_septum_IVC = ra_septum.GetPoint(loc.FindClosestPoint(IVC_p))
    
    SVC_IVC_septum_path = Method.dijkstra_path_coord(ra_epi_surface, point_septum_SVC, point_septum_IVC)
    
    middle_posterior_bridge_point = SVC_IVC_septum_path[int(len(SVC_IVC_septum_path)*0.6),:]
    
    upper_posterior_bridge_point = SVC_IVC_septum_path[int(len(SVC_IVC_septum_path)*0.4),:]
    
    mpb_tube, mpb_sphere_1, mpb_sphere_2, mpb_fiber = Method.create_free_bridge_semi_auto(la_epi_surface, ra_epi_surface, middle_posterior_bridge_point, bridge_radius)
    Method.smart_bridge_writer(mpb_tube, mpb_sphere_1, mpb_sphere_2, "middle_posterior_bridge", job)
    
    upb_tube, upb_sphere_1, upb_sphere_2, upb_fiber = Method.create_free_bridge_semi_auto(la_epi_surface, ra_epi_surface, upper_posterior_bridge_point, bridge_radius)
    Method.smart_bridge_writer(upb_tube, upb_sphere_1, upb_sphere_2, "upper_posterior_bridge", job)

    # Coronary sinus bridge point

    loc = vtk.vtkPointLocator() # it happened that the point is too close to the edge and the heart is not found
    loc.SetDataSet(mv_la)
    loc.BuildLocator()
    point_CS_on_MV = mv_la.GetPoint(loc.FindClosestPoint(CS_p+TV_p*0.1))

    #loc = vtk.vtkPointLocator()
    #loc.SetDataSet(la_wall)
    #loc.BuildLocator()
    #point_CS_on_MV = la_wall.GetPoint(loc.FindClosestPoint(CS_p+TV_p*0.1))
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(ra_septum)
    loc.BuildLocator()
    point_CS_bridge = ra_septum.GetPoint(loc.FindClosestPoint(point_CS_on_MV))
    
    csb_tube, csb_sphere_1, csb_sphere_2, csb_fiber = Method.create_free_bridge_semi_auto(la_epi_surface, ra_epi_surface, point_CS_bridge, bridge_radius)
    Method.smart_bridge_writer(csb_tube, csb_sphere_1, csb_sphere_2, "coronary_sinus_bridge", job)
    
    if args.mesh_type == "vol":
        append_filter = vtk.vtkAppendFilter()
        append_filter.AddInputData(la_epi)
        append_filter.AddInputData(ra_epi)
        append_filter.Update()

        tag = np.zeros((append_filter.GetOutput().GetNumberOfCells(),),dtype=int)
        tag[:la_epi.GetNumberOfCells()] = vtk.util.numpy_support.vtk_to_numpy(la_epi.GetCellData().GetArray('elemTag'))
        tag[la_epi.GetNumberOfCells():] = vtk.util.numpy_support.vtk_to_numpy(ra_epi.GetCellData().GetArray('elemTag'))

        meshNew = dsa.WrapDataObject(append_filter.GetOutput())
        meshNew.CellData.append(tag, "elemTag")
        append_filter = vtk.vtkAppendFilter()
        append_filter.AddInputData(meshNew.VTKObject)
        append_filter.Update()
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/la_ra_res.vtu")
        writer.SetInputData(append_filter.GetOutput())
        writer.Write()
    elif args.mesh_type == "bilayer":

        la_e = Method.smart_reader(job.ID + "/result_LA/LA_epi_with_fiber.vtu")
        geo_filter_la_epi = vtk.vtkGeometryFilter()
        geo_filter_la_epi.SetInputData(la_e)
        geo_filter_la_epi.Update()
        la_e = geo_filter_la_epi.GetOutput()

        ra_e = Method.smart_reader(job.ID + "/result_RA/RA_epi_with_fiber.vtu")
        geo_filter_ra_epi = vtk.vtkGeometryFilter()
        geo_filter_ra_epi.SetInputData(ra_e)
        geo_filter_ra_epi.Update()
        ra_e = geo_filter_ra_epi.GetOutput()

        append_filter = vtk.vtkAppendFilter()
        append_filter.AddInputData(la_e)
        append_filter.AddInputData(ra_e)
        append_filter.Update() # la_ra_usg

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID + "/result_RA/LA_epi_RA_epi_with_tag.vtu") # Good till here!
        writer.SetInputData(append_filter.GetOutput())
        writer.Write()

        # append_filter = vtk.vtkAppendFilter() # same as above but names changed  to avoid overwriting
        # append_filter.AddInputData(la_epi)
        # append_filter.AddInputData(ra_epi)
        # append_filter.Update()
        #
        # geo_filter = vtk.vtkGeometryFilter()
        # geo_filter.SetInputData(append_filter.GetOutput())
        # geo_filter.Update()
        # writer = vtk.vtkOBJWriter()
        # writer.SetFileName(job.ID+"/result_RA/la_ra_res.obj")
        # writer.SetInputData(geo_filter.GetOutput())
        # writer.Write()

    bridge_list = ['BB_intern_bridges', 'coronary_sinus_bridge', 'middle_posterior_bridge', 'upper_posterior_bridge']
    for var in bridge_list:

        mesh_A = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_tube.obj")
        mesh_B = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_sphere_1.obj")
        mesh_C = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_sphere_2.obj")

        output_mesh_1 = pymesh.boolean(mesh_A, mesh_B, operation="union", engine="igl")

        output_mesh = pymesh.boolean(output_mesh_1, mesh_C, operation="union", engine="igl")

        m = pymeshlab.Mesh(output_mesh.vertices, output_mesh.faces)
        # create a new MeshSet
        ms = pymeshlab.MeshSet()
        # add the mesh to the MeshSet
        ms.add_mesh(m, "bridge_mesh")
        # apply filter
        ms.remeshing_isotropic_explicit_remeshing(iterations=5, targetlen=0.4*args.scale, adaptive=True)
        ms.save_current_mesh(job.ID+"/bridges/"+str(var)+"_bridge_resampled.obj",\
        save_vertex_color=False, save_vertex_normal=False, save_face_color=False, save_wedge_texcoord=False, save_wedge_normal=False)

        subprocess.run(["meshtool",
                    "generate",
                    "mesh",
                    "-ofmt=vtk",
                    "-prsv_bdry=1",
                    "-surf="+job.ID+"/bridges/"+str(var)+"_bridge_resampled.obj",
                    "-outmsh="+job.ID+"/bridges/"+str(var)+"_bridge_resampled.vtk"])

    # if args.mesh_type == "vol":

    #     la_ra_usg = append_filter.GetOutput()
    #     print('reading done!')

    #     bridge_list = ['BB_intern_bridges', 'coronary_sinus_bridge', 'middle_posterior_bridge', 'upper_posterior_bridge']
    #     earth_cell_ids_list = []
    #     for var in bridge_list:
    #         reader = vtk.vtkUnstructuredGridReader()
    #         reader.SetFileName(job.ID+"/bridges/"+str(var)+'_bridge_resampled.vtk')
    #         reader.Update()
    #         bridge_usg = reader.GetOutput()

    #         geo_filter = vtk.vtkGeometryFilter()
    #         geo_filter.SetInputData(bridge_usg)
    #         geo_filter.Update()
    #         bridge = geo_filter.GetOutput()

    #         locator = vtk.vtkStaticPointLocator()
    #         locator.SetDataSet(la_ra_usg)
    #         locator.BuildLocator()

    #         intersection_points = bridge_usg.GetPoints().GetData()
    #         intersection_points = vtk.util.numpy_support.vtk_to_numpy(intersection_points)

    #         earth_point_ids_temp = vtk.vtkIdList()
    #         earth_point_ids = vtk.vtkIdList()
    #         for i in range(len(intersection_points)):
    #             locator.FindPointsWithinRadius(0.7*args.scale, intersection_points[i], earth_point_ids_temp)
    #             for j in range(earth_point_ids_temp.GetNumberOfIds()):
    #                 earth_point_ids.InsertNextId(earth_point_ids_temp.GetId(j))

    #         earth_cell_ids_temp = vtk.vtkIdList()
    #         earth_cell_ids = vtk.vtkIdList()
    #         for i in range(earth_point_ids.GetNumberOfIds()):
    #             la_ra_usg.GetPointCells(earth_point_ids.GetId(i),earth_cell_ids_temp)
    #             for j in range(earth_cell_ids_temp.GetNumberOfIds()):
    #                 earth_cell_ids.InsertNextId(earth_cell_ids_temp.GetId(j))
    #                 earth_cell_ids_list += [earth_cell_ids_temp.GetId(j)]
    #         extract = vtk.vtkExtractCells()
    #         extract.SetInputData(la_ra_usg)
    #         extract.SetCellList(earth_cell_ids)
    #         extract.Update()

    #         geo_filter = vtk.vtkGeometryFilter()
    #         geo_filter.SetInputData(extract.GetOutput())
    #         geo_filter.Update()
    #         earth = geo_filter.GetOutput()

    #         cleaner = vtk.vtkCleanPolyData()
    #         cleaner.SetInputData(earth)
    #         cleaner.Update()

    #         # meshNew = dsa.WrapDataObject(cleaner.GetOutput())
    #         writer = vtk.vtkOBJWriter()
    #         writer.SetFileName(job.ID+"/bridges/"+str(var)+"_earth.obj")
    #         writer.SetInputData(cleaner.GetOutput())
    #         writer.Write()

    #     print("Extracted earth")
    #     cell_id_all = []
    #     for i in range(la_ra_usg.GetNumberOfCells()):
    #         cell_id_all.append(i)

    #     la_diff =  list(set(cell_id_all).difference(set(earth_cell_ids_list)))
    #     la_ra_new = vtk.vtkIdList()
    #     for var in la_diff:
    #         la_ra_new.InsertNextId(var)

    #     extract = vtk.vtkExtractCells()
    #     extract.SetInputData(la_ra_usg)
    #     extract.SetCellList(la_ra_new)
    #     extract.Update()

    #     append_filter = vtk.vtkAppendFilter()
    #     append_filter.MergePointsOn()
    #     #append_filter.SetTolerance(0.01*args.scale)
    #     append_filter.AddInputData(extract.GetOutput())

    # elif args.mesh_type == "bilayer":

    # if args.mesh_type == "bilayer":
    #     la_ra_usg = append_filter.GetOutput()
    # else:
    #     la_ra_usg_vol = append_filter.GetOutput()
    #     ra_epi = Method.vtk_thr(append_filter.GetOutput(), 2, "CELLS", "elemTag", 11,18)
    #     ra_BB = Method.vtk_thr(append_filter.GetOutput(), 2, "CELLS", "elemTag", bachmann_bundel_right,bachmann_bundel_right)
    #     la_epi = Method.vtk_thr(append_filter.GetOutput(), 2, "CELLS", "elemTag", 61,70)
    #     la_BB = Method.vtk_thr(append_filter.GetOutput(), 2, "CELLS", "elemTag", bachmann_bundel_left,bachmann_bundel_left)

    #     append_filter = vtk.vtkAppendFilter()
    #     append_filter.AddInputData(la_epi)
    #     append_filter.AddInputData(ra_epi)
    #     append_filter.AddInputData(la_BB)
    #     append_filter.AddInputData(ra_BB)
    #     append_filter.Update()
    la_ra_usg = append_filter.GetOutput() # this has already elemTag

    if args.debug:
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID + "/result_RA/la_ra_usg.vtu")
        writer.SetInputData(la_ra_usg)
        writer.Write()

    print('reading done!')

    bridge_list = ['BB_intern_bridges', 'coronary_sinus_bridge', 'middle_posterior_bridge', 'upper_posterior_bridge']
    earth_cell_ids_list = []
    for var in bridge_list:
        print(var)
    
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(job.ID+"/bridges/"+str(var)+'_bridge_resampled.vtk')
        reader.Update()
        bridge_usg = reader.GetOutput()


        # geo_filter = vtk.vtkGeometryFilter()
        # geo_filter.SetInputData(bridge_usg)
        # geo_filter.Update()
        # bridge = geo_filter.GetOutput()
        
        # reverse = vtk.vtkReverseSense()
        # reverse.ReverseCellsOn()
        # reverse.ReverseNormalsOn()
        # reverse.SetInputConnection(cleaner.GetOutputPort())
        # reverse.Update()
        
        # earth = reverse.GetOutput()
        
        # vbool = vtk.vtkBooleanOperationPolyDataFilter()
        # vbool.SetOperationToDifference()
        # vbool.SetInputData( 0, epi_surf )
        # vbool.SetInputData( 1, bridge )
            
        # vbool.Update()

        locator = vtk.vtkStaticPointLocator()
        locator.SetDataSet(la_ra_usg)
        locator.BuildLocator()
        
        #intersection_points = vbool.GetOutput().GetPoints().GetData()
        intersection_points = bridge_usg.GetPoints().GetData()
        intersection_points = vtk.util.numpy_support.vtk_to_numpy(intersection_points)
        
        earth_point_ids_temp = vtk.vtkIdList()
        earth_point_ids = vtk.vtkIdList()
        for i in range(len(intersection_points)):
            locator.FindPointsWithinRadius(0.7*args.scale, intersection_points[i], earth_point_ids_temp)
            for j in range(earth_point_ids_temp.GetNumberOfIds()):
                earth_point_ids.InsertNextId(earth_point_ids_temp.GetId(j))

        earth_cell_ids_temp = vtk.vtkIdList()
        earth_cell_ids = vtk.vtkIdList()
        for i in range(earth_point_ids.GetNumberOfIds()):
            la_ra_usg.GetPointCells(earth_point_ids.GetId(i),earth_cell_ids_temp)
            for j in range(earth_cell_ids_temp.GetNumberOfIds()):
                earth_cell_ids.InsertNextId(earth_cell_ids_temp.GetId(j))
                earth_cell_ids_list += [earth_cell_ids_temp.GetId(j)]
        extract = vtk.vtkExtractCells()
        extract.SetInputData(la_ra_usg)
        extract.SetCellList(earth_cell_ids)
        extract.Update()

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(extract.GetOutput())
        geo_filter.Update()
        earth = geo_filter.GetOutput()
        
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(earth)
        cleaner.Update()
        earth = cleaner.GetOutput()
        
        # meshNew = dsa.WrapDataObject(cleaner.GetOutput())
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(job.ID+"/bridges/"+str(var)+"_earth.obj")
        writer.SetInputData(earth)
        writer.Write()
        
        # Here

        print("Extracted earth")
        cell_id_all = []
        for i in range(la_ra_usg.GetNumberOfCells()):
            cell_id_all.append(i)
        
        la_diff =list(set(cell_id_all).difference(set(earth_cell_ids_list)))
        la_ra_new = vtk.vtkIdList()
        for item in la_diff:
            la_ra_new.InsertNextId(item)
            
        extract = vtk.vtkExtractCells()
        extract.SetInputData(la_ra_usg)
        extract.SetCellList(la_ra_new)
        extract.Update()

        append_filter = vtk.vtkAppendFilter()
        append_filter.MergePointsOn()
        #append_filter.SetTolerance(0.01*args.scale)

        append_filter.AddInputData(extract.GetOutput())
        append_filter.Update() # added
        la_ra_epi = append_filter.GetOutput()  # we lose this mesh, when defining the append filter later

        if args.debug and var =='upper_posterior_bridge':
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID + "/result_RA/LA_RA_epi_with_holes.vtu") # Still has element Tag
            writer.SetInputData(la_ra_epi)
            writer.Write()

    # Now extract earth from LA_endo as well

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(job.ID + "/result_LA/LA_endo_with_fiber.vtu")  # Has elemTag! :)
    reader.Update()
    la_endo = reader.GetOutput()

    for var in bridge_list:
        print(var)

        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(job.ID + "/bridges/" + str(var) + '_bridge_resampled.vtk')
        reader.Update()
        bridge_usg = reader.GetOutput()

        locator = vtk.vtkStaticPointLocator()
        locator.SetDataSet(la_endo)
        locator.BuildLocator()

        intersection_points = bridge_usg.GetPoints().GetData()
        intersection_points = vtk.util.numpy_support.vtk_to_numpy(intersection_points)

        earth_point_ids_temp = vtk.vtkIdList()
        earth_point_ids = vtk.vtkIdList()
        for i in range(len(intersection_points)):
            locator.FindPointsWithinRadius(0.7 * args.scale, intersection_points[i], earth_point_ids_temp)
            for j in range(earth_point_ids_temp.GetNumberOfIds()):
                earth_point_ids.InsertNextId(earth_point_ids_temp.GetId(j))

        earth_cell_ids_temp = vtk.vtkIdList()
        earth_cell_ids = vtk.vtkIdList()
        for i in range(earth_point_ids.GetNumberOfIds()):
            la_endo.GetPointCells(earth_point_ids.GetId(i), earth_cell_ids_temp)
            for j in range(earth_cell_ids_temp.GetNumberOfIds()):
                earth_cell_ids.InsertNextId(earth_cell_ids_temp.GetId(j))
                earth_cell_ids_list += [earth_cell_ids_temp.GetId(j)]
        extract = vtk.vtkExtractCells()
        extract.SetInputData(la_endo)
        extract.SetCellList(earth_cell_ids)
        extract.Update()

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(extract.GetOutput())
        geo_filter.Update()
        earth = geo_filter.GetOutput()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(earth)
        cleaner.Update()
        earth = cleaner.GetOutput()

        print("Extracted earth")
        cell_id_all = []
        for i in range(la_endo.GetNumberOfCells()):
            cell_id_all.append(i)

        la_diff = list(set(cell_id_all).difference(set(earth_cell_ids_list)))
        la_endo_new = vtk.vtkIdList()
        for item in la_diff:
            la_endo_new.InsertNextId(item)

        extract = vtk.vtkExtractCells()
        extract.SetInputData(la_endo)
        extract.SetCellList(la_endo_new)
        extract.Update()

        append_filter = vtk.vtkAppendFilter()
        append_filter.MergePointsOn()
        # append_filter.SetTolerance(0.01*args.scale)

        append_filter.AddInputData(extract.GetOutput())
        append_filter.Update()  # added
        la_endo = append_filter.GetOutput()  # we lose this mesh, when defining the append filter later

        if args.debug and var == 'upper_posterior_bridge':
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID + "/result_RA/LA_endo_with_holes.vtu")  # Still has element Tag
            writer.SetInputData(la_endo)
            writer.Write()

    #     append_filter = vtk.vtkAppendFilter()
    #     append_filter.MergePointsOn()
    #     append_filter.SetTolerance(0.2*args.scale)
    #     append_filter.AddInputData(append_filter.GetOutput())
    # meshNew = dsa.WrapDataObject(extract.GetOutput())

    filename = job.ID+'/bridges/bb_fiber.dat'
    f = open(filename, 'rb')
    bb_fiber = pickle.load(f)
    
    spline_points = vtk.vtkPoints()
    for i in range(len(bb_fiber)):
        spline_points.InsertPoint(i, bb_fiber[i][0], bb_fiber[i][1], bb_fiber[i][2])
    
    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)
    
    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
    bb_fiber_points_data = vtk.util.numpy_support.vtk_to_numpy(functionSource.GetOutput().GetPoints().GetData())
    
    print("Union between earth and bridges")
    for var in bridge_list:
        
        # if args.mesh_type == "vol":
        #     mesh_D = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_bridge_resampled.obj")
        #     mesh_E = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_earth.obj")
        #     output_mesh_2 = pymesh.boolean(mesh_D, mesh_E, operation="union", engine="igl")
        # elif args.mesh_type == "bilayer":
        # Here
        mesh_D = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_bridge_resampled.obj")
        mesh_E = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_earth.obj")
        # # Warning: set -1 if pts normals are pointing outside
        # # Use union if the endo normals are pointing outside
        #output_mesh_2 = pymesh.boolean(mesh_D, mesh_E, operation="union", engine="corefinement")
        if args.mesh_type=="bilayer":
            # Use difference if the endo normals are pointing inside
            output_mesh_2 = pymesh.boolean(mesh_E, mesh_D, operation="difference", engine="corefinement")
            pymesh.save_mesh(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj", output_mesh_2, ascii=True)
        else:
            output_mesh_2 = pymesh.boolean(mesh_D, mesh_E, operation="union", engine="corefinement")
            pymesh.save_mesh(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj", output_mesh_2, ascii=True)
            
            # reader = vtk.vtkOBJReader()
            # reader.SetFileName(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj")
            # reader.Update()

            # output_mesh_2 = Method.extract_largest_region(reader.GetOutput())

            # writer = vtk.vtkOBJWriter()
            # writer.SetFileName(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj")
            # writer.SetInputData(output_mesh_2)
            # writer.Write()

            # mesh_D = pymesh.load_mesh(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj")

            # output_mesh_2 = pymesh.boolean(mesh_D, mesh_E, operation="union", engine="corefinement")

            # pymesh.save_mesh(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj", output_mesh_2, ascii=True)

        print("Union between earth and bridges in "+var)
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(job.ID+"/bridges/"+str(var)+"_union_to_resample.obj")
        ms.remeshing_isotropic_explicit_remeshing(iterations=5, targetlen=0.4*args.scale, adaptive=True)
        ms.save_current_mesh(job.ID+"/bridges/"+str(var)+"_union.obj",save_vertex_color=False, save_vertex_normal=False, save_face_color=False, save_wedge_texcoord=False, save_wedge_normal=False)
    
        if args.mesh_type == "vol":
            subprocess.run(["meshtool", 
                        "generate", 
                        "mesh", 
                        "-ofmt=vtk",
                        "-prsv_bdry=1",
                        #"-scale={}".format(0.4*args.scale),
                        "-surf="+job.ID+"/bridges/"+str(var)+"_union.obj",
                        "-outmsh="+job.ID+"/bridges/"+str(var)+"_union_mesh.vtk"])
        
            reader = vtk.vtkUnstructuredGridReader() #vtkXMLUnstructuredGridReader
            reader.SetFileName(job.ID+"/bridges/"+str(var)+"_union_mesh.vtk")
            reader.Update()
            bridge_union = reader.GetOutput()
        elif args.mesh_type == "bilayer":

            reader = vtk.vtkOBJReader()
            reader.SetFileName(job.ID+"/bridges/"+str(var)+"_union.obj")
            reader.Update() 
            bridge_union = vtk.vtkUnstructuredGrid()
            bridge_union.DeepCopy(reader.GetOutput())

        tag = np.zeros(bridge_union.GetNumberOfCells(), dtype="int")

        if var == 'BB_intern_bridges':
            tag[:] = bachmann_bundel_internal
        elif var == 'coronary_sinus_bridge':
            tag[:] = coronary_sinus_bridge_left
        elif var == 'middle_posterior_bridge':
            tag[:] = middle_posterior_bridge_left
        elif var =='upper_posterior_bridge':
            tag[:] = upper_posterior_bridge_left

        fiber = np.ones((bridge_union.GetNumberOfCells(), 3), dtype="float32")

        if var == 'BB_intern_bridges':
            fiber = Method.assign_element_fiber_around_path_within_radius(bridge_union, bb_fiber_points_data, 3*args.scale, fiber, smooth=True)
        elif var == 'coronary_sinus_bridge':
            fiber[:] = csb_fiber
        elif var == 'middle_posterior_bridge':
            fiber[:] = mpb_fiber
        elif var =='upper_posterior_bridge':
            fiber[:] = upb_fiber

        meshNew = dsa.WrapDataObject(bridge_union)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(fiber, "fiber")

        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/bridges/"+str(var)+"_union_mesh.vtk") # we have the elemTags here
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()

        if var == 'BB_intern_bridges':
            bb = meshNew.VTKObject
        elif var == 'coronary_sinus_bridge':
            cs = meshNew.VTKObject
        elif var == 'middle_posterior_bridge':
            mp = meshNew.VTKObject
        elif var =='upper_posterior_bridge':
            up = meshNew.VTKObject

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(job.ID + "/bridges/BB_intern_bridges_union_mesh.vtk")
    reader.Update()
    bb = reader.GetOutput()

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(job.ID + "/bridges/coronary_sinus_bridge_union_mesh.vtk")
    reader.Update()
    cs = reader.GetOutput()

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(job.ID + "/bridges/middle_posterior_bridge_union_mesh.vtk")
    reader.Update()
    mp = reader.GetOutput()

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(job.ID + "/bridges/upper_posterior_bridge_union_mesh.vtk")
    reader.Update()
    up = reader.GetOutput()

    append_filter = vtk.vtkAppendFilter()
    append_filter.AddInputData(bb)
    append_filter.AddInputData(cs)
    append_filter.AddInputData(mp)
    append_filter.AddInputData(up)
    append_filter.Update()
    bridges = append_filter.GetOutput()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(job.ID + "/result_RA/append_bridges_2.vtu") # Has elementTag! :)
    writer.SetInputData(bridges)
    writer.Write()

        # # Append all the bridges first
        # if var == bridge_list[0]: # just define append_filter in the first iteration
        #     append_filter = vtk.vtkAppendFilter()
        #     append_filter.AddInputData(meshNew.VTKObject)
        #     append_filter.Update()
        #     temp = append_filter.GetOutput()
        # else:
        #     append_filter.AddInputData(temp)
        #     append_filter.AddInputData(meshNew.VTKObject)
        #     append_filter.Update()
        #     temp = append_filter.GetOutput()
        #
        # if args.debug and var == bridge_list[-1]:
        #     writer = vtk.vtkXMLUnstructuredGridWriter()
        #     writer.SetFileName(job.ID + "/result_RA/append_bridges_with_tag.vtu")
        #     writer.SetInputData(temp)
        #     writer.Write()

        #append_filter.AddInputData(meshNew.VTKObject) # Check here if we still have the element tag
        #append_filter.Update()

        #writer = vtk.vtkXMLUnstructuredGridWriter()
        #writer.SetFileName(job.ID + "/result_RA/LA_RA_with_bundles_with_" + str(var) + ".vtu") # Here we lose the elemTag
        #writer.SetInputData(append_filter.GetOutput())
        #writer.Write()

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(job.ID + "/result_RA/LA_RA_epi_with_holes.vtu")
    reader.Update()
    epi_new = reader.GetOutput()

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(job.ID + "/result_RA/append_bridges_2.vtu")
    reader.Update()
    bridges = reader.GetOutput()

    append_filter = vtk.vtkAppendFilter()
    #append_filter.AddInputData(la_ra_epi)
    append_filter.AddInputData(epi_new)
    append_filter.AddInputData(bridges)
    #append_filter.MergePointsOn()
    append_filter.Update()
    epi = append_filter.GetOutput()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(job.ID+"/result_RA/LA_RA_with_bundles.vtu")
    writer.SetInputData(epi)
    writer.Write()
    
    epi = Method.generate_sheet_dir(args, epi, job)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(job.ID+"/result_RA/LA_RA_with_sheets.vtu")
    writer.SetInputData(epi)
    writer.Write()
    
    if args.mesh_type == "bilayer":

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/la_ra_epi_with_sheets.vtu") # Has elemTag! :)
        writer.SetInputData(epi)
        writer.Write()

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(job.ID+"/result_RA/RA_CT_PMs.vtu") # Has elemTag! :)
        reader.Update()
        ra_endo = reader.GetOutput()
        
        reader = vtk.vtkXMLUnstructuredGridReader()
        #reader.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtu") # Has elemTag! :)
        reader.SetFileName(job.ID + "/result_RA/LA_endo_with_holes.vtu")  # Has elemTag! :)
        reader.Update()
        la_endo = reader.GetOutput()
        
        append_filter = vtk.vtkAppendFilter()
        append_filter.AddInputData(la_endo)
        append_filter.AddInputData(ra_endo)
        append_filter.Update()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID + "/result_RA/append_LA_endo_RA_endo.vtu") # Has elemTag! :)
        writer.SetInputData(append_filter.GetOutput())
        writer.Write()

        endo = Method.move_surf_along_normals(append_filter.GetOutput(), 0.1*args.scale, 1) # # Warning: set -1 if pts normals are pointing outside
        
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/la_ra_endo.vtu") # Has elemTag! :)
        writer.SetInputData(endo)
        writer.Write()
        
        bilayer = Method.generate_bilayer(args,job, endo, epi, 0.12*args.scale) # Does not have elemTag :(!
        
        Method.write_bilayer(args, job)

    else:

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/LA_RA_vol_with_fiber.vtu")
        writer.SetInputData(epi)
        writer.Write()

        pts = vtk.util.numpy_support.vtk_to_numpy(epi.GetPoints().GetData())
        with open(job.ID+'/result_RA/LA_RA_vol_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
        
        tag_epi = vtk.util.numpy_support.vtk_to_numpy(epi.GetCellData().GetArray('elemTag'))

        with open(job.ID+'/result_RA/LA_RA_vol_with_fiber.elem',"w") as f:
            f.write("{}\n".format(epi.GetNumberOfCells()))
            for i in range(epi.GetNumberOfCells()):
                cell = epi.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_epi[i]))
                else:
                    print("strange "+ str(cell.GetNumberOfPoints()))
        el_epi = vtk.util.numpy_support.vtk_to_numpy(epi.GetCellData().GetArray('fiber'))
        sheet_epi = vtk.util.numpy_support.vtk_to_numpy(epi.GetCellData().GetArray('sheet'))
        
        with open(job.ID+'/result_RA/LA_RA_vol_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el_epi)):
                f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))
