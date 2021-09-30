#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:05:58 2020

@author: tz205
"""
#from mayavi import mlab
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
import Method
import csv
import datetime
import pandas as pd
from la_laplace import laplace_0_1
import os

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]
      
def la_generate_fiber(model, args, job):
    # Ideal tao
    # tao_mv = 0.65
    # tao_lpv = 0.65
    # tao_lpv = 0.1
    
    # size(Radius) of Bachmann Bundle in mm
    w_bb = 2*args.scale
    
    simid = job.ID+"/result_LA"
    try:
        os.makedirs(simid)
    except OSError:
        print ("Creation of the directory %s failed" % simid)
    else:
        print ("Successfully created the directory %s " % simid)
            
    with open(os.path.join(EXAMPLE_DIR,'../../element_tag.csv')) as f:
        tag_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            tag_dict[row['name']] = row['tag']
    # load epi tags
    mitral_valve_epi = int(tag_dict['mitral_valve_epi'])
    superior_left_pulmonary_vein_epi = int(tag_dict['superior_left_pulmonary_vein_epi'])
    inferior_left_pulmonary_vein_epi = int(tag_dict['inferior_left_pulmonary_vein_epi'])
    superior_right_pulmonary_vein_epi = int(tag_dict['superior_right_pulmonary_vein_epi'])
    inferior_right_pulmonary_vein_epi = int(tag_dict['inferior_right_pulmonary_vein_epi'])
    left_atrial_appendage_epi = int(tag_dict['left_atrial_appendage_epi'])
    left_atrial_wall_epi = int(tag_dict['left_atrial_wall_epi'])
    left_atrial_lateral_wall_epi = int(tag_dict['left_atrial_lateral_wall_epi'])
    left_septum_wall_epi = int(tag_dict['left_septum_wall_epi'])
    left_atrial_roof_epi = int(tag_dict['left_atrial_roof_epi'])

    # load endo tags
    mitral_valve_endo = int(tag_dict['mitral_valve_endo'])
    superior_left_pulmonary_vein_endo = int(tag_dict['superior_left_pulmonary_vein_endo'])
    inferior_left_pulmonary_vein_endo = int(tag_dict['inferior_left_pulmonary_vein_endo'])
    superior_right_pulmonary_vein_endo = int(tag_dict['superior_right_pulmonary_vein_endo'])
    inferior_right_pulmonary_vein_endo = int(tag_dict['inferior_right_pulmonary_vein_endo'])
    left_atrial_appendage_endo = int(tag_dict['left_atrial_appendage_endo'])
    left_atrial_wall_endo = int(tag_dict['left_atrial_wall_endo'])
    left_atrial_lateral_wall_endo = int(tag_dict['left_atrial_lateral_wall_endo'])
    left_septum_wall_endo = int(tag_dict['left_septum_wall_endo'])
    left_atrial_roof_endo = int(tag_dict['left_atrial_roof_endo'])
    bachmann_bundel_left = int(tag_dict['bachmann_bundel_left'])
    # Zygote or Riunet
    tao_mv = 0.85
    tao_lpv = 0.85
    tao_rpv = 0.2
    
    #start_time = datetime.datetime.now()
    #print('Reading LA_with_lp_res_gradient.vtk... ' + str(start_time))

    # TODO PLAN A: read all gradient and Phie(laplace solution) form one vtk file
    ### Plan A using one VTK file
    #reader = vtk.vtkUnstructuredGridReader()
    #reader.SetFileName('gradient/LA_with_lp_res_gradient.vtk')
    #reader.Update()
    #model = reader.GetOutput()

    # ab
    ab = model.GetCellData().GetArray('phie_ab')
    ab_grad = model.GetCellData().GetArray('grad_ab')
    ab = vtk.util.numpy_support.vtk_to_numpy(ab)
    ab_grad = vtk.util.numpy_support.vtk_to_numpy(ab_grad)

    # v
    v = model.GetCellData().GetArray('phie_v')
    v_grad = model.GetCellData().GetArray('grad_v')
    v = vtk.util.numpy_support.vtk_to_numpy(v)
    v_grad = vtk.util.numpy_support.vtk_to_numpy(v_grad)

    # r
    r = model.GetCellData().GetArray('phie_r')
    r_grad = model.GetCellData().GetArray('grad_r')
    r = vtk.util.numpy_support.vtk_to_numpy(r)
    r_grad = vtk.util.numpy_support.vtk_to_numpy(r_grad)
    
    # r2
    r2 = model.GetCellData().GetArray('phie_r2')
    r2 = vtk.util.numpy_support.vtk_to_numpy(r2)


    # phie
    if args.mesh_type == "vol":
        phie = model.GetCellData().GetArray('phie_phi')
        phie = vtk.util.numpy_support.vtk_to_numpy(phie)
    
    phie_grad = model.GetCellData().GetArray('grad_phi')
    phie_grad = vtk.util.numpy_support.vtk_to_numpy(phie_grad)


    #end_time = datetime.datetime.now()
    #running_time = end_time - start_time
    #print('Reading LA_with_lp_res_gradient.vtk......done ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(model) # vtkPolyData()
    cellid.PointIdsOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()
    
    model = cellid.GetOutput()
    
    df = pd.read_csv(args.mesh+"_surf/rings_centroids.csv")
    
    # LPV
    lb = 0
    ub = 0.4
    tao_lpv = Method.find_tau(model, ub, lb, "low", "phie_v")
    print('Calculating tao_lpv done! tap_lpv = ', tao_lpv)
    
    thr = Method.vtk_thr(model, 1, "CELLS","phie_v",tao_lpv)
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()

    PVs = dict()
    # Distinguish between LIPV and LSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "LIPV", "LSPV")

    model = laplace_0_1(args, job, model, "RPV", "LAA", "phie_ab2")

    thr = Method.vtk_thr(model, 1, "CELLS","phie_v", 0.35)
    found, val = Method.optimize_shape_PV(thr, 10, 0)
    print('Calculating opt_tao_lpv done! tap_lpv = ', val)
    if found:
        thr = Method.vtk_thr(model, 1, "CELLS","phie_v",val)
    
    phie_r2_tau_lpv = vtk.util.numpy_support.vtk_to_numpy(thr.GetCellData().GetArray('phie_r2'))
    max_phie_r2_tau_lpv = np.max(phie_r2_tau_lpv)
    
    #max_cell_phie_r2_tau_lpv=int(thresh.GetOutput().GetCellData().GetArray('Global_ids').GetTuple(np.argmax(phie_r2_tau_lpv))[0])

    # writer = vtk.vtkUnstructuredGridWriter()
    # writer.SetFileName(job.ID+"/LA_with_ab_2.vtk")
    # writer.SetInputData(model)
    # writer.Write()

    phie_ab_tau_lpv = vtk.util.numpy_support.vtk_to_numpy(thr.GetPointData().GetArray('phie_ab2'))
    max_phie_ab_tau_lpv = np.max(phie_ab_tau_lpv)

    print("max_phie_r2_tau_lpv ",max_phie_r2_tau_lpv)
    print("max_phie_ab_tau_lpv ",max_phie_ab_tau_lpv)

    # phie_ab_tau_lpv = vtk.util.numpy_support.vtk_to_numpy(thr.GetCellData().GetArray('phie_ab'))
    # max_phie_ab_tau_lpv = np.max(phie_ab_tau_lpv)
    
    # RPV
    lb = 0.6
    ub = 1
    tao_rpv = Method.find_tau(model, ub, lb, "up", "phie_v")
    print('Calculating tao_rpv done! tap_rpv = ', tao_rpv)
    
    thr = Method.vtk_thr(model, 0, "CELLS","phie_v",tao_rpv)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    
    # Distinguish between RIPV and RSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "RIPV", "RSPV")

    start_time = datetime.datetime.now()
    print('Calculating fibers... ' + str(start_time))
    
    ## plan A
    # #### Bundles selection ####
    # for i in range(len(ab_grad)):
    #     if r[i] >= tao_mv:
    #         ab_grad[i] = r_grad[i]
    #         tag[i] = mitral_valve_epi
    #     elif ab[i] < -0.04:
    #         tag[i] = left_atrial_appendage_epi
    #     elif v[i] <= tao_lpv:
    #         ab_grad[i] = v_grad[i]
    #         tag[i] = inferior_left_pulmonary_vein_epi
    #         lpv_id_test.InsertNextId(i)
    #     elif v[i] >= tao_rpv:
    #         ab_grad[i] = v_grad[i]
    #         tag[i] = inferior_right_pulmonary_vein_epi
    #         rpv_id_test.InsertNextId(i)
    #     else:
    #         tag[i] = left_atrial_wall_epi
    
    # Determine tao_bb
    # # plan C region growing
    # loc = vtk.vtkPointLocator()
    # loc.SetDataSet(model)
    # loc.BuildLocator()
    # la_appendage_basis_id = loc.FindClosestPoint(appendage_basis)
    # la_appendage_basis = model.GetPoint(la_appendage_basis_id)
    # touch_ab = 0
    # value = 0.9
    # step = 0.05
    # print(la_appendage_basis)
    # print(la_appendage_basis[0])
    # while touch_ab == 0:
    #     thresh = vtk.vtkThreshold()
    #     thresh.SetInputData(model)
    #     thresh.ThresholdByUpper(value)
    #     thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r")
    #     thresh.Update()
    #     temp = thresh.GetOutput()
    #     points_data = temp.GetPoints().GetData()
    #     temp = vtk.util.numpy_support.vtk_to_numpy(points_data)
    
    #     # touch_ab = Method.multidim_intersect_bool(la_appendage_basis_array, temp)
    #     # touch_ab = np.isin(la_appendage_basis_array, temp, invert=False)
    #     touch_ab = (temp == la_appendage_basis).all(1).any()
    #     print("touch_ab: ", touch_ab)
    #     if touch_ab == 0:
    #         value -= step
    #     print("Iteration: ", k)
    #     print("Value of tao_bb: ", value)
    #     k += 1
    
    
    
    # appendage_basis = df["LAA_base"].to_numpy()
    
    # loc = vtk.vtkPointLocator()
    # loc.SetDataSet(model)
    # loc.BuildLocator()
    # la_appendage_basis_id = loc.FindClosestPoint(appendage_basis)
    # la_appendage_basis = model.GetPoint(la_appendage_basis_id)
    
    # cell_id_temp = vtk.vtkIdList()
    # model.GetPointCells(la_appendage_basis_id, cell_id_temp)
    # cell_id_tao_bb = cell_id_temp.GetId(0)
    # tao_bb = r[cell_id_tao_bb]
    # print('tao_bb is : ', tao_bb)

    #lpv = Method.smart_reader('/Volumes/koala/Users/tz205/IBT_tz205_la816_MA/Data/AtrialLDRBM/Generate_Boundaries/LA/result_LA/la_lpv_surface.vtk')
    #rpv = Method.smart_reader('/Volumes/koala/Users/tz205/IBT_tz205_la816_MA/Data/AtrialLDRBM/Generate_Boundaries/LA/result_LA/la_rpv_surface.vtk')
    #mv = Method.smart_reader('/Volumes/koala/Users/tz205/IBT_tz205_la816_MA/Data/AtrialLDRBM/Generate_Boundaries/LA/result_LA/la_mv_surface.vtk')
    
    #lpv_mean = Method.get_mean_point(lpv)
    #rpv_mean = Method.get_mean_point(rpv)
    #mv_mean = Method.get_mean_point(mv)
    
    # lpv_mean = np.mean([df["LIPV"].to_numpy(), df["LSPV"].to_numpy()], axis = 0)
    # rpv_mean = np.mean([df["RIPV"].to_numpy(), df["RSPV"].to_numpy()], axis = 0)
    # mv_mean = df["MV"].to_numpy()
    
    # v1 = rpv_mean - mv_mean
    # v2 = lpv_mean - mv_mean
    # norm = np.cross(v2, v1)
    
    # # # normalize vector
    # norm = norm / np.linalg.norm(norm)

    # plane = vtk.vtkPlane()
    # plane.SetNormal(norm[0], norm[1], norm[2])
    # plane.SetOrigin(mv_mean[0], mv_mean[1], mv_mean[2])

    # meshExtractFilter1 = vtk.vtkExtractGeometry()
    # meshExtractFilter1.SetInputData(cellid.GetOutput())
    # meshExtractFilter1.SetImplicitFunction(plane)
    # meshExtractFilter1.Update()
    
    # band_cell_ids = vtk.util.numpy_support.vtk_to_numpy(meshExtractFilter1.GetOutput().GetCellData().GetArray('Global_ids'))
    
    # print(cell_ids)
    # print(18884 in cell_ids)    
    
    # LAA_base = df["LAA_base"].to_numpy()
    # loc = vtk.vtkPointLocator()
    # loc.SetDataSet(model)
    # loc.BuildLocator()
    # LAA_base_id = loc.FindClosestPoint(LAA_base)
    # mesh_cell_id_list = vtk.vtkIdList()
    # model.GetPointCells(LAA_base_id, mesh_cell_id_list)
    
    #### Bundles selection ####
    # Volume mesh
    if args.mesh_type == 'vol':
        tag = np.zeros(len(r))
        
        for i in range(len(ab_grad)):
            # tagging endo-layer
            if phie[i] <= 0.5:
                if r[i] >= tao_mv:
                    tag[i] = mitral_valve_endo
                elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
                    tag[i] = left_atrial_appendage_endo
                elif v[i] <= tao_lpv and i in PVs["LIPV"]:
                    tag[i] = inferior_left_pulmonary_vein_endo
                elif v[i] <= tao_lpv and i in PVs["LSPV"]:
                    tag[i] = superior_left_pulmonary_vein_endo
                elif v[i] >= tao_rpv and i in PVs["RIPV"]:
                    tag[i] = inferior_right_pulmonary_vein_endo
                elif v[i] >= tao_rpv and i in PVs["RSPV"]:
                    tag[i] = superior_right_pulmonary_vein_endo
                else:
                    tag[i] = left_atrial_wall_endo
            # tagging epi-layer
            else:
                if r[i] >= tao_mv:
                    ab_grad[i] = r_grad[i]
                    tag[i] = mitral_valve_epi
                elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
                    tag[i] = left_atrial_appendage_epi
                elif v[i] <= tao_lpv and i in PVs["LIPV"]:
                    ab_grad[i] = v_grad[i]
                    tag[i] = inferior_left_pulmonary_vein_epi
                elif v[i] <= tao_lpv and i in PVs["LSPV"]:
                    ab_grad[i] = v_grad[i]
                    tag[i] = superior_left_pulmonary_vein_epi
                elif v[i] >= tao_rpv and i in PVs["RIPV"]:
                    ab_grad[i] = v_grad[i]
                    tag[i] = inferior_right_pulmonary_vein_epi
                elif v[i] >= tao_rpv and i in PVs["RSPV"]:
                    ab_grad[i] = v_grad[i]
                    tag[i] = superior_right_pulmonary_vein_epi
                else:
                    tag[i] = left_atrial_wall_epi
        
        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        model = meshNew.VTKObject
        
        # normlize the gradient phie
        abs_phie_grad = np.linalg.norm(phie_grad, axis=1, keepdims=True)
        abs_phie_grad = np.where(abs_phie_grad != 0, abs_phie_grad, 1)
        phie_grad_norm = phie_grad / abs_phie_grad
    
        ##### Local coordinate system #####
        # et
        et = phie_grad_norm
        print('############### et ###############')
        print(et)
    
        # k
        k = ab_grad
        print('############### k ###############')
        print(k)
    
        # en
        en = ab_grad
        #for i in range(len(k)):
        #    en[i] = k[i] - np.dot(k[i], et[i]) * et[i]
            
        en = k - np.dot(k, et) * et
        # normalize the en
        abs_en = np.linalg.norm(en, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en = en / abs_en
        print('############### en ###############')
        print(en)
    
        # el
        el = np.cross(en, et)
        print('############### el ###############')
        print(el)
    
        abs_v_grad = np.linalg.norm(v_grad, axis=1, keepdims=True)
        abs_v_grad = np.where(abs_v_grad != 0, abs_v_grad, 1)
        v_grad_norm = v_grad / abs_v_grad
        ### Subendo bundle selection
        for i in range(len(v_grad_norm)):
            if phie[i] < 0.5 and v[i] <= 0.2:
                el[i] = v_grad_norm[i]
    
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName('Mesh/LA_mesh.vtk')
        reader.Update()
        model = reader.GetOutput()
    
        print("Creating bachmann bundles...")
        
        # Extract surface
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(model)
        geo_filter.PassThroughPointIdsOn()
        geo_filter.Update()
        # Get only epi
        thr = Method.vtk_thr(geo_filter.GetOutput(), 2, "CELLS","elemTag",mitral_valve_epi, left_atrial_roof_epi)
        # Bachmann Bundle
        
        # Get ending point of wide BB
        # Extract the MV
        thr = Method.vtk_thr(thr, 2, "CELLS","elemTag",mitral_valve_epi, mitral_valve_epi)
        
        # Get the closest point to the inferior appendage base in the MV
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(thr)
        loc.BuildLocator()
    
        bb_mv_id =  int(model.GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(model.GetPoint(inf_appendage_basis_id)))[0])
        
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(model)
        thresh.ThresholdBetween(max_phie_ab_tau_lpv-0.1, max_phie_ab_tau_lpv+0.1)
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_ab")
        thresh.Update()
        
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(thresh.GetOutput())
        loc.BuildLocator()
    
        inf_appendage_basis_id = int(thresh.GetOutput().GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(df["MV"].to_numpy()))[0])
        
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(thresh.GetOutput())
        thresh.ThresholdBetween(max_phie_r2_tau_lpv-0.1, max_phie_r2_tau_lpv+0.1)
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r2")
        thresh.Update()
        
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(thresh.GetOutput())
        loc.BuildLocator()
    
        sup_appendage_basis_id = int(thresh.GetOutput().GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(df["LAA"].to_numpy()))[0])
    
    
        bb_left = Method.get_wide_bachmann_path_left(thresh.GetOutput(), inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id)
        
        tag = Method.assign_element_tag_around_path_within_radius(model, bb_left, 2, tag, bachmann_bundel_left)
        el = Method.assign_element_fiber_around_path_within_radius(model, bb_left, 2, el, smooth=True)
    
        # # Bachmann_Bundle internal connection
        # la_connect_point, ra_connect_point = Method.get_connection_point_la_and_ra(la_appex_point)
        # la_connect_point = np.asarray(la_connect_point)
        #
        # path_start_end = np.vstack((appendage_basis, la_connect_point))
        # path_bb_ra = Method.creat_center_line(path_start_end)
        # tag = Method.assign_element_tag_around_path_within_radius(model, path_bb_ra, 2, tag, bachmann_bundel_left)
        # el = Method.assign_element_fiber_around_path_within_radius(model, path_bb_ra, 2, el, smooth=True)
    
        print("Creating bachmann bundles... done")
        abs_el = np.linalg.norm(el, axis=1, keepdims=True)
        interpolate_arr = np.asarray([0, 0, 1])
        index = np.argwhere(abs_el == 0)
        print('There is',len(index),'zero vector(s).')
        for var in index:
            el[var[0]] = interpolate_arr
    
        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))
    
        # fiber_data = vtk.util.numpy_support.numpy_to_vtk(el, deep=True, array_type=vtk.VTK_DOUBLE)
        # fiber_data.SetNumberOfComponents(3)
        # fiber_data.SetName("fiber")
        # model.GetCellData().SetVectors(fiber_data)  # AddArray(fiber_data)
    
        # tag_data = vtk.util.numpy_support.numpy_to_vtk(tag, deep=True, array_type=vtk.VTK_DOUBLE)
        # tag_data.SetNumberOfComponents(1)
        # tag_data.SetName("elemTag")
        # model.GetCellData().RemoveArray("Cell_ids")
        # model.GetCellData().RemoveArray("elemTag")
        # model.GetCellData().SetScalars(tag_data)

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
    
        model = meshNew.VTKObject
        # meshNew = dsa.WrapDataObject(model)
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName(job.ID+"/result_LA/LA_with_fiber.vtk")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
    
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
        
    
    # Bilayer mesh
    elif args.mesh_type == 'bilayer':
        epi = vtk.vtkUnstructuredGrid()
        epi.DeepCopy(model)
        
        tag_epi = np.zeros(len(r), dtype=int)
        tag_epi[:] = left_atrial_wall_epi
        tag_endo = np.zeros(len(r), dtype=int)
        tag_endo[:] = left_atrial_wall_endo
        # # tagging endo-layer
        # for i in range(len(r)):
        #     if tag_endo[i] == 0:
        #         if r[i] >= tao_mv:
        #             tag_endo[i] = mitral_valve_endo
        #         elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
        #             tag_endo[i] = left_atrial_appendage_endo
        #         elif v[i] <= tao_lpv and i in PVs["LIPV"]:
        #             tag_endo[i] = inferior_left_pulmonary_vein_endo
        #         elif v[i] <= tao_lpv and i in PVs["LSPV"]:
        #             tag_endo[i] = superior_left_pulmonary_vein_endo
        #         elif v[i] >= tao_rpv and i in PVs["RIPV"]:
        #             tag_endo[i] = inferior_right_pulmonary_vein_endo
        #         elif v[i] >= tao_rpv and i in PVs["RSPV"]:
        #             tag_endo[i] = superior_right_pulmonary_vein_endo
        #         else:
        #             tag_endo[i] = left_atrial_wall_endo
        
        # # tagging epi-layer
        # for i in range(len(r)):
        #     if tag_epi[i] == 0:
        #         if r[i] >= tao_mv:
        #             ab_grad[i] = r_grad[i]
        #             tag_epi[i] = mitral_valve_epi
        #         elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
        #             tag_epi[i] = left_atrial_appendage_epi
        #         elif v[i] <= tao_lpv and i in PVs["LIPV"]:
        #             ab_grad[i] = v_grad[i]
        #             tag_epi[i] = inferior_left_pulmonary_vein_epi
        #         elif v[i] <= tao_lpv and i in PVs["LSPV"]:
        #             ab_grad[i] = v_grad[i]
        #             tag_epi[i] = superior_left_pulmonary_vein_epi
        #         elif v[i] >= tao_rpv and i in PVs["RIPV"]:
        #             ab_grad[i] = v_grad[i]
        #             tag_epi[i] = inferior_right_pulmonary_vein_epi
        #         elif v[i] >= tao_rpv and i in PVs["RSPV"]:
        #             ab_grad[i] = v_grad[i]
        #             tag_epi[i] = superior_right_pulmonary_vein_epi
        #         else:
        #             tag_epi[i] = left_atrial_wall_epi
                    
        # tagging endo-layer
        #MV_s = Method.vtk_thr(model,0,"CELLS","phie_r",tao_mv)
        
        LAA_s = Method.vtk_thr(model,0,"CELLS","phie_r2",max_phie_r2_tau_lpv+0.01)
        
        LAA_s = Method.vtk_thr(LAA_s,1,"POINTS","phie_ab2",max_phie_ab_tau_lpv+0.01)
        
        ## Optimize shape of LAA solving a laplacian with 0 in LAA and 1 in the boundary of LAA_s
        
        LAA_bb = Method.vtk_thr(LAA_s,2,"POINTS","phie_ab2",max_phie_ab_tau_lpv, max_phie_ab_tau_lpv+0.01)
        
        LAA_bb_ids = vtk.util.numpy_support.vtk_to_numpy(LAA_bb.GetPointData().GetArray('Global_ids'))
        
        MV_ring_ids = np.loadtxt('{}_surf/ids_MV.vtx'.format(args.mesh), skiprows=2, dtype=int)

        LAA_bb_ids = np.append(LAA_bb_ids, MV_ring_ids)

        fname = '{}_surf/ids_LAA_bb.vtx'.format(args.mesh)
        f = open(fname, 'w')
        f.write('{}\n'.format(len(LAA_bb_ids)))
        f.write('extra\n')
        for i in LAA_bb_ids:
            f.write('{}\n'.format(i))
        f.close()
        
        LAA_s = laplace_0_1(args, job, model, "LAA", "LAA_bb", "phie_ab3")
        
        LAA_s = Method.vtk_thr(LAA_s,1,"POINTS","phie_ab3",0.92)
        
        #LPV_s = Method.vtk_thr(model,1,"CELLS","phie_v",tao_lpv)
        
        #RPV_s = Method.vtk_thr(model,0,"CELLS","phie_v",tao_rpv)
        
        #MV_ids = vtk.util.numpy_support.vtk_to_numpy(MV_s.GetCellData().GetArray('Global_ids'))
        
        ring_ids = np.loadtxt('{}_surf/'.format(args.mesh) + 'ids_MV.vtx', skiprows=2, dtype=int)
        
        rings_pts = vtk.util.numpy_support.vtk_to_numpy(model.GetPoints().GetData())[ring_ids,:]
        
        MV_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4*args.scale)
        
        LAA_ids = vtk.util.numpy_support.vtk_to_numpy(LAA_s.GetCellData().GetArray('Global_ids'))
        
        #LPV_ids = vtk.util.numpy_support.vtk_to_numpy(LPV_s.GetCellData().GetArray('Global_ids'))
        
        #RPV_ids = vtk.util.numpy_support.vtk_to_numpy(RPV_s.GetCellData().GetArray('Global_ids'))
        
        # tagging endo-layer
        
        tag_endo[MV_ids] = mitral_valve_endo
        ab_grad[MV_ids] = r_grad[MV_ids]
        
        tag_endo[LAA_ids] = left_atrial_appendage_endo
        
        #tag_endo[LPV_ids] = inferior_left_pulmonary_vein_endo
        #ab_grad[LPV_ids] = v_grad[LPV_ids]
        
        #tag_endo[RPV_ids] = inferior_right_pulmonary_vein_endo
        #ab_grad[RPV_ids] = v_grad[RPV_ids]
        
        tag_endo[PVs["RIPV"]] = inferior_right_pulmonary_vein_endo
        tag_endo[PVs["LIPV"]] = inferior_left_pulmonary_vein_endo
        
        tag_endo[PVs["RSPV"]] = superior_right_pulmonary_vein_endo
        tag_endo[PVs["LSPV"]] = superior_left_pulmonary_vein_endo
        
        ab_grad[PVs["RIPV"]] = v_grad[PVs["RIPV"]]
        ab_grad[PVs["LIPV"]] = v_grad[PVs["LIPV"]]
        ab_grad[PVs["RSPV"]] = v_grad[PVs["RSPV"]]
        ab_grad[PVs["LSPV"]] = v_grad[PVs["LSPV"]]
        
        # tagging epi-layer
        
        tag_epi[MV_ids] = mitral_valve_epi
        
        tag_epi[LAA_ids] = left_atrial_appendage_epi
        
        #tag_epi[LPV_ids] = inferior_left_pulmonary_vein_epi
        
        #tag_epi[RPV_ids] = inferior_right_pulmonary_vein_epi
        
        tag_epi[PVs["RIPV"]] = inferior_right_pulmonary_vein_epi
        tag_epi[PVs["LIPV"]] = inferior_left_pulmonary_vein_epi
        
        tag_epi[PVs["RSPV"]] = superior_right_pulmonary_vein_epi
        tag_epi[PVs["LSPV"]] = superior_left_pulmonary_vein_epi
        
        
        ab_grad_epi = np.copy(ab_grad)
        
        # Get epi bundle band
        
        lpv_mean = np.mean([df["LIPV"].to_numpy(), df["LSPV"].to_numpy()], axis = 0)
        rpv_mean = np.mean([df["RIPV"].to_numpy(), df["RSPV"].to_numpy()], axis = 0)
        mv_mean = df["MV"].to_numpy()
        
        v1 = rpv_mean - mv_mean
        v2 = lpv_mean - mv_mean
        norm = np.cross(v2, v1)
        
        norm = norm / np.linalg.norm(norm)
        
        band_s = Method.vtk_thr(epi,0,"CELLS","phie_r2",max_phie_r2_tau_lpv)
        
        plane = vtk.vtkPlane()
        plane.SetNormal(norm[0], norm[1], norm[2])
        plane.SetOrigin(mv_mean[0], mv_mean[1], mv_mean[2])
    
        meshExtractFilter = vtk.vtkExtractGeometry()
        meshExtractFilter.SetInputData(band_s)
        meshExtractFilter.SetImplicitFunction(plane)
        meshExtractFilter.Update()
        
        band_cell_ids = vtk.util.numpy_support.vtk_to_numpy(meshExtractFilter.GetOutput().GetCellData().GetArray('Global_ids'))
        
        ab_grad_epi[band_cell_ids] = r_grad[band_cell_ids]
        
        # for i in range(len(r)):
        #     if r[i] >= tao_mv:
        #         ab_grad[i] = r_grad[i]
        #         tag_endo[i] = mitral_valve_endo
        #     elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
        #         tag_endo[i] = left_atrial_appendage_endo
        #     elif v[i] <= tao_lpv:
        #         ab_grad[i] = v_grad[i]
        #         tag_endo[i] = inferior_left_pulmonary_vein_endo
        #     elif v[i] >= tao_rpv:
        #         ab_grad[i] = v_grad[i]
        #         tag_endo[i] = inferior_right_pulmonary_vein_endo
        #     else:
        #         tag_endo[i] = left_atrial_wall_endo
        
        # tag_endo[PVs["RSPV"]] = superior_right_pulmonary_vein_endo
        # tag_endo[PVs["LSPV"]] = superior_left_pulmonary_vein_endo
        
        # # tagging epi-layer
        # for i in range(len(r)):
        #     if r[i] >= tao_mv:
        #         tag_epi[i] = mitral_valve_epi
        #     elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
        #         tag_epi[i] = left_atrial_appendage_epi
        #     elif v[i] <= tao_lpv:
        #         tag_epi[i] = inferior_left_pulmonary_vein_epi
        #     elif v[i] >= tao_rpv:
        #         tag_epi[i] = inferior_right_pulmonary_vein_epi
        #     else:
        #         tag_epi[i] = left_atrial_wall_epi
        # tag_epi[PVs["RSPV"]] = superior_right_pulmonary_vein_epi
        # tag_epi[PVs["LSPV"]] = superior_left_pulmonary_vein_epi
        
        # ab_grad_epi = np.copy(ab_grad)
        
        # for var in band_cell_ids:
        #     if r2[var] >= max_phie_r2_tau_lpv:
        #         ab_grad_epi[var] = r_grad[var]
    
        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag_endo, "elemTag")
        endo = meshNew.VTKObject
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName("gradient/LA_endo_with_tags.vtk")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
        
        meshNew = dsa.WrapDataObject(epi)
        meshNew.CellData.append(tag_epi, "elemTag")
        epi = meshNew.VTKObject
        
        # meshNew = dsa.WrapDataObject(epi)
        # meshNew.CellData.append(tag_epi, "elemTag")
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName("gradient/LA_epi_with_tags.vtk")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
        
        # normlize the gradient phie
        phie_grad_norm = phie_grad
    
        ##### Local coordinate system #####
        # et
        et = phie_grad_norm
        print('############### et ###############')
        #print(et)
    
        # k
        k_endo = ab_grad
        k_epi = ab_grad_epi
        print('############### k ###############')
        #print(k)
    
        # en
        # en_endo = np.zeros((len(et),3))
        # en_epi = np.zeros((len(et),3))
        # # Gram-Schmidt orthogonalization 
        # for i in range(len(k_endo)):
        #     en_endo[i] = k_endo[i] - np.dot(k_endo[i], et[i])* et[i]
        #     en_epi[i] = k_epi[i] - np.dot(k_epi[i], et[i])* et[i]
        
        en_endo = k_endo - et*np.sum(k_endo*et,axis=1).reshape(len(et),1)
        en_epi = k_epi - et*np.sum(k_epi*et,axis=1).reshape(len(et),1)
        
        # normalize the en
        abs_en = np.linalg.norm(en_endo, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en_endo = en_endo / abs_en
        
        abs_en = np.linalg.norm(en_epi, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en_epi = en_epi / abs_en
        print('############### en ###############')
        #print(en)
    
        # el
        el_endo = np.cross(en_endo, et)
        el_epi = np.cross(en_epi, et)
        print('############### el ###############')
        #print(el)
    
        abs_v_grad = np.linalg.norm(v_grad, axis=1, keepdims=True)
        abs_v_grad = np.where(abs_v_grad != 0, abs_v_grad, 1)
        v_grad_norm = v_grad / abs_v_grad
        
        ### Subendo PVs bundle selection
        
        #el_endo[LPV_ids] = v_grad_norm[LPV_ids]
        
        el_endo[PVs["LIPV"]] = v_grad_norm[PVs["LIPV"]]
        el_endo[PVs["LSPV"]] = v_grad_norm[PVs["LSPV"]]
        
        # for i in range(len(v_grad_norm)):
        #     if v[i] <= 0.2:
        #         el_endo[i] = v_grad_norm[i]
    
        end_time = datetime.datetime.now()
        
        el_endo = np.where(el_endo == [0,0,0], [1,0,0], el_endo).astype("float32")
        
        sheet_endo = np.cross(el_endo, et)
        sheet_endo = np.where(sheet_endo == [0,0,0], [1,0,0], sheet_endo).astype("float32")
        
        for i in range(endo.GetPointData().GetNumberOfArrays()-1, -1, -1):
            endo.GetPointData().RemoveArray(endo.GetPointData().GetArrayName(i))
        
        for i in range(endo.GetCellData().GetNumberOfArrays()-1, -1, -1):
            endo.GetCellData().RemoveArray(endo.GetCellData().GetArrayName(i))
        
        meshNew = dsa.WrapDataObject(endo)
        meshNew.CellData.append(tag_endo, "elemTag")
        meshNew.CellData.append(el_endo, "fiber")
        meshNew.CellData.append(sheet_endo, "sheet")
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtk")
        writer.SetFileTypeToBinary()
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        pts = numpy_support.vtk_to_numpy(endo.GetPoints().GetData())
        with open(job.ID+'/result_LA/LA_endo_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
        with open(job.ID+'/result_LA/LA_endo_with_fiber.elem',"w") as f:
            f.write("{}\n".format(endo.GetNumberOfCells()))
            for i in range(endo.GetNumberOfCells()):
                cell = endo.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_endo[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_endo[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_endo[i]))
        
        
        with open(job.ID+'/result_LA/LA_endo_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el_endo)):
                f.write("{} {} {} {} {} {}\n".format(el_endo[i][0], el_endo[i][1], el_endo[i][2], sheet_endo[i][0], sheet_endo[i][1], sheet_endo[i][2]))
            
        running_time = end_time - start_time
        
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName("gradient/LA_endo_with_fiber.vtk")
        # writer.SetInputData(endo)
        # writer.Write()
        
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName("gradient/LA_epi_with_fiber.vtk")
        # writer.SetInputData(epi)
        # writer.Write()
        
        print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    
        print("Creating bachmann bundles...")
        # Bachmann Bundle
        
        bb_left, LAA_basis_inf, LAA_basis_sup, LAA_far_from_LIPV = Method.compute_wide_BB_path_left(epi, df, left_atrial_appendage_epi, mitral_valve_epi)
        
        tag_epi = Method.assign_element_tag_around_path_within_radius(epi, bb_left, w_bb, tag_epi, bachmann_bundel_left)
        el_epi = Method.assign_element_fiber_around_path_within_radius(epi, bb_left, w_bb, el_epi, smooth=True)
        
        df["LAA_basis_inf"] = LAA_basis_inf
        df["LAA_basis_sup"] = LAA_basis_sup
        df["LAA_far_from_LIPV"] = LAA_far_from_LIPV
        
        df.to_csv(args.mesh+"_surf/rings_centroids.csv", index = False)
        
        # # Bachmann_Bundle internal connection
        # la_connect_point, ra_connect_point = Method.get_connection_point_la_and_ra(la_appex_point)
        # la_connect_point = np.asarray(la_connect_point)
        #
        # path_start_end = np.vstack((appendage_basis, la_connect_point))
        # path_bb_ra = Method.creat_center_line(path_start_end)
        # tag = Method.assign_element_tag_around_path_within_radius(model, path_bb_ra, 2, tag, bachmann_bundel_left)
        # el = Method.assign_element_fiber_around_path_within_radius(model, path_bb_ra, 2, el, smooth=True)
    
        print("Creating bachmann bundles... done")
        el_epi = np.where(el_epi == [0,0,0], [1,0,0], el_epi).astype("float32")
    
        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))
        
        sheet_epi = np.cross(el_epi, et)
        sheet_epi = np.where(sheet_epi == [0,0,0], [1,0,0], sheet_epi).astype("float32")
        
        for i in range(epi.GetPointData().GetNumberOfArrays()-1, -1, -1):
            epi.GetPointData().RemoveArray(epi.GetPointData().GetArrayName(i))
        
        for i in range(epi.GetCellData().GetNumberOfArrays()-1, -1, -1):
            epi.GetCellData().RemoveArray(epi.GetCellData().GetArrayName(i))
    
        meshNew = dsa.WrapDataObject(epi)
        meshNew.CellData.append(tag_epi, "elemTag")
        meshNew.CellData.append(el_epi, "fiber")
        meshNew.CellData.append(sheet_epi, "sheet")
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_LA/LA_epi_with_fiber.vtk")
        writer.SetFileTypeToBinary()
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        pts = numpy_support.vtk_to_numpy(epi.GetPoints().GetData())
        with open(job.ID+'/result_LA/LA_epi_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
        with open(job.ID+'/result_LA/LA_epi_with_fiber.elem',"w") as f:
            f.write("{}\n".format(epi.GetNumberOfCells()))
            for i in range(epi.GetNumberOfCells()):
                cell = epi.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_epi[i]))
        
        with open(job.ID+'/result_LA/LA_epi_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el_epi)):
                f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))
                
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
        
        endo = Method.move_surf_along_normals(endo, 0.1*args.scale, 1) #  Warning: set -1 if pts normals are pointing outside
        bilayer = Method.generate_bilayer(endo, epi)
        
        Method.write_bilayer(bilayer, job)
        
    # ##### show the result #####
    # print("Visualizing in mayavi...")
    # vtkCenters = vtk.vtkCellCenters()
    # vtkCenters.SetInputData(model)
    # vtkCenters.Update()
    # centersOutput = vtkCenters.GetOutput().GetPoints().GetData()
    # center_points = vtk.util.numpy_support.vtk_to_numpy(centersOutput)

    # global mesh, cursor3d
    # fig = mlab.figure('Rush B, blyat!')
    # mlab.clf()
    # vector = mlab.quiver3d(center_points[:, 0], center_points[:, 1], center_points[:, 2], el[:, 0], el[:, 1], el[:, 2])
    # source = mlab.pipeline.open('Mesh/LA_mesh.vtk')  # Open the source
    # surf = mlab.pipeline.surface(source, color=(0.0, 0.0, 0.9))
    # mlab.show()

    # start_time = datetime.datetime.now()
    # print('Reading LA_with_lp_res_gradient.vtk... ' + str(start_time))
    # end_time = datetime.datetime.now()
    # running_time = end_time - start_time
    # print('Reading LA_with_lp_res_gradient.vtk......done ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    # en = k-(k*et)*et
    # abs_en = np.linalg.norm(en, axis=1, keepdims=True)
    # for i in range(len(abs_en)):
    #     if abs_en[i] == 0:
    #         abs_en[i] =1
    # en_norm = en/abs_en
    # el =
    # print(type(phie_grad))
    # print(abs_phie_grad)
    # phie_grad = r_grad.tolist()
    # print(len(ab_grad))
    # print(ab_grad[2])
