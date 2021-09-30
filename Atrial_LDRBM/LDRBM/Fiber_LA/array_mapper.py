#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:13:35 2021

@author: luca
"""
import vtk
import Method
from carputils import tools
from vtk.numpy_interface import dataset_adapter as dsa

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments    
    parser.add_argument('--mesh1',
                    type=str,
                    default="",
                    help='path to meshname')
    parser.add_argument('--mesh2',
                    type=str,
                    default="",
                    help='path to meshname')
    parser.add_argument('--idat',
                    type=str,
                    default="",
                    help='input mesh format')
    parser.add_argument('--odat',
                    type=str,
                    default="",
                    help='input mesh format')
    parser.add_argument('--pts_or_cells',
                        default='points',
                        choices=['points',
                                 'cells'],
                        help='Mesh type')

    return parser

def jobID(args):
    ID = '{}_fibers'.format(args.mesh1.split('/')[-1])
    return ID

@tools.carpexample(parser, jobID)
def run(args, job):
    
    mesh1 = Method.smart_reader(args.mesh1)
    
    mesh2 = Method.smart_reader(args.mesh2)
    
    if args.pts_or_cells == "points":
        Method.point_array_mapper(mesh1, mesh2, args.mesh2, args.idat)
    else:
        Method.cell_array_mapper(mesh1, mesh2, args.mesh2, args.idat)
    # data2_list = []
    # if args.pts_or_cells == "points":
    #     if args.idat == "all":
    #         for i in range(mesh1.GetPointData().GetNumberOfArrays()):
    #             data2_list.append(Method.point_array_mapper(mesh1, mesh2, mesh1.GetPointData().GetArrayName(i)))
    #     else:
    #         data2_list.append(Method.point_array_mapper(mesh1, mesh2, args.idat))
    # else:
    #     if args.idat == "all":
    #         for i in range(mesh1.GetCellData().GetNumberOfArrays()):
    #             data2_list.append(Method.cell_array_mapper(mesh1, mesh2, mesh1.GetCellData().GetArrayName(i)))
    #     else:
    #         data2_list.append(Method.cell_array_mapper(mesh1, mesh2, args.idat))
        
    # meshNew = dsa.WrapDataObject(mesh2)
    
    # for i in range(len(data2_list)):
    #     if args.pts_or_cells == "points":
    #         if args.idat == "all":
    #             meshNew.PointData.append(data2_list[i], mesh1.GetPointData().GetArrayName(i))
    #         else:    
    #             meshNew.PointData.append(data2_list[i], args.odat)
    #     else:
    #         if args.idat == "all":
    #             meshNew.CellData.append(data2_list[i], mesh1.GetCellData().GetArrayName(i))
    #         else:    
    #             meshNew.CellData.append(data2_list[i], args.odat)
    # writer = vtk.vtkUnstructuredGridWriter()
    # writer.SetFileName("{}_with_data.vtk".format(args.mesh2.split('.')[0]))
    # writer.SetInputData(meshNew.VTKObject)
    # writer.Write()
    
if __name__ == '__main__':
    run()