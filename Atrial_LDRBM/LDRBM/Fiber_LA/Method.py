import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
from scipy.spatial import cKDTree
from vtk.util import numpy_support
from scipy.spatial.distance import cosine
import collections

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]

def smart_reader(path):
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName(str(path))
    data_checker.Update()

    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()
    else:
        print("No polydata or unstructured grid")

    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()

    return output

def vtk_thr(model,mode,points_cells,array,thr1,thr2="None"):
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if mode == 0:
        thresh.ThresholdByUpper(thr1)
    elif mode == 1:
        thresh.ThresholdByLower(thr1)
    elif mode ==2:
        if int(vtk_version) >= 9:
            thresh.ThresholdBetween(thr1,thr2)
        else:
            thresh.ThresholdByUpper(thr1)
            thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_"+points_cells, array)
            thresh.Update()
            thr = thresh.GetOutput()
            thresh = vtk.vtkThreshold()
            thresh.SetInputData(thr)
            thresh.ThresholdByLower(thr2)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_"+points_cells, array)
    thresh.Update()
    
    output = thresh.GetOutput()
    
    return output
        
 
def mark_LA_endo_elemTag(model,tag,tao_mv,tao_lpv,tao_rpv,max_phie_ab_tau_lpv,max_phie_r2_tau_lpv):
    
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    thresh.ThresholdByUpper(tao_mv)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r")
    thresh.Update()
    
    MV_ids = vtk.util.numpy_support.vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))
    
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    thresh.ThresholdByUpper(max_phie_r2_tau_lpv+0.01)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r2")
    thresh.Update()
    
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(thresh.GetOutputPort())
    thresh.ThresholdByLower(max_phie_ab_tau_lpv+0.01)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_ab")
    thresh.Update()
    
    LAA_ids = vtk.util.numpy_support.vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))
    
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    thresh.ThresholdByLower(tao_lpv)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")
    thresh.Update()
    
    LPV_ids = vtk.util.numpy_support.vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))
    
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    thresh.ThresholdByUpper(tao_rpv)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")
    thresh.Update()
    
    RPV_ids = vtk.util.numpy_support.vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))
    
    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(tag, "elemTag")
    endo = meshNew.VTKObject

def move_surf_along_normals(mesh, eps, direction):

    extract_surf = vtk.vtkGeometryFilter()
    extract_surf.SetInputData(mesh)
    extract_surf.Update()
    
    # reverse = vtk.vtkReverseSense()
    # reverse.ReverseCellsOn()
    # reverse.ReverseNormalsOn()
    # reverse.SetInputConnection(extract_surf.GetOutputPort())
    # reverse.Update()
    
    # polydata = reverse.GetOutput()
    polydata = extract_surf.GetOutput()
    
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
    normalGenerator.AutoOrientNormalsOff()
    normalGenerator.SplittingOff() 
    normalGenerator.Update()
    
    PointNormalArray = numpy_support.vtk_to_numpy(normalGenerator.GetOutput().GetPointData().GetNormals())
    atrial_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    
    atrial_points = atrial_points + eps*direction*PointNormalArray
    
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_support.numpy_to_vtk(atrial_points))
    polydata.SetPoints(vtkPts)
    
    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(polydata)
    
    return mesh
    
def generate_bilayer(endo, epi):
    
    extract_surf = vtk.vtkGeometryFilter()
    extract_surf.SetInputData(epi)
    extract_surf.Update()
    
    reverse = vtk.vtkReverseSense()
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.SetInputConnection(extract_surf.GetOutputPort())
    reverse.Update()
    
    epi = vtk.vtkUnstructuredGrid()
    epi.DeepCopy(reverse.GetOutput())
            
    endo_pts = numpy_support.vtk_to_numpy(endo.GetPoints().GetData())
    epi_pts = numpy_support.vtk_to_numpy(epi.GetPoints().GetData())
    
    tree = cKDTree(endo_pts)
    dd, ii = tree.query(epi_pts)
    
    lines = vtk.vtkCellArray()
    
    for i in range(len(endo_pts)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,i);
        line.GetPointIds().SetId(1,len(endo_pts)+ii[i]);
        lines.InsertNextCell(line)
    
    points = np.concatenate((endo_pts, epi_pts[ii,:]), axis=0)
    polydata = vtk.vtkUnstructuredGrid()
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_support.numpy_to_vtk(points))
    polydata.SetPoints(vtkPts)
    polydata.SetCells(3, lines)
    
    fibers = np.zeros((len(endo_pts),3),dtype="float32")
    fibers[:,0] = 1
    
    tag = np.ones((len(endo_pts),1), dtype=int)
    tag[:,] = 100
    
    meshNew = dsa.WrapDataObject(polydata)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(fibers, "fiber")
    fibers = np.zeros((len(endo_pts),3),dtype="float32")
    fibers[:,1] = 1
    meshNew.CellData.append(fibers, "sheet")
    
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(endo)
    appendFilter.AddInputData(epi)
    appendFilter.AddInputData(meshNew.VTKObject)
    appendFilter.MergePointsOn()
    appendFilter.Update()
    
    bilayer = appendFilter.GetOutput()
    
    return bilayer

def write_bilayer(bilayer, args, job):
    
    if args.ofmt == 'vtk':
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_LA/LA_bilayer_with_fiber.vtk")
        writer.SetFileTypeToBinary()
    else:
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_LA/LA_bilayer_with_fiber.vtu")
    writer.SetInputData(bilayer)
    writer.Write()
    
    pts = numpy_support.vtk_to_numpy(bilayer.GetPoints().GetData())
    with open(job.ID+'/result_LA/LA_bilayer_with_fiber.pts',"w") as f:
        f.write("{}\n".format(len(pts)))
        for i in range(len(pts)):
            f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
    tag_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('elemTag'))

    with open(job.ID+'/result_LA/LA_bilayer_with_fiber.elem',"w") as f:
        f.write("{}\n".format(bilayer.GetNumberOfCells()))
        for i in range(bilayer.GetNumberOfCells()):
            cell = bilayer.GetCell(i)
            if cell.GetNumberOfPoints() == 2:
                f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_epi[i]))
            elif cell.GetNumberOfPoints() == 3:
                f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_epi[i]))
            elif cell.GetNumberOfPoints() == 4:
                f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_epi[i]))
    
    el_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('fiber'))
    sheet_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('sheet'))
    
    with open(job.ID+'/result_LA/LA_bilayer_with_fiber.lon',"w") as f:
        f.write("2\n")
        for i in range(len(el_epi)):
            f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))
                
                
def creat_tube_around_spline(points_data, radius):
    # Creat a points set
    spline_points = vtk.vtkPoints()
    for i in range(len(points_data)):
        spline_points.InsertPoint(i, points_data[i][0], points_data[i][1], points_data[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(10 * spline_points.GetNumberOfPoints())
    functionSource.Update()

    # Interpolate the scalars
    interpolatedRadius = vtk.vtkTupleInterpolator()
    interpolatedRadius.SetInterpolationTypeToLinear()
    interpolatedRadius.SetNumberOfComponents(1)

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tubeRadius.SetNumberOfTuples(n)
    tubeRadius.SetName("TubeRadius")

    # TODO make the radius variable???
    tMin = interpolatedRadius.GetMinimumT()
    tMax = interpolatedRadius.GetMaximumT()
    for i in range(n):
        t = (tMax - tMin) / (n - 1) * i + tMin
        r = radius
        # interpolatedRadius.InterpolateTuple(t, r)
        tubeRadius.SetTuple1(i, r)

    # Add the scalars to the polydata
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tubeRadius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes TODO: SidesShareVerticesOn()???
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(20)
    tuber.SidesShareVerticesOn()
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.SetCapping(0)
    tuber.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(tuber.GetOutput())
    triangle.Update()

    tuber = triangle
    return tuber


def dijkstra_path(polydata, StartVertex, EndVertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)
    # attention the return value will be reversed
    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk.util.numpy_support.vtk_to_numpy(points_data)
    return points_data


def dijkstra_path_on_a_plane(polydata, StartVertex, EndVertex, plane_point):
    point_start = np.asarray(polydata.GetPoint(StartVertex))
    point_end = np.asarray(polydata.GetPoint(EndVertex))
    point_third = plane_point

    v1 = point_start - point_end
    v2 = point_start - point_third
    norm = np.cross(v1, v2)
    #
    # # normlize norm
    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm / n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0][0], norm_1[0][1], norm_1[0][2])
    plane.SetOrigin(point_start[0], point_start[1], point_start[2])

    meshExtractFilter1 = vtk.vtkExtractGeometry()
    meshExtractFilter1.SetInputData(polydata)
    meshExtractFilter1.SetImplicitFunction(plane)
    meshExtractFilter1.Update()

    point_moved = point_start - 1.5 * norm_1
    # print(point_moved[0][0])
    plane2 = vtk.vtkPlane()
    plane2.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    plane2.SetOrigin(point_moved[0][0], point_moved[0][1], point_moved[0][2])

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(meshExtractFilter1.GetOutput())
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    band = meshExtractFilter2.GetOutput()
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(band)
    geo_filter.Update()
    band = geo_filter.GetOutput()
    # print(band)
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(band)
    loc.BuildLocator()
    StartVertex = loc.FindClosestPoint(point_start)
    EndVertex = loc.FindClosestPoint(point_end)

    points_data = dijkstra_path(band, StartVertex, EndVertex)
    return points_data


def creat_sphere(center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetThetaResolution(40)
    sphere.SetPhiResolution(40)
    sphere.SetRadius(radius)
    sphere.Update()
    return sphere


def creat_tube(center1, center2, radius):
    line = vtk.vtkLineSource()
    line.SetPoint1(center1[0], center1[1], center1[2])
    line.SetPoint2(center2[0], center2[1], center2[2])
    line.Update()

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line.GetOutput())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(20)
    tube.Update()
    return tube


def get_element_ids_around_path_within_radius(mesh, points_data, radius):
    
    gl_ids = vtk.util.numpy_support.vtk_to_numpy(mesh.GetCellData().GetArray('Global_ids'))
    
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))
    
    ids = []
    
    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        ids.append(gl_ids[index])

    return ids

def assign_element_tag_around_path_within_radius(mesh, points_data, radius, tag, element_tag):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = element_tag

    return tag


def normalize_vector(vector):
    abs = np.linalg.norm(vector)
    if abs != 0:
        vector_norm = vector / abs
    else:
        vector_norm = vector

    return vector_norm


def assign_element_fiber_around_path_within_radius(mesh, points_data, radius, fiber, smooth=True):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    if smooth:
        for i in range(len(points_data)):
            if i % 5 == 0 and i < 5:
                vector = points_data[5] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 5]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    else:
        for i in range(len(points_data)):
            if i < 1:
                vector = points_data[1] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 1]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    return fiber


def get_mean_point(data):
    ring_points = data.GetPoints().GetData()
    ring_points = vtk.util.numpy_support.vtk_to_numpy(ring_points)
    center_point = [np.mean(ring_points[:, 0]), np.mean(ring_points[:, 1]), np.mean(ring_points[:, 2])]
    center_point = np.array(center_point)
    return center_point


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_intersect_bool(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    if len(intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])) == 0:
        res = 0
    else:
        res = 1
    return res


def get_ct_end_points_id(endo, ct, scv, icv):
    # endo
    points_data = endo.GetPoints().GetData()
    endo_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # ct
    points_data = ct.GetPoints().GetData()
    ct_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # scv
    points_data = scv.GetPoints().GetData()
    scv_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # icv
    points_data = icv.GetPoints().GetData()
    icv_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # intersection
    # inter_ct_endo = multidim_intersect(endo_points, ct_points)
    # inter_icv = multidim_intersect(inter_ct_endo, icv_points)
    # inter_scv = multidim_intersect(inter_ct_endo, scv_points)`
    inter_icv = multidim_intersect(ct_points, icv_points)
    inter_scv = multidim_intersect(ct_points, scv_points)

    # calculating mean point
    path_icv = np.asarray([np.mean(inter_icv[:, 0]), np.mean(inter_icv[:, 1]), np.mean(inter_icv[:, 2])])
    path_scv = np.asarray([np.mean(inter_scv[:, 0]), np.mean(inter_scv[:, 1]), np.mean(inter_scv[:, 2])])

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_ct_id_icv = loc.FindClosestPoint(path_icv)
    path_ct_id_scv = loc.FindClosestPoint(path_scv)

    return path_ct_id_icv, path_ct_id_scv


def get_tv_end_points_id(endo, ra_tv_s_surface, ra_ivc_surface, ra_svc_surface, ra_tv_surface):
    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_tv_s_surface.vtk')
    # reader.Update()
    # tv_s = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_ivc_surface.vtk')
    # reader.Update()
    # tv_ivc = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_svc_surface.vtk')
    # reader.Update()
    # tv_svc = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_tv_surface.vtk')
    # reader.Update()
    # tv = reader.GetOutput()

    tv_center = get_mean_point(ra_tv_surface)
    tv_ivc_center = get_mean_point(ra_ivc_surface)
    tv_svc_center = get_mean_point(ra_svc_surface)

    v1 = tv_center - tv_ivc_center
    v2 = tv_center - tv_svc_center
    norm = np.cross(v1, v2)

    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm / n
    moved_center = tv_center - norm_1 * 5

    plane = vtk.vtkPlane()
    plane.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    plane.SetOrigin(moved_center[0][0], moved_center[0][1], moved_center[0][2])

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(ra_tv_s_surface)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(meshExtractFilter.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(connect.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_1 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(connect.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_2 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])
    dis_1 = np.linalg.norm(center_point_1 - tv_ivc_center)
    dis_2 = np.linalg.norm(center_point_1 - tv_svc_center)
    # print(dis_1)
    # print(dis_2)
    if dis_1 < dis_2:
        center_point_icv = center_point_1
        center_point_scv = center_point_2
    else:
        center_point_icv = center_point_2
        center_point_scv = center_point_1
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_tv_id_icv = loc.FindClosestPoint(center_point_icv)
    path_tv_id_scv = loc.FindClosestPoint(center_point_scv)

    return path_tv_id_icv, path_tv_id_scv


def extract_largest_region(mesh):
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(mesh)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    surface = connect.GetOutput()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(surface)
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    res = cln.GetOutput()

    return res


def assign_ra_appendage(model, SCV, appex_point, tag):
    appex_point = np.asarray(appex_point)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(model)
    locator.BuildLocator()

    locator2 = vtk.vtkStaticPointLocator()
    locator2.SetDataSet(SCV)
    locator2.BuildLocator()
    SCV_id = locator2.FindClosestPoint(appex_point)
    SCV_closed_point = SCV.GetPoint(SCV_id)
    radius = np.linalg.norm(appex_point - SCV_closed_point)
    print(radius)

    mesh_point_temp_id_list = vtk.vtkIdList()
    locator.FindPointsWithinRadius(radius, appex_point, mesh_point_temp_id_list)
    print(mesh_point_temp_id_list.GetNumberOfIds())
    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_point_temp_id_list.GetNumberOfIds()):
        model.GetPointCells(mesh_point_temp_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = 59

    return tag


def get_endo_ct_intersection_cells(endo, ct):
    points_data = ct.GetPoints().GetData()
    ct_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    points_data = endo.GetPoints().GetData()
    endo_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    intersection = multidim_intersect(ct_points, endo_points)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    endo_id_list = []
    for i in range(len(intersection)):
        endo_id_list.append(loc.FindClosestPoint(intersection[i]))
    endo_cell_id_list = vtk.vtkIdList()
    endo_cell_temp_id_list = vtk.vtkIdList()
    for i in range(len(endo_id_list)):
        endo.GetPointCells(endo_id_list[i], endo_cell_temp_id_list)
        for j in range(endo_cell_temp_id_list.GetNumberOfIds()):
            endo_cell_id_list.InsertNextId(endo_cell_temp_id_list.GetId(j))
    print(endo_cell_id_list.GetNumberOfIds())
    extract = vtk.vtkExtractCells()
    extract.SetInputData(endo)
    extract.SetCellList(endo_cell_id_list)
    extract.Update()
    endo_ct = extract.GetOutput()

    return endo_ct


def get_connection_point_la_and_ra(appen_point):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_rpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_rpv_inf_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    la_epi_surface = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')
    ra_epi_surface = smart_reader('../../Generate_Boundaries/RA/result/ra_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    point_1_id = loc_mv.FindClosestPoint(appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    loc_rpv_inf = vtk.vtkPointLocator()
    loc_rpv_inf.SetDataSet(la_rpv_inf_surface)
    loc_rpv_inf.BuildLocator()
    point_2_id = loc_rpv_inf.FindClosestPoint(appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    loc_endo = vtk.vtkPointLocator()
    loc_endo.SetDataSet(endo)
    loc_endo.BuildLocator()

    point_1_id_endo = loc_endo.FindClosestPoint(point_1)
    point_2_id_endo = loc_endo.FindClosestPoint(point_2)

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(endo)
    geo_filter.Update()

    bb_aux_l_points = dijkstra_path(geo_filter.GetOutput(), point_1_id_endo, point_2_id_endo)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    geo_filter_la = vtk.vtkGeometryFilter()
    geo_filter_la.SetInputData(la_epi_surface)
    geo_filter_la.Update()
    la_epi_surface = geo_filter_la.GetOutput()

    geo_filter_ra = vtk.vtkGeometryFilter()
    geo_filter_ra.SetInputData(ra_epi_surface)
    geo_filter_ra.Update()
    ra_epi_surface = geo_filter_ra.GetOutput()

    loc_la_epi = vtk.vtkPointLocator()
    loc_la_epi.SetDataSet(la_epi_surface)
    loc_la_epi.BuildLocator()

    loc_ra_epi = vtk.vtkPointLocator()
    loc_ra_epi.SetDataSet(ra_epi_surface)
    loc_ra_epi.BuildLocator()

    la_connect_point_id = loc_la_epi.FindClosestPoint(la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = loc_ra_epi.FindClosestPoint(la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point

def point_array_mapper(mesh1, mesh2, mesh2_name, idat):
    
    pts1 = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPoints().GetData())
    pts2 = vtk.util.numpy_support.vtk_to_numpy(mesh2.GetPoints().GetData())
    
    tree = cKDTree(pts1)

    dd, ii = tree.query(pts2, n_jobs=-1)
    
    meshNew = dsa.WrapDataObject(mesh2)
    if idat == "all":
        for i in range(mesh1.GetPointData().GetNumberOfArrays()):
            data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPointData().GetArray(mesh1.GetPointData().GetArrayName(i)))
            if isinstance(data[0], collections.Sized):
                data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)
            
            data2 = data[ii]
            data2 = np.where(np.isnan(data2), 10000, data2)
            # ghosts = np.zeros(meshNew.GetNumberOfPoints(), dtype=np.uint8)
            # ghosts[1] = vtk.vtkDataSetAttributes.DUPLICATEPOINT
            # meshNew.PointData.append(ghosts, vtk.vtkDataSetAttributes.GhostArrayName())
            # assert algs.make_point_mask_from_NaNs(meshNew, data2)[1] == vtk.vtkDataSetAttributes.DUPLICATEPOINT | vtk.vtkDataSetAttributes.HIDDENPOINT
            meshNew.PointData.append(data2, mesh1.GetPointData().GetArrayName(i))
    else:
        data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPointData().GetArray(idat))
        if isinstance(data[0], collections.Sized):
            data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)
        
        data2 = data[ii]
        meshNew.PointData.append(data2, idat)
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("{}_with_data.vtk".format(mesh2_name.split('.')[0]))
    writer.SetFileTypeToBinary()
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

def cell_array_mapper(mesh1, mesh2, mesh2_name, idat):
    
    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(mesh1)
    filter_cell_centers.Update()
    centroids1 = filter_cell_centers.GetOutput().GetPoints()
    centroids1_array = vtk.util.numpy_support.vtk_to_numpy(centroids1.GetData())
    
    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(mesh2)
    filter_cell_centers.Update()
    centroids2 = filter_cell_centers.GetOutput().GetPoints()
    pts2 = vtk.util.numpy_support.vtk_to_numpy(centroids2.GetData())
    
    tree = cKDTree(centroids1_array)

    dd, ii = tree.query(pts2, n_jobs=-1)
    
    meshNew = dsa.WrapDataObject(mesh2)
    if idat == "all":
        for i in range(mesh1.GetCellData().GetNumberOfArrays()):
            data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetCellData().GetArray(mesh1.GetCellData().GetArrayName(i)))
            if isinstance(data[0], collections.Sized):
                data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)
            
            data2 = data[ii]
            meshNew.PointData.append(data2, mesh1.GetCellData().GetArrayName(i))
    else:
        data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetCellData().GetArray(idat))
        if isinstance(data[0], collections.Sized):
            data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)
        
        data2 = data[ii]
        meshNew.CellData.append(data2, idat)
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("{}_with_data.vtk".format(mesh2_name.split('.')[0]))
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    return data2

def get_bachmann_path_left(appendage_basis, lpv_sup_basis):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    epi = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    loc_epi = vtk.vtkPointLocator()
    loc_epi.SetDataSet(epi)
    loc_epi.BuildLocator()

    appendage_basis_id = loc_epi.FindClosestPoint(appendage_basis)
    lpv_sup_basis_id = loc_epi.FindClosestPoint(lpv_sup_basis)

    left_inf_pv_center = get_mean_point(la_lpv_inf_surface)
    point_l1_id = loc_mv.FindClosestPoint(left_inf_pv_center)
    point_l1 = la_mv_surface.GetPoint(point_l1_id)
    bb_mv_id = loc_epi.FindClosestPoint(point_l1)

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(epi)
    geo_filter.Update()

    bb_1_points = dijkstra_path(geo_filter.GetOutput(), lpv_sup_basis_id, appendage_basis_id)
    bb_2_points = dijkstra_path(geo_filter.GetOutput(), appendage_basis_id, bb_mv_id)
    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points), axis=0)

    return bb_left, appendage_basis

def compute_wide_BB_path_left(epi, df, left_atrial_appendage_epi, mitral_valve_epi):
    # Extract the LAA
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(epi)
    thresh.ThresholdBetween(left_atrial_appendage_epi, left_atrial_appendage_epi)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")
    thresh.Update()
    
    LAA = thresh.GetOutput()
    
    min_r2_cell_LAA = np.argmin(vtk.util.numpy_support.vtk_to_numpy(LAA.GetCellData().GetArray('phie_r2')))
    
    ptIds = vtk.vtkIdList()
  
    LAA.GetCellPoints(min_r2_cell_LAA, ptIds)
    
    sup_appendage_basis_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])
    
    max_r2_cell_LAA = np.argmax(vtk.util.numpy_support.vtk_to_numpy(LAA.GetCellData().GetArray('phie_r2')))
    
    ptIds = vtk.vtkIdList()
  
    LAA.GetCellPoints(max_r2_cell_LAA, ptIds)
    
    bb_mv_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])
    
    max_v_cell_LAA = np.argmax(vtk.util.numpy_support.vtk_to_numpy(LAA.GetCellData().GetArray('phie_v')))
    
    ptIds = vtk.vtkIdList()
  
    LAA.GetCellPoints(max_v_cell_LAA, ptIds)
    
    inf_appendage_basis_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])
    
    # Extract the border of the LAA
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputConnection(thresh.GetOutputPort())
    geo_filter.Update()
    
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(geo_filter.GetOutput())
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()
    
    LAA_border = boundaryEdges.GetOutput()
    LAA_pts_border = vtk.util.numpy_support.vtk_to_numpy(LAA_border.GetPoints().GetData())
    max_dist = 0
    for i in range(len(LAA_pts_border)):
        if np.sqrt(np.sum((LAA_pts_border[i]-df["LIPV"].to_numpy())**2, axis=0)) > max_dist:
            max_dist = np.sqrt(np.sum((LAA_pts_border[i]-df["LIPV"].to_numpy())**2, axis=0))
            LAA_pt_far_from_LIPV = LAA_pts_border[i]

    # Extract the MV
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(epi)
    thresh.ThresholdBetween(mitral_valve_epi, mitral_valve_epi)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")
    thresh.Update()
    
    MV = thresh.GetOutput()
    # MV_pts = vtk.util.numpy_support.vtk_to_numpy(MV.GetPoints().GetData())
    
    # tree = cKDTree(LAA_pts)
    # max_dist = np.sqrt(np.sum((df["MV"].to_numpy()-df["LAA"].to_numpy())**2, axis=0))
    # dd, ii = tree.query(MV_pts, distance_upper_bound=max_dist)
    
    # inf_appendage_basis_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ii[np.argmin(dd)])[0])
    
    # Get the closest point to the inferior appendage base in the MV
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(MV)
    loc.BuildLocator()
    
    bb_mv_id = int(MV.GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(epi.GetPoint(bb_mv_id)))[0])
    
    if int(vtk_version) >= 9:
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(epi)
        thresh.ThresholdBetween(left_atrial_appendage_epi, left_atrial_appendage_epi)
        thresh.InvertOn()
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")
        thresh.Update()
    else:
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(epi)
        thresh.ThresholdByLower(left_atrial_appendage_epi-1)
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")
        thresh.Update()

        low_LAA = thresh.GetOutput()

        thresh = vtk.vtkThreshold()
        thresh.SetInputData(epi)
        thresh.ThresholdByUpper(left_atrial_appendage_epi+1)
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")
        thresh.Update()

        up_LAA = thresh.GetOutput()

        thresh = vtk.vtkAppendFilter()
        thresh.AddInputData(low_LAA)
        thresh.AddInputData(up_LAA)
        thresh.MergePointsOn()
        thresh.Update()

    inf_appendage_basis_id = get_in_surf1_closest_point_in_surf2(thresh.GetOutput(), epi, inf_appendage_basis_id)
    sup_appendage_basis_id = get_in_surf1_closest_point_in_surf2(thresh.GetOutput(), epi, sup_appendage_basis_id)
    bb_mv_id = get_in_surf1_closest_point_in_surf2(thresh.GetOutput(), epi, bb_mv_id)
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(thresh.GetOutput())
    loc.BuildLocator()
    LAA_pt_far_from_LIPV_id = loc.FindClosestPoint(LAA_pt_far_from_LIPV)

    bb_left = get_wide_bachmann_path_left(thresh.GetOutput(), inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id, LAA_pt_far_from_LIPV_id)
    #bb_left = savgol_filter(bb_left, 5, 2, mode='interp', axis =0)
    
    # new = []
    # diff = 1000
    # for n in range(len(bb_left)):
    #     if not new or np.sqrt(np.sum((np.array(bb_left[n])-np.array(new[-1]))**2, axis=0)) >= diff:
    #         new.append(bb_left[n])
    # new = np.array(new)

    # tck, u = interpolate.splprep([new[:,0],new[:,1],new[:,2]], s=100)
    # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    
    #u_fine = np.linspace(0,1,len(bb_left))
    #x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    # bb_left = np.array([x_knots, y_knots, z_knots]).T
    #bb_left = np.array([x_fine, y_fine, z_fine]).T
    
    
    
    
    #u_fine = np.linspace(0,1,len(bb_left))
    #x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    #new_points = splev(u, tck)
    #window_width = 1000
    #cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    #ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    # pts = vtk.vtkPoints()
    # for n in range(len(bb_left)):
    #     pts.InsertNextPoint(bb_left[n])
    
    # smoothingIterations = 15
    # passBand = 0.001
    # featureAngle = 120.0
    
    # smoother = vtk.vtkWindowedSincPolyDataFilter()
    # smoother.SetInputConnection(discrete->GetOutputPort())
    # smoother.SetNumberOfIterations(smoothingIterations)
    # smoother.BoundarySmoothingOff()
    # smoother.FeatureEdgeSmoothingOff()
    # smoother.SetFeatureAngle(featureAngle)
    # smoother.SetPassBand(passBand)
    # smoother.NonManifoldSmoothingOn()
    # smoother.NormalizeCoordinatesOn()
    # smoother.Update()
  
    # Smooth the path fitting a spline?
    # new = []
    # diff = 1000
    # for n in range(len(bb_left)):
    #     if not new or np.sqrt(np.sum((np.array(bb_left[n])-np.array(new[-1]))**2, axis=0)) >= diff:
    #         new.append(bb_left[n])
    # new = np.array(new)
    # bb_left = np.array(new)
    # n = bb_left.shape[0]
    # new = []  # change dtype if you need to
    # tree = cKDTree(bb_left)
    # for i in range(n):
    #     neighbors = tree.query_ball_point(bb_left[i], 1000)
    #     temp = [bb_left[k] for k in neighbors]
    #     new.append(np.mean(temp, axis =0))

    # # mesh.VPos = final
    # bb_left = np.array(new)
    
    return bb_left, thresh.GetOutput().GetPoint(inf_appendage_basis_id), thresh.GetOutput().GetPoint(sup_appendage_basis_id), thresh.GetOutput().GetPoint(LAA_pt_far_from_LIPV_id)

def get_in_surf1_closest_point_in_surf2(surf1, surf2, pt_id_in_surf2):
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(surf1)
    loc.BuildLocator()
    return loc.FindClosestPoint(surf2.GetPoint(pt_id_in_surf2))
    

def get_wide_bachmann_path_left(epi, inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id, LAA_pt_far_from_LIPV_id):
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(epi)
    geo_filter.Update()
    
    bb_1_points = dijkstra_path(geo_filter.GetOutput(), sup_appendage_basis_id, LAA_pt_far_from_LIPV_id)
    bb_2_points = dijkstra_path(geo_filter.GetOutput(), LAA_pt_far_from_LIPV_id, inf_appendage_basis_id)
    bb_3_points = dijkstra_path(geo_filter.GetOutput(), inf_appendage_basis_id, bb_mv_id)

    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points, bb_3_points), axis=0)
    # bb_left = dijkstra_path(geo_filter.GetOutput(), sup_appendage_basis_id, bb_mv_id)

    return bb_left

def creat_center_line(start_end_point):
    spline_points = vtk.vtkPoints()
    for i in range(len(start_end_point)):
        spline_points.InsertPoint(i, start_end_point[i][0], start_end_point[i][1], start_end_point[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
    tubePolyData = functionSource.GetOutput()
    points = tubePolyData.GetPoints().GetData()
    points = vtk.util.numpy_support.vtk_to_numpy(points)

    return points


def smart_bridge_writer(tube, sphere_1, sphere_2, name, job):
    meshNew = dsa.WrapDataObject(tube.GetOutput())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_tube.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

    meshNew = dsa.WrapDataObject(sphere_1.GetOutput())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_sphere_1.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

    meshNew = dsa.WrapDataObject(sphere_2.GetOutput())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_sphere_2.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
def find_tau(model, ub, lb, low_up, scalar):
    k = 1
    while ub - lb > 0.01:
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(model)
        if low_up == "low":
            thresh.ThresholdByLower((ub + lb) / 2)
        else:
            thresh.ThresholdByUpper((ub + lb) / 2)
        thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", scalar)
        thresh.Update()

        connect = vtk.vtkConnectivityFilter()
        connect.SetInputData(thresh.GetOutput())
        connect.SetExtractionModeToAllRegions()
        connect.Update()
        num = connect.GetNumberOfExtractedRegions()
        
        print("Iteration: ", k)
        print("Value of tao: ", (ub + lb) / 2)
        print("Number of regions: ", num,  "\n")
        
        if low_up == "low":
            if num == 1:
                ub = (ub + lb) / 2
            elif num > 1:
                lb = (ub + lb) / 2
        else:
            if num == 1:
                lb = (ub + lb) / 2
            elif num > 1:
                ub = (ub + lb) / 2

        k += 1
    
    if low_up == "low":
        return lb
    else:
        return ub

def distinguish_PVs(connect, PVs, df, name1, name2):
    num = connect.GetNumberOfExtractedRegions()
    connect.SetExtractionModeToSpecifiedRegions()
    
    centroid1 = df[name1].to_numpy()
    centroid2 = df[name2].to_numpy()
    
    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        single_PV = connect.GetOutput()
        
        # Clean unused points
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(single_PV)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()
        
        if name1.startswith("L"):
            phie_v = np.max(vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))
        elif name1.startswith("R"):
            phie_v = np.min(vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))

        if name1.startswith("L") and phie_v>0.025:
            found, val = optimize_shape_PV(surface, 10, 0)
            if found:
                single_PV = vtk_thr(single_PV, 1,"CELLS", "phie_v", val)
                geo_filter = vtk.vtkGeometryFilter()
                geo_filter.SetInputData(single_PV)
                geo_filter.Update()
                surface = geo_filter.GetOutput()
        elif name1.startswith("R") and phie_v<0.975:
            found, val = optimize_shape_PV(surface, 10, 1)
            if found:
                single_PV = vtk_thr(single_PV, 0,"CELLS", "phie_v", val)
                geo_filter = vtk.vtkGeometryFilter()
                geo_filter.SetInputData(single_PV)
                geo_filter.Update()
                surface = geo_filter.GetOutput()
        
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(surface)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        
        c_mass = centerOfMassFilter.GetCenter()
        
        centroid1_d = np.sqrt(np.sum((np.array(centroid1) - np.array(c_mass))**2, axis=0))
        centroid2_d = np.sqrt(np.sum((np.array(centroid2) - np.array(c_mass))**2, axis=0))
        
        if centroid1_d < centroid2_d:
            PVs[name1] = vtk.util.numpy_support.vtk_to_numpy(single_PV.GetCellData().GetArray('Global_ids'))
        else:
            PVs[name2] = vtk.util.numpy_support.vtk_to_numpy(single_PV.GetCellData().GetArray('Global_ids'))
        
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
 
    return PVs

def optimize_shape_PV(surface, num, bound):
    
    if bound == 0:
        phie_v = np.max(vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))
    else:
        phie_v = np.min(vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))
    
    arr = np.linspace(bound, phie_v, num)

    c_mass_l = []
    found = 0
    for l in range(num-1):
        if bound == 0:
            out = vtk_thr(surface, 2,"CELLS", "phie_v", arr[l], arr[l+1])
        else:
            out = vtk_thr(surface, 2,"CELLS", "phie_v", arr[l+1], arr[l])
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(out)
        geo_filter.Update()
        
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(geo_filter.GetOutput())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        
        c_mass_l.append(centerOfMassFilter.GetCenter())

    v1 = np.array(c_mass_l[0])-np.array(c_mass_l[1])
    for l in range(1,num-2):
        v2 = np.array(c_mass_l[l])-np.array(c_mass_l[l+1])
        if 1-cosine(v1, v2) < 0:
            found = 1
            break
    
    return found, arr[l-1]