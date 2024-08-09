import numpy as np
import scipy.io as sio
from scipy import sparse
import scipy.linalg
import platform
import copy
from scipy import spatial
import os

from . import nirfasteruff_cpu

if nirfasteruff_cpu.isCUDA():
    from . import nirfasteruff_cuda

class utils:
    '''
    Dummy class holding some helper functions and helper classes  
    Dummy class used so the function hierarchy can be compatible with the full version
    '''
    def isCUDA():
        '''
        Checks if system has a CUDA device with compute capability >=5.2  
        On a Mac machine, it automatically returns False without checking  
                
        Input: None  
        Output: bool value of whether compatible CUDA device exists  
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        return nirfasteruff_cpu.isCUDA()
    
    def get_solver():
        '''
        Get the default solver. If isCUDA is true, returns GPU, otherwise CPU
        
        solver = get_solver()
        
        Input: None
        Output: (string) 'CPU' or 'GPU'
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if utils.isCUDA():
            solver = 'GPU'
        else:
            solver = 'CPU'     
        return solver            
    
    def pointLocation(mesh, pointlist):
        '''
        Similar to Matlab's pointLocation function, queries which elements in mesh the points belong to.  
        Also calculate the barycentric coordinates.  
        
        ind, int_func = pointLocation(mesh, pointlist)  
        
        Input:  
            - mesh: A nirfasteruff.base.stnd_mesh object. Can be 2D or 3D  
            - pointlist: A list of points to query. Should be a double NumPy array of shape (N, dim),
                        where N is number of points
        Output:  
            - ind: (double NumPy array) i-th queried point is in element ind[i] of mesh (zero-based).  
                    If not in mesh, ind[i]=-1. Size: (N,)  
            - int_func: (double NumPy array) i-th row is the barycentric coordinates of i-th queried point  
                    If not in mesh, corresponding row is all zero. Size: (N, dim+1)  
                    
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        ind, int_func = nirfasteruff_cpu.pointLocation(mesh.elements, mesh.nodes, np.atleast_2d(pointlist))
        return ind, int_func
    
    def check_element_orientation_2d(ele, nodes):
        '''
        Make sure the 2D triangular elements are oriented in CCW.
        This is a direct translation from the Matlab version.
        
        ele2 = check_element_orientation_2d(ele, nodes)
        
        Input:
            ele: (NumPy array) Elements in a 2D mesh. Size: (N, 3)
            nodes: (NumPy array) Node locations in a 2D mesh. Size: (N, 2)
        Output:
            ele2: (NumPy array) Re-oriented element list
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if ele.shape[1] != 3:
            raise TypeError('check_element_orientation_2d expects a 2D triangular mesh!')
        if nodes.shape[1] == 2:
            nodes = np.c_[nodes, np.zeros(nodes.shape[0])]
        v1 = nodes[np.int32(ele[:,1]-1),:] - nodes[np.int32(ele[:,0]-1),:]
        v2 = nodes[np.int32(ele[:,2]-1),:] - nodes[np.int32(ele[:,0]-1),:]
        
        z = np.cross(v1, v2)
        idx = z[:,2]<0
        if np.any(idx):
            ele[np.ix_(idx, [0,1])] = ele[np.ix_(idx, [1,0])]
        return ele
    
    def pointLineDistance(A, B, p):
        '''
        Calculate the distance between a point and a line (defined by two point), and find the projection point
        This is a direct translation  from the Matlab version
        
        dist, point = pointLineDistance(A, B, p)
        
        Input:
            A: first point on the line
            B: second point on the line
            p: point of query
        Output:
            dist: point-line distant
            point: projection point
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        t = np.dot(p-A, B-A) / np.dot(B-A, B-A)
        if t<0:
            t = 0
        elif t>1:
            t = 1
        
        point = A + (B-A)*t
        dist = np.norm(p - point)     
        return dist, point
    
    def pointTriangleDistance(TRI, P):
        '''
        Joshua Shafer's translation: https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e
        function [dist,PP0] = pointTriangleDistance(TRI,P)
        calculate distance between a point and a triangle in 3D
        SYNTAX
          dist = pointTriangleDistance(TRI,P)
          [dist,PP0] = pointTriangleDistance(TRI,P)
        
        DESCRIPTION
          Calculate the distance of a given point P from a triangle TRI.
          Point P is a row vector of the form 1x3. The triangle is a matrix
          formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
          dist = pointTriangleDistance(TRI,P) returns the distance of the point P
          to the triangle TRI.
          [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
          closest point PP0 to P on the triangle TRI.
        
        Author: Gwendolyn Fischer
        Release: 1.0
        Release date: 09/02/02
        Release: 1.1 Fixed Bug because of normalization
        Release: 1.2 Fixed Bug because of typo in region 5 20101013
        Release: 1.3 Fixed Bug because of typo in region 2 20101014
    
        Possible extention could be a version tailored not to return the distance
        and additionally the closest point, but instead return only the closest
        point. Could lead to a small speed gain.
    
       
        The algorithm is based on
        "David Eberly, 'Distance Between Point and Triangle in 3D',
        Geometric Tools, LLC, (1999)"
        http://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
        s
        '''
        
        B = TRI[0, :]
        E0 = TRI[1, :] - B
        # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
        E1 = TRI[2, :] - B
        # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
        D = B - P
        a = np.dot(E0, E0)
        b = np.dot(E0, E1)
        c = np.dot(E1, E1)
        d = np.dot(E0, D)
        e = np.dot(E1, D)
        f = np.dot(D, D)
    
        #print "{0} {1} {2} ".format(B,E1,E0)
        det = a * c - b * b
        s = b * e - c * d
        t = b * d - a * e
    
        # Terible tree of conditionals to determine in which region of the diagram
        # shown above the projection of the point into the triangle-plane lies.
        if (s + t) <= det:
            if s < 0.0:
                if t < 0.0:
                    # region4
                    if d < 0:
                        t = 0.0
                        if -d >= a:
                            s = 1.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                    else:
                        s = 0.0
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            if -e >= c:
                                t = 1.0
                                sqrdistance = c + 2.0 * e + f
                            else:
                                t = -e / c
                                sqrdistance = e * t + f
    
                                # of region 4
                else:
                    # region 3
                    s = 0
                    if e >= 0:
                        t = 0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 3
            else:
                if t < 0:
                    # region 5
                    t = 0
                    if d >= 0:
                        s = 0
                        sqrdistance = f
                    else:
                        if -d >= a:
                            s = 1
                            sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                else:
                    # region 0
                    invDet = 1.0 / det
                    s = s * invDet
                    t = t * invDet
                    sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
        else:
            if s < 0.0:
                # region 2
                tmp0 = b + d
                tmp1 = c + e
                if tmp1 > tmp0:  # minimum on edge s+t=1
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f
    
                else:  # minimum on edge s=0
                    s = 0.0
                    if tmp1 <= 0.0:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 2
            else:
                if t < 0.0:
                    # region6
                    tmp0 = b + e
                    tmp1 = a + d
                    if tmp1 > tmp0:
                        numer = tmp1 - tmp0
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            t = 1.0
                            s = 0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = numer / denom
                            s = 1 - t
                            sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    
                    else:
                        t = 0.0
                        if tmp1 <= 0.0:
                            s = 1
                            sqrdistance = a + 2.0 * d + f
                        else:
                            if d >= 0.0:
                                s = 0.0
                                sqrdistance = f
                            else:
                                s = -d / a
                                sqrdistance = d * s + f
                else:
                    # region 1
                    numer = c + e - b - d
                    if numer <= 0:
                        s = 0.0
                        t = 1.0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            s = 1.0
                            t = 0.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = numer / denom
                            t = 1 - s
                            sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    
        # account for numerical round-off error
        if sqrdistance < 0:
            sqrdistance = 0
    
        dist = np.sqrt(sqrdistance)
    
        PP0 = B + s * E0 + t * E1
        return dist, PP0
    
    class SolverOptions:
        '''
        Contains the parameters used by the FEM solvers, Equivalent to 'solver_options' in the Matlab version
        
        max_iter (default=1000): maximum number of iterations allowed
        AbsoluteTolerance (default=1e-12): Absolute tolerance for convergence
        RelativeTolerance (default=1e-12): Relative (to the initial residual norm) tolerance for convergence
        divergence (default=1e8): Stop the solver when residual norm greater than this value
        GPU (default=-1): GPU selection. -1 for automatic, 0, 1, ... for manual selection on multi-GPU systems
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self, max_iter = 1000, AbsoluteTolerance = 1e-12, RelativeTolerance = 1e-12, divergence = 1e8, GPU = -1):
            self.max_iter = max_iter
            self.AbsoluteTolerance = AbsoluteTolerance
            self.RelativeTolerance = RelativeTolerance
            self.divergence = divergence
            self.GPU = GPU
            
    class ConvergenceInfo:
        '''
        Convergence information of the FEM solvers. Only used as a return type of functions nirfasteruff.math.get_field_*
        Constructed using the output of the internal C++ functions
        
        Fields:
            isConverged: if solver converged to relative tolerance, for each rhs
            isConvergedToAbsoluteTolerance: if solver converged to absolute tolerance, for each rhs
            iteration: iterations taken to converge, for each rhs
            residual: final residual, for each rhs
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self, info = None):
            self.isConverged = []
            self.isConvergedToAbsoluteTolerance = []
            self.iteration = []
            self.residual = []
            if info != None:
                for item in info:
                    self.isConverged.append(item.isConverged)
                    self.isConvergedToAbsoluteTolerance.append(item.isConvergedToAbsoluteTolerance)
                    self.iteration.append(item.iteration)
                    self.residual.append(item.residual)
    
    class MeshingParams:
        '''
        Parameters to be used by the CGAL mesher. Note: they should all be double
        
        Fields:
            xPixelSpacing (default=1): voxel distance in x direction
            yPixelSpacing (default=1): voxel distance in y direction
            SliceThickness (default=1): voxel distance in z direction
                            
            The following parameters are explained in detail in CGAL documentation:
                https://doc.cgal.org/latest/Mesh_3/index.html#Chapter_3D_Mesh_Generation, Section 2.4
            facet_angle (default= 25) 
            facet_size (default= 3)
            facet_distance (default= 2)
            cell_radius_edge (default= 3) 
            general_cell_size (default= 3)
            subdomain (default= np.array([0., 0.])): Specify cell size for each region, in format:
                                                    [region_label1, cell_size1]
                                                    [region_label2, cell_size2]
                                                        ...
                                                    If a region is not specified, value in general_cell_size will be used
            lloyd_smooth (default= True): Switch for Lloyd smoother before local optimization. This can take up to 120s but improves mesh quality
            offset (default= None): offset value to be added to the nodes after meshing. NumPy array of size (3,)
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self, xPixelSpacing=1., yPixelSpacing=1., SliceThickness=1.,
                                 facet_angle = 25., facet_size = 3., facet_distance = 2.,
                                 cell_radius_edge = 3., general_cell_size = 3., subdomain = np.array([0., 0.]),
                                 lloyd_smooth = True, offset = None):
            self.xPixelSpacing = xPixelSpacing
            self.yPixelSpacing = yPixelSpacing
            self.SliceThickness = SliceThickness
            self.facet_angle = facet_angle
            self.facet_size = facet_size
            self.facet_distance = facet_distance
            self.cell_radius_edge = cell_radius_edge
            self.general_cell_size = general_cell_size
            self.subdomain = subdomain
            self.smooth = lloyd_smooth
            self.offset = offset

class io:
    '''
    Dummy class holding some I/O functions
    Dummy class used so the function hierarchy can be compatible with the full version
    '''
    def saveinr(vol, fname, xPixelSpacing=1., yPixelSpacing=1., SliceThickness=1.):
        '''
        Save a volume in the INRIA format. This is for the CGAL mesher.
        Directly translated from the Matlab version
        
        saveinr(vol, fname, xPixelSpacing=1., yPixelSpacing=1., SliceThickness=1.)
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if not '.inr' in fname:
            fname = fname + '.inr'
        if vol.dtype == 'bool' or vol.dtype == 'uint8':
            btype = 'unsigned fixed'
            bitlen = 8
        elif vol.dtype == 'uint16':
            btype = 'unsigned fixed'
            bitlen = 16
        elif vol.dtype == 'float32':
            btype = 'float'
            bitlen = 32
        elif vol.dtype == 'float64':
            btype = 'float'
            bitlen = 64
        else:
            print('volume format not supported')
            return
            
        header = '#INRIMAGE-4#{\nXDIM=%d\nYDIM=%d\nZDIM=%d\nVDIM=1\nTYPE=%s\nPIXSIZE=%d bits\nSCALE=2**0\nCPU=decm\nVX=%f\nVY=%f\nVZ=%f\n#GEOMETRY=CARTESIAN\n' \
                    % (vol.shape[0], vol.shape[1], vol.shape[2], btype, bitlen, xPixelSpacing, yPixelSpacing, SliceThickness)
        for _ in range(256-4-len(header)):
            header += '\n'
        header += '##}\n'
        
        with open(fname, 'wb') as file:
            file.write(header.encode('ascii'))
            if vol.dtype == 'bool':
                vol = vol.astype(np.uint8)
            file.write(vol.tobytes('F'))
        
        return
        
    def readMEDIT(fname):
        '''
        Read a mesh generated by the CGAL mesher, which is saved in MEDIT format
        Directly translated from the Matlab version
        
        elements, nodes, faces, nnpe = readMEDIT(fname)
        
        Outputs:
            elements: list of elements in the mesh
            nodes: node locations of the mesh
            faces: list of faces in the mesh. In case of 2D, it's the same as elements
            nnpe: size of dimension 1 of elements, i.e. 4 for 3D mesh and 3 for 2D mesh
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if not '.mesh' in fname:
            fname = fname + '.mesh'
        file = open(fname)
        all_lines = file.readlines()
        file.close()
        cur = 0
        while(cur < len(all_lines)):
            line = all_lines[cur]
            if 'Vertices' in line:
                break
            else:
                cur += 1
        
        line = all_lines[cur+1]
        nn = np.fromstring(line, dtype=int, sep='\t')[0]
        nodes = []
        for i in range(nn):
            nodes.append(np.fromstring(all_lines[cur+2+i], sep='\t'))
        nodes = np.array(nodes)[:,:-1]
        # Continue reading
        cur += 2+nn
        while(cur < len(all_lines)):
            line = all_lines[cur]
            if 'Triangles' in line:
                tri = True
                break
            elif 'Tetrahedra' in line:
                tet = True
                break
            else:
                cur += 1
        
        if tri:
            line = all_lines[cur+1]
            nt = np.fromstring(line, dtype=int, sep='\t')[0]
            faces = []
            for i in range(nt):
                faces.append(np.fromstring(all_lines[cur+2+i], dtype=int, sep='\t'))
            faces = np.array(faces)
            cur += 2+nt
            while(cur < len(all_lines)):
                line = all_lines[cur]
                if 'Tetrahedra' in line:
                    tet = True
                    break
                else:
                    cur += 1
                    
        # Read the tetrahedrons
        if tet:
            line = all_lines[cur+1]
            ne = np.fromstring(line, dtype=int, sep='\t')[0]
            elements = []
            for i in range(ne):
                elements.append(np.fromstring(all_lines[cur+2+i], dtype=int, sep='\t'))
            elements = np.array(elements)
            nnpe = 4
        else:
            elements = faces
            nnpe = 3
            
        return elements, nodes, faces, nnpe
    
        
class meshing:
    def RunCGALMeshGenerator(mask, opt = utils.MeshingParams()):
        '''
        Generate a tetrahedral mesh from a volume using CGAL mesher, where different regions are labeled used a distinct integer.
        
        ele, nodes = RunCGALMeshGenerator(mask, opt = utils.MeshingParams())
        
        Input:
            mask: (uint8 Numpy array) 3D volumetric data defining the space to mesh. Regions defined by different integers. 0 is background.
            opt (optional): parameters used. See nirfasteruff.utils.MeshingParams for detailed definition and default values
            
        Output:
            ele: element list, one-based
            nodes: node locations of the generated mesh
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if mask.dtype != 'uint8':
            print('Warning: CGAL only supports uint8. I am doing the conversion now, but this can lead to unexpected errors!', flush=1)
            mask = np.uint8(mask)
        
        tmpmeshfn = '._out.mesh'
        tmpinrfn = '._cgal_mesh.inr'

        # Save the tmp INRIA file
        io.saveinr(mask, tmpinrfn)
        
        # call the mesher
        nirfasteruff_cpu.cgal_mesher(tmpinrfn, tmpmeshfn, facet_angle=opt.facet_angle, facet_size=opt.facet_size,
                                     facet_distance=opt.facet_distance, cell_radius_edge_ratio=opt.cell_radius_edge,
                                     general_cell_size=opt.general_cell_size,subdomain=opt.subdomain, smooth=opt.smooth)
        # read result and cleanup
        ele_raw, nodes_raw, _, _ = io.readMEDIT(tmpmeshfn)
        if np.all(opt.offset != None):
            nodes_raw = nodes_raw + opt.offset
        
        ele_tmp = ele_raw[:,:-1]
        nids, ele = np.unique(ele_tmp, return_inverse=1)
        ele += 1 # to one-based
        nodes = nodes_raw[nids-1,:]
        ele = np.c_[ele.reshape((-1,4)), ele_raw[:,-1]]
        if nodes.shape[0] != nodes_raw.shape[0]:
            print(' Removed %d unused nodes from mesh!\n' % (nodes_raw.shape[0]-nodes.shape[0]))
        # remove the tmpfiles
        os.remove(tmpmeshfn)
        os.remove(tmpinrfn)
        return ele, nodes
                

class base:  
    '''
    Dummy class holding the core classes
    Dummy class used so the function hierarchy can be compatible with the full version
    '''    
    class FDdata:
        '''
        Class holding FD/CW data.
        
        Fields:
            phi: fluence from each source. If mesh contains non-tempty field vol, this will be represented on the grid
                last dimension has the size of the number of sources
            complex: Complex amplitude of each channel. Same as amplitude in case of CW data
            link: Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
            amplitude: Absolution amplitude of each channel. I.e. amplitude=abs(complex)
            phase: phase data of each channel. All zero in case of CW data
            vol: Information needed to convert between volumetric and mesh space. Copied from mesh.vol
        
        Methods:
            togrid(mesh): convert data to volumetric space as is defined in mesh.vol. This OVERRIDES the field phi
            isvol(): check if data is in volumetric space
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self):
            self.phi = []
            self.complex = []
            self.link = []
            self.amplitude = []
            self.phase = []
            self.vol = []
            
        def togrid(self, mesh):
            '''
            togrid(mesh): convert data to volumetric space as is defined in mesh.vol. If it is empty, the function does nothing.
            CAUTION: This OVERRIDES the field phi
            
            Input:
                mesh: a nirfasteruff.base.stnd_mesh object. Only the vol field will be used. 
            Output:
                No output
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(mesh.vol.xgrid)>0:
                if len(self.vol.xgrid)>0:
                    print('Warning: data already in volumetric space. Recasted to the new volume.')
                    phi_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]), order='F'))
                    if len(self.vol.zgrid)>0:
                        tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    else:
                        tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                else:
                    if len(mesh.vol.zgrid)>0:
                        tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    else:
                        tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                
                self.phi = tmp
                self.vol = copy.deepcopy(mesh.vol)
            else:
                print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
        
        def isvol(self):
            '''
            Checks if data is in volumetric space.
            Function takes no input and outputs boolean
            '''
            if len(self.vol.xgrid):
                return True
            else:
                return False
    
    class optode:
        '''
        Class for NIRFASTer optodes, which can be either a group of sources or a group of detectors. The field fwhm for sources in the Matlab version has been dropped.
        
        Fields:
            fixed (default=0): whether an optode is fixed. 
                                If not, it will be moved to one scattering length inside the surface (source) or on the surface (detector).
                                The moving is done by calling optode.move_sources, optode.move_detectors, or mesh.touch_optodes
            num: indexing of the optodes, starting from one (1,2,3,...)
            coord: each row is the location of an optode
            int_func: Size (N, dim+2). First column is the index (one-based) of the element each optode is in, the subsequent columns are the barycentric coordinates
        Methods:
            move_sources(mesh): move sources to one scattering length inside of the mesh
            move_detectors(mesh): move detectors to the surface of the mesh
            touch_sources(mesh): make proper adjustments and fill in the missing fields for manually added sources
            touch_detectors(mesh): make proper adjustments and fill in the missing fields for manually added detectors
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self, coord = []):
            self.fixed = 0
            self.num = []
            self.coord = np.ascontiguousarray(coord, dtype=np.float64)
            self.int_func = []
            
        def move_sources(self, mesh):
            '''
            For each source, first move it to the closest point on the surface of the mesh, and then move inside by one scattering length.
            This is function is called when sources are not fixed. Integration functions are also calculated after moving.
            
            Input:
                mesh: a nirfasteruff.base.stnd_mesh object. Can be either 2D or 3D
            Output:
                None
                
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if mesh.type == 'stnd':
                mus_eff = mesh.mus
            else:
                raise TypeError('mesh type: '+mesh.type+' unsupported')
            if len(self.coord)==0:
                print('Warning: no optodes to move')
                return
            if len(mus_eff)==0:
                print('Warning: Source cannot be moved. No optical property found.')
                return
            
            scatt_dist = 1.0 / np.mean(mus_eff)
            mesh_size = np.max(mesh.nodes, axis=0) - np.min(mesh.nodes, axis=0)
            if scatt_dist*10. > np.min(mesh_size):
                print('Warning: Mesh is too small for the scattering coefficient given. Minimal mesh size: ' + str(mesh_size.min()) + 'mm. Scattering distance: '
                      + str(scatt_dist) + 'mm. ' + str(mesh_size.min()/10.) + ' mm will be used for scattering distance. \n You might want to ensure that the scale of your mesh and the scattering coefficient are in mm.')
                scatt_dist = mesh_size.min()/10.
            
            if mesh.elements.shape[1] == 4:
                mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
                # Get all faces that touch the boundary
                faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1,2])], 
                              mesh.elements[np.ix_(mask_touch_surface, [0,1,3])],
                              mesh.elements[np.ix_(mask_touch_surface, [0,2,3])],
                              mesh.elements[np.ix_(mask_touch_surface, [1,2,3])]]
                # sort vertex indices to make them comparable
                faces = np.sort(faces)
                # take unique faces
                faces = np.unique(faces, axis=0)
                #take faces where all three vertices are on the boundary
                faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
            elif mesh.elements.shape[1] == 3:
                mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
                # Get all faces that touch the boundary
                faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1])], 
                              mesh.elements[np.ix_(mask_touch_surface, [0,2])],
                              mesh.elements[np.ix_(mask_touch_surface, [1,2])]]
                # sort vertex indices to make them comparable
                faces = np.sort(faces)
                # take unique faces
                faces = np.unique(faces, axis=0)
                # take edges where both two vertices are on the boundary
                faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
            else:
                raise TypeError('mesh.elements has wrong dimensions')
            
            pos1 = np.zeros(self.coord.shape)
            pos2 = np.zeros(self.coord.shape)
            self.int_func = np.zeros((self.coord.shape[0], mesh.dimension+2))
            
            for i in range(self.coord.shape[0]):
                if mesh.dimension == 2:
                    # find the closest boundary node
                    dist = 1000. * np.ones(mesh.nodes.shape[0])
                    dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                    r0_ind = np.argmin(dist)
                    # find edges including the closest boundary node
                    fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                    # find closest edge
                    dist = np.zeros(fi.shape[0])
                    point = np.zeros((fi.shape[0], 2))
                    for j in range(fi.shape[0]):
                        dist[j], point[j,:] = utils.pointLineDistance(mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], self.coord[i,:2])
                    smallest = np.argmin(dist)
                    
                    # find norm of that edge
                    a = mesh.nodes[fi[smallest, 0], :]
                    b = mesh.nodes[fi[smallest, 1], :]
                    n = np.array([b[1]-a[1], b[0]-a[0]])
                    n = n/np.linalg.norm(n)
                    
                    # move inside by 1 scattering distance
                    pos1[i,:] = point[smallest,:] + n * scatt_dist
                    pos2[i,:] = point[smallest,:] - n * scatt_dist
                elif mesh.dimension == 3:
                    # find the closest boundary node
                    dist = 1000. * np.ones(mesh.nodes.shape[0])
                    dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                    r0_ind = np.argmin(dist)
                    # find edges including the closest boundary node
                    fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                    # find closest edge
                    dist = np.zeros(fi.shape[0])
                    point = np.zeros((fi.shape[0], 3))
                    for j in range(fi.shape[0]):
                        dist[j], point[j,:] = utils.pointTriangleDistance(np.array([mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], mesh.nodes[fi[j,2],:]]), self.coord[i,:])
                    smallest = np.argmin(dist)
                    
                    # find norm of that edge
                    a = mesh.nodes[fi[smallest, 0], :]
                    b = mesh.nodes[fi[smallest, 1], :]
                    c = mesh.nodes[fi[smallest, 2], :]
                    n = np.cross(b-a, c-a)
                    n = n/np.linalg.norm(n)
                    
                    # move inside by 1 scattering distance
                    pos1[i,:] = point[smallest,:] + n * scatt_dist
                    pos2[i,:] = point[smallest,:] - n * scatt_dist
                else:
                    raise TypeError('mesh.dimension should be 2 or 3')
            
            ind, int_func = utils.pointLocation(mesh, pos1)
            in_ind = ind>-1
            self.coord[in_ind,:] = pos1[in_ind,:]
            self.int_func[in_ind,:] = np.c_[ind[in_ind]+1, int_func[in_ind,:]]  # to one-based
            if np.all(in_ind):
                return
            else:
                nan_ind = ~in_ind
                ind2, int_func2 = utils.pointLocation(mesh, pos2[nan_ind,:])
                self.coord[nan_ind,:] = pos2[nan_ind, :]
                self.int_func[nan_ind,:] = np.c_[ind2+1, int_func2] # to one-based
            
            if np.any(ind2==-1):
                print('Warning: Source(s) could not be moved. The mesh structure may be poor.')
            
        
        def move_detectors(self, mesh):
            '''
            For each detector, move it to the closest point on the surface of the mesh.
            This is function is called when detectors are not fixed. Integration functions are NOT calculated after moving.
            
            Input:
                mesh: a nirfasteruff.base.stnd_mesh object. Can be either 2D or 3D
            Output:
                None
                
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(self.coord)==0:
                print('Warning: no optodes to move')
                return
            
            if mesh.elements.shape[1] == 4:
                mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
                # Get all faces that touch the boundary
                faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1,2])], 
                              mesh.elements[np.ix_(mask_touch_surface, [0,1,3])],
                              mesh.elements[np.ix_(mask_touch_surface, [0,2,3])],
                              mesh.elements[np.ix_(mask_touch_surface, [1,2,3])]]
                # sort vertex indices to make them comparable
                faces = np.sort(faces)
                # take unique faces
                faces = np.unique(faces, axis=0)
                #take faces where all three vertices are on the boundary
                faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
            elif mesh.elements.shape[1] == 3:
                mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
                # Get all faces that touch the boundary
                faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1])], 
                              mesh.elements[np.ix_(mask_touch_surface, [0,2])],
                              mesh.elements[np.ix_(mask_touch_surface, [1,2])]]
                # sort vertex indices to make them comparable
                faces = np.sort(faces)
                # take unique faces
                faces = np.unique(faces, axis=0)
                # take edges where both two vertices are on the boundary
                faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
            else:
                raise TypeError('mesh.elements has wrong dimensions')
            
            for i in range(self.coord.shape[0]):
                if mesh.dimension == 2:
                    # find the closest boundary node
                    dist = 1000. * np.ones(mesh.nodes.shape[0])
                    dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                    r0_ind = np.argmin(dist)
                    # find edges including the closest boundary node
                    fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                    # find closest edge
                    dist = np.zeros(fi.shape[0])
                    point = np.zeros((fi.shape[0], 2))
                    for j in range(fi.shape[0]):
                        dist[j], point[j,:] = utils.pointLineDistance(mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], self.coord[i,:2])
                    smallest = np.argmin(dist)
                    # move detector to the closest point on that edge
                    self.coord[i,:] = point[smallest,:]
                elif mesh.dimension == 3:
                    # find the closest boundary node
                    dist = 1000. * np.ones(mesh.nodes.shape[0])
                    dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                    r0_ind = np.argmin(dist)
                    # find edges including the closest boundary node
                    fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                    # find closest edge
                    dist = np.zeros(fi.shape[0])
                    point = np.zeros((fi.shape[0], 3))
                    for j in range(fi.shape[0]):
                        dist[j], point[j,:] = utils.pointTriangleDistance(np.array([mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], mesh.nodes[fi[j,2],:]]), self.coord[i,:])
                    smallest = np.argmin(dist)
                    # move detector to the closest point on that edge
                    self.coord[i,:] = point[smallest,:]
                else:
                    raise TypeError('mesh.dimension should be 2 or 3')
            
        
        def touch_sources(self, mesh):
            '''
            Recalculate/fill in all other fields based on 'fixed' and 'coord'. This is useful when a set of sources are manually added and only the locations are specified.
            For non-fixed sources, function 'move_sources' is called, otherwise recalculates integration functions directly
            If no source locations are specified, the function does nothing
            
            Input:
                mesh: a nirfasteruff.base.stnd_mesh object. Can be either 2D or 3D
            Output: None
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(self.coord)==0:
                return
            n_src = self.coord.shape[0]
            self.num = np.arange(1, n_src+1, dtype=np.float64)
            if not self.fixed:
                self.move_sources(mesh)
            else:
                ind, int_func = utils.pointLocation(mesh, self.coord)
                self.int_func = np.c_[ind+1, int_func]
        
        def touch_detectors(self, mesh):
            '''
            Recalculate/fill in all other fields based on 'fixed' and 'coord'. This is useful when a set of detectors are manually added and only the locations are specified.
            For non-fixed detectors, function 'move_detectors' is first called, and integration functions are calculated subsequentely.
            For fixed detectors, recalculates integration functions directly.
            If no source locations are specified, the function does nothing
            
            Input:
                mesh: a nirfasteruff.base.stnd_mesh object. Can be either 2D or 3D
            Output: None
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(self.coord)==0:
                return
            n_det = self.coord.shape[0]
            self.num = np.arange(1, n_det+1, dtype=np.float64)
            if not self.fixed:
                self.move_detectors(mesh)
            ind, int_func = utils.pointLocation(mesh, self.coord)
            self.int_func = np.c_[ind+1, int_func]
    
    class meshvol:
        '''
        Small class holding the information needed for converting between mesh and volumetric space. Values calculated by nirfasteruff.base.stnd_mesh.gen_intmat
        Note that the volumetric space, defined by xgrid, ygrid, and zgrid (empty for 2D mesh), must be uniform
        All fields should be directly compatible with the Matlab version
        
        Fields:
            xgrid: x grid of the volumetric space
            ygrid: y grid of the volumetric space
            zgrid: z grid of the volumetric space. Empty for 2D meshes
            mesh2grid: sparse matrix of size (len(xgrid)*len(ygrid)*len(ygrid), Nnodes). 
                        For mesh-space data with size (Nnodes,), convertion to volumetric space is done by mesh2grid.dot(data)
                        The result is vectorized in 'F' (Matlab) order
            gridinmesh: indices (one-based) of data points in the volumetric space that are within the mesh space, vectorized in 'F' order.
            res: resolution in x, y, z (if 3D) direction
            grid2mesh: sparse matrix of size (Nnodes, len(xgrid)*len(ygrid)*len(ygrid)). 
                        For volumetric data vectorized in 'F' order, convertion to mesh space is done by grid2mesh.dot(data)
            meshingrid: indices (one-based) of data points in the mesh space that are within the volumetric space
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self):
            self.xgrid = []
            self.ygrid = []
            self.zgrid = []
            self.mesh2grid = sparse.csc_matrix([])
            self.gridinmesh = []
            self.res = []
            self.grid2mesh = sparse.csc_matrix([])
            self.meshingrid = []
    
    class stndmesh:
        '''
        Main class for standard mesh. The methods should cover most of the commonly-used functions
        
        Fields:
            name (default='EmptyMesh'): string of mesh name
            nodes: (double NumPy array) locations of nodes in the mesh. Size (NNodes, dim)
            bndvtx: (double NumPy array) indicator of whether a node is at boundary (1) or internal (0). Size (NNodes,)
            type (always='stnd'): type of the mesh
            mua: (double NumPy array) absorption coefficient (mm^-1) at each node. Size (NNodes,)
            kappa: (double NumPy array) diffusion coefficient (mm) at each node. Size (NNodes,). Defined as 1/(3*(mua + mus))
            ri: (double NumPy array) refractive index at each node. Size (NNodes,)
            mus: (double NumPy array) reduced scattering coefficient (mm^-1) at each node. Size (NNodes,)
            elements: (double NumPy array) triangulation (tetrahedrons or triangles) of the mesh, Size (NElements, dim+1)
                        Row i contains the indices (one-based) of the nodes that form tetrahedron/triangle i
            region: (double NumPy array) region labeling of each node. Starting from 1. Size (NNodes,)
            source: a nirfasteruff.base.optode object containing the sources
            meas: a nirfasteruff.base.optode object containing the detectors
            link: (int32 NumPy array) list of source-detector pairs, i.e. channels. Size (NChannels,3)
                    First column: source; Second column: detector; Third column: active (1) or not (0)
            c: (double NumPy array) light speed (mm/sec) at each node.  Size (NNodes,). Defined as 1/ri
            ksi: (double NumPy array) photon fluence rate scale factor on the mesh-outside_mesh boundary as derived from Fresenel's law. Size (NNodes,)
            element_area: (double NumPy array) volume/area (mm^3 or mm^2) of each element. Size (NElements,)
            support: (double NumPy array) total volume/area of all the elements each node belongs to. Size (NNodes,)
            vol: a nirfasteruff.base.meshvol object holding information for converting between mesh and volumetric space. Empty if not available
        Methods:
            from_copy(mesh): deep copy all fields from another mesh.
            from_file(file): construct mesh by reading the classic NIRFASTer ASCII files
            from_mat(matfile): construct mesh from a Matlab .mat file containing a NIRFASTer mesh struct
            from_solid(ele, nodes, prop = None, src = None, det = None, link = None): construct mesh from a triangularization generated by an external mesher
            from_volume(vol, param = utils.MeshingParams(), prop = None, src = None, det = None, link = None): construct mesh from a segmented 3D volume
            from_cedalion(*PLACEHOLDER*): construct mesh from cedalion format
            save_nirfast(filename): save mesh to the classic NIRFASTer ASCII files
            set_prop(prop): set optical properties of the mesh using information in 'prop'
            touch_optodes(): move (if not fixed) the optodes and (re)calculate the barycentric coordinates (i.e. integration functions)
            isvol(): check if field 'vol' is already calculated
            gen_intmat(xgrid,ygrid,zgrid=None): calculate the information needed for converting between mesh and volumetric space, defined by x,y,z grids
            femdata(freq, solver=utils.get_solver(), opt=utils.SolverOptions()): calculate the fluence for each source
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        def __init__(self):
            self.name = 'EmptyMesh'
            self.nodes = []
            self.bndvtx = []
            self.type = 'stnd'
            self.mua = []
            self.kappa = []
            self.ri = []
            self.mus = []
            self.elements = []
            self.dimension = []
            self.region = []
            self.source = base.optode()
            self.meas = base.optode()
            self.link = []
            self.c = []
            self.ksi = []
            self.element_area = []
            self.support = []
            self.vol = base.meshvol()
        
        def from_copy(self, mesh):
            '''
            Deep copy all fields from another mesh.
            
            Input:
                mesh: the mesh to copy from
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            self.name = copy.deepcopy(mesh.name)
            self.nodes = copy.deepcopy(mesh.nodes)
            self.bndvtx = copy.deepcopy(mesh.bndvtx)
            self.type = 'stnd'
            self.mua = copy.deepcopy(mesh.mua)
            self.kappa = copy.deepcopy(mesh.kappa)
            self.ri = copy.deepcopy(mesh.ri)
            self.mus = copy.deepcopy(mesh.mus)
            self.elements = copy.deepcopy(mesh.elements)
            self.dimension = copy.deepcopy(mesh.dimension)
            self.region = copy.deepcopy(mesh.region)
            self.source = copy.deepcopy(mesh.source)
            self.meas = copy.deepcopy(mesh.meas)
            self.link = copy.deepcopy(mesh.link)
            self.c = copy.deepcopy(mesh.c)
            self.ksi = copy.deepcopy(mesh.ksi)
            self.element_area = copy.deepcopy(mesh.element_area)
            self.support = copy.deepcopy(mesh.support)
            self.vol = copy.deepcopy(mesh.vol)
        
        def from_file(self, file):
            '''
            Read from classic NIRFAST mesh format, not checking the correctness of the loaded integration functions.
            All fields after loading should be directly compatible with Matlab version.
            
            mesh = nirfasteruff.base.stdn_mesh()
            mesh.from_file('meshname')
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if '.mat' in file:
                print('Error: seemingly you are trying to load a mat file. Please use .from_mat method instead')
                return
            # clear data
            self.__init__()
            file = os.path.splitext(file)[0] # in case an extension is accidentally included
            self.name = os.path.split(file)[1]
            # Read the nodes
            if os.path.isfile(file + '.node'):
                fullname = file + '.node'
                tmp = np.genfromtxt(fullname, delimiter='\t', dtype=np.float64)
                self.bndvtx = np.ascontiguousarray(tmp[:,0])
                self.nodes = np.ascontiguousarray(tmp[:,1:])
                self.dimension = tmp.shape[1]-1
            else:
                print('Error: ' + file + '.node file is not present')
                self.__init__()
                return
            # Read the parameters
            if os.path.isfile(file + '.param'):
                fullname = file + '.param'
                with open(fullname, 'r') as paramfile:
                    header = paramfile.readline()
                if ord(header[0])>=48 and ord(header[0])<=57:
                    print('Error: header missing in .param file. You are probably using the old format, which is no longer supported')
                    self.__init__()
                    return
                elif 'stnd' not in header:
                    print('Error: only stnd mesh is supported in this version')
                    self.__init__()
                    return
                else:
                    tmp = np.genfromtxt(fullname, skip_header=1, dtype=np.float64)
                    self.mua = np.ascontiguousarray(tmp[:,0])
                    self.kappa = np.ascontiguousarray(tmp[:,1])
                    self.ri = np.ascontiguousarray(tmp[:,2])
                    self.mus = (1./self.kappa)/3. - self.mua
            else:
                print('Error: ' + file + '.param file is not present')
                self.__init__()
                return
            # Read the elements
            if os.path.isfile(file + '.elem'):
                fullname = file + '.elem'
                ele = np.genfromtxt(fullname, delimiter='\t', dtype=np.float64)
                ele = np.sort(ele)
                if self.dimension==2:
                    self.elements = np.ascontiguousarray(utils.check_element_orientation_2d(ele, self.nodes))
                else:
                    self.elements = np.ascontiguousarray(ele)
                if ele.shape[1]-1 != self.dimension:
                    print('Warning: nodes and elements seem to have incompatable dimentions. Are you using an old 2D mesh?')
            else:
                print('Error: ' + file + '.elem file is not present')
                self.__init__()
                return
            # Read the region information
            if os.path.isfile(file + '.region'):
                fullname = file + '.region'
                self.region = np.ascontiguousarray(np.genfromtxt(fullname, dtype=np.float64))                    
            else:
                print('Error: ' + file + '.region file is not present')
                self.__init__()
                return
            # Read the source file
            if not os.path.isfile(file + '.source'):
                print('Warning: source file is not present')
            else:
                fullname = file + '.source'
                errorflag = False
                with open(fullname, 'r') as srcfile:
                    hdr1 = srcfile.readline()
                    hdr2 = srcfile.readline()
                if ord(hdr1[0])>=48 and ord(hdr1[0])<=57:
                    print('WARNING: header missing in .source file. You are probably using the old format, which is no longer supported.\nSource not loaded')
                    errorflag = True
                elif 'num' not in hdr1 and 'num' not in hdr2:
                    print('WARNING: Incorrect or old header format.\nSource not loaded')
                    errorflag = True
                elif 'fixed' in hdr1:
                    fixed = 1
                    N_hdr = 2
                    hdr = hdr2.split()
                else:
                    fixed = 0
                    N_hdr = 1
                    hdr = hdr1.split()
                if not errorflag:
                    tmp = np.genfromtxt(fullname, skip_header=N_hdr, dtype=np.float64)
                    src = base.optode()
                    src.fixed = fixed
                    src.num = tmp[:, hdr.index('num')]
                    if 'z' not in hdr:
                        src.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')]]
                    else:
                        src.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')], tmp[:, hdr.index('z')]]
                        if self.dimension==2:
                            print('Warning: Sources are 3D, mesh is 2D.')
                    if 'fwhm' in hdr:
                        fwhm = tmp[:, hdr.index('fwhm')]
                        if np.any(fwhm):
                            print('Warning: Only point sources supported. Ignoring field fwhm')
                    if 'ele' in hdr:
                        if 'ip1' in hdr and 'ip2' in hdr and 'ip3' in hdr:
                            src.int_func = np.c_[tmp[:, hdr.index('ele')], tmp[:, hdr.index('ip1')], tmp[:, hdr.index('ip2')], tmp[:, hdr.index('ip3')]]
                            if 'ip4' in hdr:
                                src.int_func = np.c_[src.int_func, tmp[:, hdr.index('ip4')]]
                                if self.dimension==2:
                                    print('Warning: Sources ''int_func'' are 3D, mesh is 2D.')
                            else:
                                if self.dimension==3:
                                    print('Warning: Sources ''int_func'' are 2D, mesh is 3D. Will recalculate')
                                    src.int_func = []
                        else:
                            print('Warning: source int_func stored in wrong format. Will recalculate')
                    
                    if src.fixed==1  or len(src.int_func)>0:
                        if src.fixed==1:
                            print('Fixed sources')
                        if len(src.int_func)>0:
                            print('Sources integration functions loaded')
                    else:
                        # non-fixed sources and no int_func loaded. Let's move the sources by one scattering length now
                        print('Moving Sources', flush = 1)
                        src.move_sources(self)
                    if len(src.int_func)==0:
                        print('Calculating sources integration functions', flush = 1)
                        ind, int_func = utils.pointLocation(self, src.coord)
                        src.int_func = np.c_[ind+1, int_func]
                    src.int_func[src.int_func[:,0]==0, 0] = np.nan
                    self.source = src
                    if not np.all(np.isfinite(src.int_func[:,0])):
                        print('Warning: some sources might be outside the mesh')
            # Read the detector file
            if not os.path.isfile(file + '.meas'):
                print('Warning: detector file is not present')
            else:
                fullname = file + '.meas'
                errorflag = False
                with open(fullname, 'r') as srcfile:
                    hdr1 = srcfile.readline()
                    hdr2 = srcfile.readline()
                if ord(hdr1[0])>=48 and ord(hdr1[0])<=57:
                    print('WARNING: header missing in .meas file. You are probably using the old format, which is no longer supported.\nDetector not loaded')
                    errorflag = True
                elif 'num' not in hdr1 and 'num' not in hdr2:
                    print('WARNING: Incorrect or old header format.\nDetector not loaded')
                    errorflag = True
                elif 'fixed' in hdr1:
                    fixed = 1
                    N_hdr = 2
                    hdr = hdr2.split()
                else:
                    fixed = 0
                    N_hdr = 1
                    hdr = hdr1.split()
                if not errorflag:
                    tmp = np.genfromtxt(fullname, skip_header=N_hdr, dtype=np.float64)
                    det = base.optode()
                    det.fixed = fixed
                    det.num = tmp[:, hdr.index('num')]
                    if 'z' not in hdr:
                        det.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')]]
                    else:
                        det.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')], tmp[:, hdr.index('z')]]
                        if self.dimension==2:
                            print('Warning: Detectors are 3D, mesh is 2D.')
                    if 'ele' in hdr:
                        if 'ip1' in hdr and 'ip2' in hdr and 'ip3' in hdr:
                            det.int_func = np.c_[tmp[:, hdr.index('ele')], tmp[:, hdr.index('ip1')], tmp[:, hdr.index('ip2')], tmp[:, hdr.index('ip3')]]
                            if 'ip4' in hdr:
                                det.int_func = np.c_[det.int_func, tmp[:, hdr.index('ip4')]]
                                if self.dimension==2:
                                    print('Warning: Detectors ''int_func'' are 3D, mesh is 2D.')
                            else:
                                if self.dimension==3:
                                    print('Warning: Detectors ''int_func'' are 2D, mesh is 3D. Will recalculate')
                                    det.int_func = []
                        else:
                            print('Warning: detector int_func stored in wrong format. Will recalculate')
                    
                    if det.fixed==1  or len(det.int_func)>0:
                        if det.fixed==1:
                            print('Fixed detectors')
                        if len(det.int_func)>0:
                            print('Detectors integration functions loaded')
                    else:
                        # non-fixed sources and no int_func loaded. Let's move the sources by one scattering length now
                        print('Moving Detectors', flush = 1)
                        det.move_detectors(self)
                    if len(det.int_func)==0:
                        print('Calculating detectors integration functions', flush = 1)
                        ind, int_func = utils.pointLocation(self, det.coord)
                        det.int_func = np.c_[ind+1, int_func]
                    det.int_func[det.int_func[:,0]==0, 0] = np.nan
                    self.meas = det
                    if not np.all(np.isfinite(det.int_func[:,0])):
                        print('Warning: some detectors might be outside the mesh')
            # load link list
            if os.path.isfile(file + '.link'):
                fullname = file + '.link'
                with open(fullname, 'r') as linkfile:
                    header = linkfile.readline()
                if ord(header[0])>=48 and ord(header[0])<=57:
                    print('Warning: header missing in .link file. You are probably using the old format, which is no longer supported')
                else:
                    self.link = np.genfromtxt(fullname, skip_header=1, dtype=np.int32)
            else:
                print('Warning: link file is not present')
            # Speed of light in medium
            c0 = 299792458000.0 # mm/s
            self.c = c0/self.ri
            # Set boundary coefficient using definition of baundary attenuation A using the Fresenel's law; Robin type
            n_air = 1.
            n = self.ri/n_air
            R0 = ((n-1.)**2)/((n+1.)**2)
            theta = np.arcsin(1.0/n)
            A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
            self.ksi = 1.0 / (2*A)
            # area and support for each element
            self.element_area = nirfasteruff_cpu.ele_area(self.nodes, self.elements)
            self.support = nirfasteruff_cpu.mesh_support(self.nodes, self.elements, self.element_area)
        
        def from_mat(self, matfile, varname = None):
            '''
            Read from Matlab .mat file that contains a NIRFASTer mesh struct. All fields copied as is without error checking.
            
            Input:
                matfile: (string) name of the .mat file to load. Use of extension is optional
                varname (optional): (string) if your .mat file contains multiple variables, use this argument to specify which one to load
            
            mesh = nirfasteruff.base.stdn_mesh()
            mesh.from_mat('meshname.mat', 'mymesh')
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if type(matfile) != str:
                print('argument 1 must be a string!')
                return
            if varname != None and type(varname) != str:
                print('argument 2 must be a string!')
                return
            
            try:
                tmp = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
            except:
                print('Failed to load Matlab file ' + matfile + '!')
                return
            
            if varname != None:
                try:
                    mesh = tmp[varname]
                except:
                    print('Cannot load mesh ' + varname + ' from mat file ' + matfile)
            else:
                allkeys = list(tmp.keys())
                is_struct = [type(tmp[key])==sio.matlab._mio5_params.mat_struct for key in allkeys]
                if sum(is_struct) != 1:
                    print('There must be precisely one struct in the mat file, if "varname" is not provided')
                    return
                else:
                    varname = allkeys[is_struct.index(True)]
                    mesh = tmp[varname]
                    
            if mesh.type != 'stnd':
                print('mesh type must be standard')
                return
            # Now let's copy the data
            self.__init__()
            self.name = mesh.name
            self.nodes = np.atleast_2d(np.ascontiguousarray(mesh.nodes, dtype=np.float64))
            self.bndvtx = np.ascontiguousarray(mesh.bndvtx, dtype=np.float64)
            self.mua = np.ascontiguousarray(mesh.mua, dtype=np.float64)
            self.kappa = np.ascontiguousarray(mesh.kappa, dtype=np.float64)
            self.ri = np.ascontiguousarray(mesh.ri, dtype=np.float64)
            self.mus = np.ascontiguousarray(mesh.mus, dtype=np.float64)
            self.elements = np.atleast_2d(np.ascontiguousarray(mesh.elements, dtype=np.float64))
            self.dimension = mesh.dimension
            self.region = np.ascontiguousarray(mesh.region, dtype=np.float64)
            self.ksi = np.ascontiguousarray(mesh.ksi, dtype=np.float64)
            self.c = np.ascontiguousarray(mesh.c, dtype=np.float64)
            self.element_area = np.ascontiguousarray(mesh.element_area, dtype=np.float64)
            self.support = np.ascontiguousarray(mesh.support, dtype=np.float64)
            allfields = mesh._fieldnames
            if 'source' in allfields:
                self.source.fixed = mesh.source.fixed
                self.source.num = np.ascontiguousarray(mesh.source.num, dtype=np.int32)
                self.source.coord = np.ascontiguousarray(mesh.source.coord, dtype=np.float64)
                self.source.int_func = np.ascontiguousarray(mesh.source.int_func, dtype=np.float64)
            else:
                print('Warning: sources are not present in mesh')
            if 'meas' in allfields:
                self.meas.fixed = mesh.meas.fixed
                self.meas.num = np.ascontiguousarray(mesh.meas.num, dtype=np.int32)
                self.meas.coord = np.ascontiguousarray(mesh.meas.coord, dtype=np.float64)
                self.meas.int_func = np.ascontiguousarray(mesh.meas.int_func, dtype=np.float64)
            else:
                print('Warning: detectors are not present in mesh')
            if 'link' in allfields:
                self.link = np.ascontiguousarray(mesh.link, dtype=np.int32)
            else:
                print('Warning: link is not present in mesh')
            if 'vol' in allfields:
                self.vol.xgrid = mesh.vol.xgrid
                self.vol.ygrid = mesh.vol.ygrid
                self.vol.zgrid = mesh.vol.zgrid
                self.vol.mesh2grid = mesh.vol.mesh2grid
                self.vol.gridinmesh = mesh.vol.gridinmesh
                self.vol.res = mesh.vol.res
                self.vol.grid2mesh = mesh.vol.grid2mesh
                self.vol.meshingrid = mesh.vol.meshingrid
                
        def from_solid(self, ele, nodes, prop = None, src = None, det = None, link = None):
            '''
            Construct mesh from a solid mesh generated by a mesher. Similar to the solidmesh2nirfast function in Matlab version.
            Can also set the optical properties and optodes if supplied
            
            Input:
                ele: (int/double NumPy array) element list in one-based indexing. If four columns, all nodes will be labeled as region 1
                    If five columns, the last column will be used for region labeling
                nodes: (double NumPy array) node locations in the mesh. Size (NNodes,3)
                prop (optional): (double NumPy array) optical property information. See stnd_mesh.set_prop() for details
                src (optional): a nirfasteruff.base.optode object of all sources. Only fields src.fixed and src.coord are mandatory.
                det (optional): a nirfasteruff.base.optode object of all detectors. Only fields det.fixed and det.coord are mandatory.
                link (optional): (NumPy array, any dtype convertible to int32) channel information. See class definition for details.
            
            mesh = nirfasteruff.base.stdn_mesh()
            mesh.from_solid(ele, nodes, prop = None, src = None, det = None, link = None)
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            self.__init__()
            num_nodes = nodes.shape[0]
            self.nodes = np.ascontiguousarray(nodes, dtype=np.float64)
            if ele.shape[1] == 4:
                # no region label
                self.elements = np.ascontiguousarray(np.sort(ele,axis=1), dtype=np.float64)
                self.region = np.ones(num_nodes)
            elif ele.shape[1] == 5:
                self.region = np.zeros(num_nodes)
                # convert element label to node label
                labels = np.unique(ele[:,-1])
                for i in range(len(labels)):
                    tmp = ele[ele[:,-1]==labels[i], :-1]
                    idx = np.int32(np.unique(tmp) - 1)
                    self.region[idx] = labels[i]
                self.elements = np.ascontiguousarray(np.sort(ele[:,:-1],axis=1), dtype=np.float64)
            else:
                print('Error: elements in wrong format')
                self.__init__()
                return
            # find the boundary nodes: find faces that are referred to only once
            faces = np.r_[ele[:, [0,1,2]], 
                          ele[:, [0,1,3]],
                          ele[:, [0,2,3]],
                          ele[:, [1,2,3]]]
            
            faces = np.sort(faces)
            unique_faces, cnt = np.unique(faces, axis=0, return_counts=1)
            bnd_faces = unique_faces[cnt==1, :]
            bndvtx = np.unique(bnd_faces)
            self.bndvtx = np.zeros(nodes.shape[0])
            self.bndvtx[np.int32(bndvtx-1)] = 1
            # area and support for each element
            self.element_area = nirfasteruff_cpu.ele_area(self.nodes, self.elements)
            self.support = nirfasteruff_cpu.mesh_support(self.nodes, self.elements, self.element_area)
            self.dimension = 3
            if np.any(prop != None):
                self.set_prop(prop)
            else:
                print('Warning: optical properties not specified')
            if src != None:
                self.source = copy.deepcopy(src)
                self.source.touch_sources(self)
            else:
                print('Warning: no sources specified')
            if det != None:
                self.meas = copy.deepcopy(det)
                self.meas.touch_detectors(self)
            else:
                print('Warning: no detectors specified')
            if np.all(link != None):
                if link.shape[1]==3:
                    self.link = copy.deepcopy(np.ascontiguousarray(link, dtype=np.int32))
                elif link.shape[1]==2:
                    self.link = copy.deepcopy(np.ascontiguousarray(np.c_[link, np.ones(link.shape[0])], dtype=np.int32))
                else:
                    print('Warning: link in wrong format. Ignored.')
            else:
                print('Warning: no link specified')
        
        def from_volume(self, vol, param = utils.MeshingParams(), prop = None, src = None, det = None, link = None):
            '''
            Construct mesh from a segmented 3D volume using the built-in CGAL mesher. Calls stnd_mesh.from_solid after meshing step.
            
            Input:
                vol: (uint8 NumPy array) 3D segmented volume to be meshed. 0 is considered as outside. Regions labeled using unique integers
                param (optional): parameters used for the mesher. See nirfasteruff.utils.MeshingParams for details. Default will be used if not specified.
                                    Please modify fields xPixelSpacing, yPixelSpacing, and SliceThickness if your volume doesn't have [1,1,1] resolution
                prop (optional): see stnd_mesh.from_solid
                src (optional): see stnd_mesh.from_solid
                det (optional): see stnd_mesh.from_solid
                link (optional): see stnd_mesh.from_solid
                
            mesh = nirfasteruff.base.stnd_mesh()
            mesh.from_volume(vol, param = utils.MeshingParams(), prop = None, src = None, det = None, link = None)
                
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(vol.shape) != 3:
                print('Error: vol should be a 3D matrix in unit8')
                return
            print('Running CGAL mesher', flush=1)
            ele, nodes = meshing.RunCGALMeshGenerator(vol, param)
            print('Converting to NIRFAST format', flush=1)
            self.from_solid(ele, nodes, prop, src, det, link)
        
        def from_cedalion(self, something):
            # TO DO: create a mesh from cedalion model
            pass
        
        def set_prop(self, prop):
            '''
            Set optical properties of the whole mesh, using information provided in prop.
            
            Input:
                prop: (double NumPy array) opttical property info, similar to the MCX format:
                                            [region mua(mm-1) musp(mm-1) ri]
                                            [region mua(mm-1) musp(mm-1) ri]
                                            [...]
                        where 'region' is the region label, and they should match exactly with unique(mesh.region). The order doesn't matter.
            
            mesh.set_prop(prop)
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            num_nodes = self.nodes.shape[0]
            self.mua = np.zeros(num_nodes)
            self.mus = np.zeros(num_nodes)
            self.kappa = np.zeros(num_nodes)
            self.c = np.zeros(num_nodes)
            self.ksi = np.zeros(num_nodes)
            self.ri = np.zeros(num_nodes)
            
            if prop.shape[0]!=len(np.unique(self.region)) or (prop.shape[0]==len(np.unique(self.region)) and np.any(np.sort(prop[:,0])-np.unique(self.region))):
                print('Warning: regions in mesh and regions in prop matrix mismatch. Ignored.')
            elif prop.shape[1]!=4:
                print('Warning: prop matrix has wrong number of columns. Should be: region mua(mm-1) musp(mm-1) ri. Ignored')
            else:
                labels = prop[:,0]
                for i in range(len(labels)):
                    self.mua[self.region==labels[i]] = prop[i,1]
                    self.mus[self.region==labels[i]] = prop[i,2]
                    self.ri[self.region==labels[i]] = prop[i,3]
                self.kappa = 1.0/(3.0*(self.mua + self.mus))
                c0 = 299792458000.0 # mm/s
                self.c = c0/self.ri
                n_air = 1.
                n = self.ri/n_air
                R0 = ((n-1.)**2)/((n+1.)**2)
                theta = np.arcsin(1.0/n)
                A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
                self.ksi = 1.0 / (2*A)
                
        def change_prop(self, idx, prop):
            '''
            Change optical properties (mua, musp, and ri) at nodes specified in idx, and automatically change fields kappa, c, and ksi as well
            
            mesh.change_prop(idx, prop)
            
            Input:
                idx: (list or NumPy array) zero-based indices of nodes to change
                prop: (list or NumPy array of length 3) new optical properties to be assigned to the specified nodes. [mua(mm-1) musp(mm-1) ri]
                
            Output:
                None
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            idx = np.array(idx, dtype = np.int32)
            self.mua[idx] = prop[0]
            self.mus[idx] = prop[1]
            self.ri[idx] = prop[2]
            
            self.kappa = 1.0 / (3.0*(self.mua + self.mus))
            c0 = 299792458000.0 # mm/s
            self.c = c0/self.ri
            n_air = 1.
            n = self.ri/n_air
            R0 = ((n-1.)**2)/((n+1.)**2)
            theta = np.arcsin(1.0/n)
            A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
            self.ksi = 1.0 / (2*A)
        
        def touch_optodes(self):
            '''
            Moves all optodes (if non fixed) and recalculate the integration functions (i.e. barycentric coordinates). 
            See optode.touch_sources and optode.touch_detectors for details
            
            Function has no input or output.
            
            mesh.touch_optodes()
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            # make sure the optodes sit correctly: moved if needed, calculate the int func
            print('touching sources', flush=1)
            self.source.touch_sources(self)
            print('touching detectors', flush=1)
            self.meas.touch_detectors(self)
        
        def save_nirfast(self, filename):
            '''
            Save mesh in the classic NIRFASTer ASCII format.
            
            Input:
                filename: name of the file to be saved as
                
            mesh.save_nirfast(filename)
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            # save nodes
            np.savetxt(filename+'.node', np.c_[self.bndvtx, self.nodes], fmt='%.16g', delimiter='\t')
            # save elements
            np.savetxt(filename+'.elem', self.elements, fmt='%g', delimiter='\t')
            # save params
            kappa = 1.0/(3.0*(self.mua + self.mus))
            np.savetxt(filename+'.param', np.c_[self.mua, kappa, self.ri], fmt='%g',
                       delimiter=' ', header='stnd', comments='')
            # save regions
            np.savetxt(filename+'.region', self.region, fmt='%g', delimiter='\t')
            # save sources, if exist
            if len(self.source.coord)==0:
                if os.path.isfile(filename+'.source'):
                    os.remove(filename+'.source')
            else:
                if len(self.source.int_func)>0 and self.source.int_func.shape[0]==self.source.coord.shape[0]:
                    if self.dimension==2:
                        hdr = 'num x y ele ip1 ip2 ip3'
                    elif self.dimension==3:
                        hdr = 'num x y z ele ip1 ip2 ip3 ip4'
                    data = np.c_[self.source.num, self.source.coord, self.source.int_func]
                else:
                    if self.dimension==2:
                        hdr = 'num x y'
                    elif self.dimension==3:
                        hdr = 'num x y z'
                    data = np.c_[self.source.num, self.source.coord]
                if self.source.fixed == 1:
                    hdr = 'fixed\n' + hdr
                np.savetxt(filename+'.source', data, fmt='%.16g', delimiter=' ', header=hdr, comments='')
            # save detectors, if exist
            if len(self.meas.coord)==0:
                if os.path.isfile(filename+'.meas'):
                    os.remove(filename+'.meas')
            else:
                if len(self.meas.int_func)>0 and self.meas.int_func.shape[0]==self.meas.coord.shape[0]:
                    if self.dimension==2:
                        hdr = 'num x y ele ip1 ip2 ip3'
                    elif self.dimension==3:
                        hdr = 'num x y z ele ip1 ip2 ip3 ip4'
                    data = np.c_[self.meas.num, self.meas.coord, self.meas.int_func]
                else:
                    if self.dimension==2:
                        hdr = 'num x y'
                    elif self.dimension==3:
                        hdr = 'num x y z'
                    data = np.c_[self.meas.num, self.meas.coord]
                if self.meas.fixed == 1:
                    hdr = 'fixed\n' + hdr
                np.savetxt(filename+'.meas', data, fmt='%.16g', delimiter=' ', header=hdr, comments='')
            # save link, if exist
            if len(self.link)==0:
                if os.path.isfile(filename+'.link'):
                    os.remove(filename+'.link')
            else:
                hdr = 'source detector active'
                np.savetxt(filename+'.link', self.link, fmt='%g', delimiter=' ', header=hdr, comments='')
        
        def femdata(self, freq, solver=utils.get_solver(), opt=utils.SolverOptions()):
            '''
            Calculates fluences for each source using a FEM solver, and then the boudary measurables for each channel 
            See nirfasteruff.forward.femdata_stnd_CW and nirfasteruff.forward.femdata_stnd_FD for details
            
            data, info = mesh.femdata(freq, solver=utils.get_solver(), opt=utils.SolverOptions())
            
            Input:
                freq: modulation frequency in Hz. If CW, set to zero and a more efficient CW solver will be used
                solver (optional): Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
                opt (optional): Solver options. See nirfasteruff.utils.SolverOptions for details
            Output:
                data: a nirfasteruff.base.FDdata object containing the fluence and measurables. See nirfasteruff.base.FDdata for details.
                info: a nirfasteruff.utils.ConvergenceInfo object containing the convergence information. See nirfasteruff.utils.ConvergenceInfo for details
                
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            # freq in Hz
            if freq==0:
                data, info = forward.femdata_stnd_CW(self, solver, opt)
                return data, info
            else:
                data, info = forward.femdata_stnd_FD(self, freq, solver, opt)
                return data, info
            
        def isvol(self):
            '''
            Check if convertion matrices between mesh and volumetric spaces are calculated
            
            isvol = mesh.isvol()
            
            Input: None
            Output: bool
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            if len(self.vol.xgrid):
                return True
            else:
                return False
            
        def gen_intmat(self, xgrid, ygrid, zgrid=[]):
            '''
            Calculate the information needed to convert data between mesh and volumetric space, specified by x, y, z (if 3D) grids.
            All grids must be uniform. The results will from a nirfasteruff.base.meshvol object stored in field .vol
            If field .vol already exists, it will be calculated again, and a warning will be thrown
            
            See nirfasteruff.base.meshvol for details
            
            Input:
                xgrid: x grid in mm
                ygrid: y grid in mm
                zgrid: z grid in mm. Leave empty for 2D meshes.
            Output:
                None
            
            J Cao, MILAB@UoB, 2024, NIRFASTerFF
            '''
            xgrid = np.float64(np.array(xgrid).squeeze())
            ygrid = np.float64(np.array(ygrid).squeeze())
            zgrid = np.float64(np.array(zgrid).squeeze())
            tmp = np.diff(xgrid)
            if np.any(tmp-tmp[0]):
                print('Error: xgrid must be uniform')
                return
            tmp = np.diff(ygrid)
            if np.any(tmp-tmp[0]):
                print('Error: ygrid must be uniform')
                return
            if len(zgrid)>0:
                tmp = np.diff(zgrid)
                if np.any(tmp-tmp[0]):
                    print('Error: zgrid must be uniform')
                    return
                
            if self.isvol():
                print('Warning: recalculating intmat', flush=1)
            if len(zgrid)==0:
                X, Y = np.meshgrid(xgrid, ygrid)
                coords = np.c_[X.flatten('F'), Y.flatten('F')]
            else:
                X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
                coords = np.c_[X.flatten('F'), Y.flatten('F'), Z.flatten('F')]
            # DT = spatial.Delaunay(self.nodes)
            # ind, int_func = nirfasteruff_cpu.pointLocation(np.float64(DT.simplices+1), DT.points, np.atleast_2d(coords))
            ind, int_func = nirfasteruff_cpu.pointLocation(self.elements, self.nodes, np.atleast_2d(coords*1.0))
            inside = np.flatnonzero(ind>-1) # This is zero-based
            int_func_inside = int_func[inside, :]
            # nodes = np.int32(DT.simplices[ind[inside],:])
            nodes = np.int32(self.elements[ind[inside],:] - 1)
            int_mat = sparse.csc_matrix((int_func_inside.flatten('F'), (np.tile(inside, int_func.shape[1]), nodes.flatten('F'))), shape=(ind.size, self.nodes.shape[0]))
            self.vol.xgrid = xgrid
            self.vol.ygrid = ygrid
            if len(zgrid)>0:
                self.vol.zgrid = zgrid
                self.vol.res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0]])
            else:
                self.vol.res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0]])
            self.vol.mesh2grid = int_mat
            self.vol.gridinmesh = inside + 1 # convert to one-based to be compatible with matlab
            
            # Now calculate the transformation from grid to mesh
            # We can cheat a little bit because of the regular grid: we can triangularize one voxel and replicate
            if len(zgrid)>0:
                start = np.array([xgrid[0], ygrid[0], zgrid[0]])
                nodes0 = np.array([[0,0,0], [0, self.vol.res[1], 0], 
                                   [self.vol.res[0], 0, 0], [self.vol.res[0],self.vol.res[1],0], 
                                   [0,0,self.vol.res[2]], [0, self.vol.res[1], self.vol.res[2]], 
                                   [self.vol.res[0], 0, self.vol.res[2]], [self.vol.res[0],self.vol.res[1],self.vol.res[2]]])
                # hard-coded element list
                ele0 = np.array([[2,5,6,3],
                                 [5,7,6,3],
                                 [3,2,5,1],
                                 [1,2,5,4],
                                 [0,2,1,4],
                                 [5,2,6,4]], dtype=np.int32)
                # Calculate integration function within the small cube
                loweridx = np.floor((self.nodes - start) / self.vol.res)
                pos_in_cube = self.nodes - (loweridx * self.vol.res + start)
                ind0, int_func0 = nirfasteruff_cpu.pointLocation(np.float64(ele0+1), nodes0, pos_in_cube)
                # Convert back to the node numbering of the full grid
                raw_idx = np.zeros((self.nodes.shape[0], 4))
                for i in range(self.nodes.shape[0]):
                    cube_coord = nodes0 + (loweridx[i,:] * self.vol.res)
                    tet_vtx = cube_coord[ele0[ind0[i], :], :]
                    rel_idx = (tet_vtx - start) / self.vol.res
                    raw_idx[i,:] = rel_idx[:,2]*len(xgrid)*len(ygrid) + rel_idx[:,0]*len(xgrid) + rel_idx[:,1] # zero-based
                
                outvec = (loweridx[:,0]<0) | (loweridx[:,1]<0) | (loweridx[:,2]<0)
                inside = np.flatnonzero(~outvec)
                # if any of the queried nodes was not asigned a value in the previous step,
                # treat it as an outside node and extrapolate. Otherwise the boundary elements will have smaller values than they should
                tmp = raw_idx[inside, :]
                tmp2 = np.isin(tmp, self.vol.gridinmesh-1)
                outside = np.r_[np.flatnonzero(outvec), inside[tmp2.sum(axis=1)<tmp2.shape[1]]]
                inside = np.array(list(set(inside) - set(outside)))
            else:
                start = np.array([xgrid[0], ygrid[0]])
                nodes0 = np.array([[0,0], [0, self.vol.res[1]], 
                                   [self.vol.res[0], 0], [self.vol.res[0],self.vol.res[1]]])
                # hard-coded element list
                ele0 = np.array([[2,1,0],
                                 [1,2,3]], dtype=np.int32)
                # Calculate integration function within the small cube
                loweridx = np.floor((self.nodes - start) / self.vol.res)
                pos_in_cube = self.nodes - (loweridx * self.vol.res + start)
                ind0, int_func0 = nirfasteruff_cpu.pointLocation(np.float64(ele0+1), nodes0, pos_in_cube)
                # Convert back to the node numbering of the full grid
                raw_idx = np.zeros((self.nodes.shape[0], 3))
                for i in range(self.nodes.shape[0]):
                    cube_coord = nodes0 + (loweridx[i,:] * self.vol.res)
                    tet_vtx = cube_coord[ele0[ind0[i], :], :]
                    rel_idx = (tet_vtx - start) / self.vol.res
                    raw_idx[i,:] = rel_idx[:,0]*len(ygrid) + rel_idx[:,1] # zero-based
                
                outvec = (loweridx[:,0]<0) | (loweridx[:,1]<0)
                inside = np.flatnonzero(~outvec)
                # if any of the queried nodes was not asigned a value in the previous step,
                # treat it as an outside node and extrapolate. Otherwise the boundary elements will have smaller values than they should
                tmp = raw_idx[inside, :]
                tmp2 = np.isin(tmp, self.vol.gridinmesh-1)
                outside = np.r_[np.flatnonzero(outvec), inside[tmp2.sum(axis=1)<tmp2.shape[1]]]
                inside = np.array(list(set(inside) - set(outside)))
            
            gridTree = spatial.KDTree(coords[self.vol.gridinmesh-1, :])
            _,nn = gridTree.query(self.nodes[outside,:])
            int_func_inside = int_func0[inside, :]
            nodes = np.int64(raw_idx[inside,:])
            int_mat = sparse.csc_matrix((np.r_[int_func_inside.flatten('F'), np.ones(nn.size)], 
                                         (np.r_[np.tile(inside, int_func.shape[1]), outside], np.r_[nodes.flatten('F'), self.vol.gridinmesh[nn]-1])), shape=(ind0.size, coords.shape[0]))
            self.vol.grid2mesh = int_mat
            self.vol.meshingrid = inside + 1 # convert to one-based

class math:
    '''
    Dummy class holding some low-level functions. Be careful using them: they interact closely with the C++ functions and wrong arguments used can cause unexpected crashes.
    Dummy class used so the function hierarchy can be compatible with the full version
    '''   
    def gen_mass_matrix(mesh, freq, solver = utils.get_solver(), GPU = -1):
        '''
        Calculate the MASS matrix, and return the coordinates in CSR format.
        The current Matlab version outputs COO format, so the results are NOT directly compatible
        If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU
        
        csrI, csrJ, csrV = gen_mass_matrix(mesh, freq, solver = utils.get_solver(), GPU = -1)
        
        Input:
            mesh: a nirfasteruff.base.stnd_mesh object
            freq: modulation frequency, in Hz
            solver (optional): choose between the GPU and CPU solver by specifying 'GPU' or 'CPU' (case insensitive).
                                Automatic selection if left unspecified, in which case GPU will take priority
            GPU (default=-1): GPU selection. -1 for automatic, 0, 1, ... for manual selection on multi-GPU systems
        Output:
            csrI: (uint32 NumPy array) I indices of the MASS matrix, in CSR format. Size (NNodes,)
            csrJ: (uint32 NumPy array) J indices of the MASS matrix, in CSR format. Size (nnz(MASS),)
            csrV: (float64/complex128 NumPy array) values of the MASS matrix, in CSR format. Size (nnz(MASS),)
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if solver.lower()=='gpu' and not utils.isCUDA():
            solver = 'CPU'
            print('Warning: No capable CUDA device found. using CPU instead')
            
        if solver.lower()=='gpu' and utils.isCUDA():
            try:
                [csrI, csrJ, csrV] = nirfasteruff_cuda.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, freq, GPU)
                if freq==0:
                    csrV = np.real(csrV)
            except:
                print('Warning: GPU code failed. Rolling back to CPU code')
                try:
                    [csrI, csrJ, csrV] = nirfasteruff_cpu.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, freq)
                    if freq==0:
                        csrV = np.real(csrV)    
                except:
                    print('Error: couldn''t generate mass matrix')
                    return 0, 0, 0
        elif solver.lower()=='cpu':
            try:
                [csrI, csrJ, csrV] = nirfasteruff_cpu.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, freq)
                if freq==0:
                    csrV = np.real(csrV) 
            except:
                print('Error: couldn''t generate mass matrix')
                return 0, 0, 0
        else:
            print('Error: Solver should be ''GPU'' or ''CPU''')
            return 0, 0, 0
            
        return csrI, csrJ, np.ascontiguousarray(csrV)
    
    def get_field_CW(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver()):
        '''
        Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. For CW data only.
        The current Matlab version uses COO format input, so they are NOT directly compatible
        If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
        On GPU, the algorithm first tries to solve for all sources simultaneously, but this can fail due to insufficient GPU memory.
        If this is the case, it will generate a warning and solve the sources one by one. The latter is not as fast, but requires much less memory.
        On CPU, the algorithm only solves the sources one by one.
        
        phi, info = get_field_CW(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver())
        
        Input:
            csrI, csrJ, csrV: MASS matrix in CSR format. See nirfasteruff.math.gen_mass_matrix for details
            qvec: (float64 NumPy array, or Scipy CSC sparse matrix) source vectors.
            opt (optional): Solver options. See nirfasteruff.utils.SolverOptions for details
            solver (optional): choose between the GPU and CPU solver by specifying 'GPU' or 'CPU' (case insensitive).
                                Automatic selection if left unspecified, in which case GPU will take priority
        Output:
            phi: (float64 NumPy array) Calculated fluence at each source. Size (NNodes, Nsources)
            info: a nirfasteruff.utils.ConvergenceInfo object containing the convergence information
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if not (np.isreal(csrV).all() and np.isrealobj(qvec)):
            raise TypeError('MASS matrix and qvec should be both real')
        if solver.lower()=='gpu' and not utils.isCUDA():
            solver = 'CPU'
            print('Warning: No capable CUDA device found. using CPU instead')
            
        if solver.lower()=='gpu' and utils.isCUDA():
            try:
                [phi, info] = nirfasteruff_cuda.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
            except:
                print('Warning: GPU solver failed. Rolling back to CPU solver')
                try:
                    [phi, info] = nirfasteruff_cpu.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence)  
                except:
                    print('Error: solver failed')
                    return np.array([]), []
        elif solver.lower()=='cpu':
            try:
                [phi, info] = nirfasteruff_cpu.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence)  
            except:
                print('Error: solver failed')
                return np.array([]), []
        else:
            print('Error: Solver should be ''GPU'' or ''CPU''')
            return np.array([]), []
            
        return phi, utils.ConvergenceInfo(info)
    
    def get_field_FD(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver()):
        '''
        Call the Preconditioned BiConjugate Stablized solver with FSAI preconditioner. For FD data only
        The current Matlab version uses COO format input, so they are NOT directly compatible
        If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
        On GPU, the algorithm first tries to solve for all sources simultaneously, but this can fail due to insufficient GPU memory.
        If this is the case, it will generate a warning and solve the sources one by one. The latter is not as fast, but requires much less memory.
        On CPU, the algorithm only solves the sources one by one.
        
        phi, info = get_field_FD(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver())
        
        Input:
            csrI, csrJ, csrV: MASS matrix in CSR format. See nirfasteruff.math.gen_mass_matrix for details
            qvec: (complex128 NumPy array, or Scipy CSC sparse matrix) source vectors
            opt (optional): Solver options. See nirfasteruff.utils.SolverOptions for details
            solver (optional): choose between the GPU and CPU solver by specifying 'GPU' or 'CPU' (case insensitive).
                                Automatic selection if left unspecified, in which case GPU will take priority
        Output:
            phi: (complex128 NumPy array) Calculated fluence at each source. Size (NNodes, Nsources)
            info: a nirfasteruff.utils.ConvergenceInfo object containing the convergence information
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if not np.all(np.iscomplex(csrV).all() and np.iscomplexobj(qvec)):
            raise TypeError('MASS matrix and qvec should be both complex')
        if solver.lower()=='gpu' and not utils.isCUDA():
            solver = 'CPU'
            print('Warning: No capable CUDA device found. using CPU instead')
            
        if solver.lower()=='gpu' and utils.isCUDA():
            try:
                [phi, info] = nirfasteruff_cuda.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
            except:
                print('Warning: GPU solver failed. Rolling back to CPU solver')
                try:
                    [phi, info] = nirfasteruff_cpu.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence)  
                except:
                    print('Error: solver failed')
                    return np.array([]), []
        elif solver.lower()=='cpu':
            try:
                [phi, info] = nirfasteruff_cpu.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence)  
            except:
                print('Error: solver failed')
                return np.array([]), []
        else:
            print('Error: Solver should be ''GPU'' or ''CPU''')
            return np.array([]), []
            
        return phi, utils.ConvergenceInfo(info)
    
    def gen_sources(mesh):
        '''
        Calculate the source vectors (point source only) for the sources in mesh.source field
        
        qvec = gen_sources(mesh)
        
        Input:
            mesh: a nirfasteruff.base.stnd_mesh object
        Output:
            qvec: (complex128 Scipy CSC sparse matrix) source vectors. Size (NNodes, Nsources)
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        link = copy.deepcopy(mesh.link)
        active = np.unique(link[link[:,2]==1,0]) - 1
        qvec = np.zeros((mesh.nodes.shape[0], active.size), dtype=np.complex128)
        if len(mesh.source.int_func) == 0:
            [ind, int_func] = utils.pointLocation(mesh, mesh.source.coord)
            print('int function calculated')
        else:
            ind = np.int32(mesh.source.int_func[:, 0]) - 1 # to zero-indexing
            int_func = mesh.source.int_func[:, 1:]
        for i in range(active.size):
            src = active[i]
            qvec[np.int32(mesh.elements[ind[src],:]-1), i] = int_func[src,:]
        return sparse.csc_matrix(qvec)
    
    def get_boundary_data(mesh, phi):
        '''
        Calculate the measured fluence at each channel
        
        data = get_boundary_data(mesh, phi)
        
        Input:
            mesh: a nirfasteruff.base.stnd_mesh object
            phi: (float64/complex128 NumPy array) fluence as is calculated by the CW or FD solver. 
                    See nirfasteruff.math.get_field_CW and nirfasteruff.math.get_field_FD
        Output:
            data: (float64/complex128 NumPy array) measured boundary data at each channel. Size (NChannels,)
            
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if len(mesh.meas.int_func) == 0:
            print('Calculating missing detectors integration functions.')
            ind, int_func = utils.pointLocation(mesh, mesh.meas.coord)
        else:
            ind = np.int32(mesh.meas.int_func[:,0]) - 1
            int_func = mesh.meas.int_func[:,1:]
        
        link = copy.deepcopy(mesh.link)
        link[:,:2] -= 1  # to zero-indexing
        active_src = list(np.unique(link[link[:,2]==1,0]))
        bnd = mesh.bndvtx>0
        data = np.zeros(link.shape[0], dtype=np.complex128)
        for i in range(link.shape[0]):
            if link[i,2]==0:
                data[i] = np.nan
                continue
            tri = list(np.int32(mesh.elements[ind[link[i,1]], :] - 1))
            int_func_tmp = int_func[link[i,1],:] * bnd[tri]
            int_func_tmp /= int_func_tmp.sum()
            data[i] = int_func_tmp.dot(phi[tri, active_src.index(link[i,0])])
        
        return data
            

class forward:
    '''
    Dummy class holding the forward modeling functions.
    Dummy class used so the function hierarchy can be compatible with the full version
    ''' 
    def femdata_stnd_CW(mesh, solver = utils.get_solver(), opt = utils.SolverOptions()):
        '''
        Forward modeling for CW. Please consider using mesh.femdata(0) instead.
        The function calculates the MASS matrix, the source vectors, and calls the CW solver.
        
        data, info = femdata_stnd_CW(mesh, solver = utils.get_solver(), opt = utils.SolverOptions())
        
        See nirfasteruff.base.stnd_mesh.femdata, nirfasteruff.math.get_field_CW, nirfasteruff.math.gen_mass_matrix, and nirfasteruff.math.gen_sources for details.
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        print("Calculating the MASS matrix", flush=1)
        csrI, csrJ, csrV = math.gen_mass_matrix(mesh, 0., solver, opt.GPU)
        qvec = math.gen_sources(mesh)
        qvec = np.abs(qvec)
        data = base.FDdata()
        print("Solving the system", flush=1)
        data.phi, info = math.get_field_CW(csrI, csrJ, csrV, qvec, opt, solver)
        data.complex = math.get_boundary_data(mesh, data.phi)
        data.link = copy.deepcopy(mesh.link)
        data.amplitude = np.abs(data.complex)
        data.phase = np.zeros(data.complex.size)
        data.phase[np.isnan(data.complex)] = np.nan
        data.vol = copy.deepcopy(mesh.vol)
        
        if len(mesh.vol.xgrid)>0:
            if len(mesh.vol.zgrid)>0:
                tmp = np.reshape(mesh.vol.mesh2grid.dot(data.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, qvec.shape[1]), order='F')
            else:
                tmp = np.reshape(mesh.vol.mesh2grid.dot(data.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, qvec.shape[1]), order='F')
            data.phi = tmp
        
        return data, info
    
    def femdata_stnd_FD(mesh, freq, solver = utils.get_solver(), opt = utils.SolverOptions()):
        '''
        Forward modeling for FD. Please consider using mesh.femdata(freq) instead. freq in Hz
        The function calculates the MASS matrix, the source vectors, and calls the FD solver.
        
        data, info = femdata_stnd_FD(mesh, freq, solver = utils.get_solver(), opt = utils.SolverOptions())
        
        See nirfasteruff.base.stnd_mesh.femdata, nirfasteruff.math.get_field_FD, nirfasteruff.math.gen_mass_matrix, and nirfasteruff.math.gen_sources for details.
        
        J Cao, MILAB@UoB, 2024, NIRFASTerFF
        '''
        if freq==0:
            print('Warning: Use femdata_stnd_CW for better performance')
        freq = freq * 2 * np.pi
        print("Calculating the MASS matrix", flush=1)
        csrI, csrJ, csrV = math.gen_mass_matrix(mesh, freq, solver, opt.GPU)
        qvec = math.gen_sources(mesh)
        data = base.FDdata()
        print("Solving the system", flush=1)
        data.phi, info = math.get_field_FD(csrI, csrJ, csrV, qvec, opt, solver)
        data.complex = math.get_boundary_data(mesh, data.phi)
        data.link = copy.deepcopy(mesh.link)
        data.amplitude = np.abs(data.complex)
        data.phase = np.angle(data.complex)*180./np.pi
        data.vol = copy.deepcopy(mesh.vol)
        
        if len(mesh.vol.xgrid)>0:
            if len(mesh.vol.zgrid)>0:
                tmp = np.reshape(mesh.vol.mesh2grid.dot(data.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, qvec.shape[1]), order='F')
            else:
                tmp = np.reshape(mesh.vol.mesh2grid.dot(data.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, qvec.shape[1]), order='F')
            data.phi = tmp
        
        return data, info
