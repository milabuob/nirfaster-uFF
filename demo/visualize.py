import numpy as np
import matplotlib.pyplot as plt

def plot3dmesh(mesh, data, selector):
    if mesh.dimension!=3:
        print('Error: only 3D meshes are supported.')
        return
    ele = mesh.elements
    nodes = mesh.nodes
    x = np.mean(nodes[np.int32(ele-1),0], axis=1)
    y = np.mean(nodes[np.int32(ele-1),1], axis=1)
    z = np.mean(nodes[np.int32(ele-1),2], axis=1)
    
    # select the subset of elements
    idx = eval('np.nonzero(' + selector + ')[0]')
    
    faces = np.r_[ele[np.ix_(idx, [0,1,2])], 
                  ele[np.ix_(idx, [0,1,3])],
                  ele[np.ix_(idx, [0,2,3])],
                  ele[np.ix_(idx, [1,2,3])]]
    faces = np.sort(faces)
    # boundary faces: they are referred to only once
    faces,cnt=np.unique(faces,return_counts=1,axis=0)
    bndfaces=faces[cnt==1,:]
    
    # plot
    ax = plt.figure().add_subplot(projection='3d')
    h = ax.plot_trisurf(nodes[:,0], nodes[:,1], nodes[:,2], triangles=bndfaces-1, linewidth=0.2, edgecolor=[0.5,0.5,0.5], antialiased=True)
    colors = np.max(data[np.int32(bndfaces-1)], axis=1)
    h.set_array(colors)
    plt.show()
    return ax
    