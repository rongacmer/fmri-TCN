import os
import numpy as np
import pandas as pd
import networkx as nx
import zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d
import matplotlib.pyplot as plt
import time
# from ripser import ripser
# from persim import plot_diagrams, PersImage
import multiprocessing
import argparse
parse = argparse.ArgumentParser()
parse.add_argument('--subject_dir',type=str)
config = parse.parse_args()


# parameter setting
alpha = 1e-3
NVertices = 90# Number of vertices
scaleParameter = 1.0 # Scale Parameter (Maximum) # the maximal edge weight
maxDimHoles = 2 # Maximum Dimension of Holes (It means.. 0 and 1)
sizeWindow = 12 # Number of Graphs

# Zigzag persistence diagram (ZPD) for the regular sliding window
def zigzag_persistence_diagrams(nameFolderNet, index,  NVertices, scaleParameter, maxDimHoles, sizeWindow,alpha):
    print("Loading data...")  # Beginning
    # Graphs = []
    GraphsNetX = []
    start_time = time.time()
    for i in range(index, index + sizeWindow):
        # edgesList = np.loadtxt(nameFolderNet+str(i+1)+".txt") # Load data
        # edgesList = np.loadtxt(nameFolderNet+".csv", delimiter=',') # Load data
        ArrMatrix = np.load(nameFolderNet + str(i) + ".npy")
        ArrMatrix = ArrMatrix[:NVertices, :NVertices]
        # Graphs.append(ArrMatrix)
        g = nx.Graph()
        # Generate Graph
        g.add_nodes_from(list(range(0, NVertices)))  # Add vertices...
        for start in range(NVertices):
            for end in range(NVertices):
                weight = ArrMatrix[start][end]
                if ArrMatrix[start][end] < alpha:
                    weight = 0
                else:
                    weight = 1 - ArrMatrix[start][end]
                g.add_edge(start, end, weight=weight) #创建图
        GraphsNetX.append(g)
    print("  --- End Loading...")  # Ending


    # Building unions and computing distance matrices
    print("Building unions and computing distance matrices...")  # Beginning
    GUnions = []
    MDisGUnions = []

    for i in range(0, sizeWindow - 1):
        # --- To concatenate graphs
        unionAux = []
        MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
        A = nx.adjacency_matrix(GraphsNetX[i]).todense()
        B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
        # ----- Version Original (2)
        C = (A + B) / 2
        A[A == 0] = 1.1
        A[range(NVertices), range(NVertices)] = 0
        B[B == 0] = 1.1
        B[range(NVertices), range(NVertices)] = 0
        MDisAux[0:NVertices, 0:NVertices] = A
        C[C == 0] = 1.1
        C[range(NVertices), range(NVertices)] = 0
        MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
        MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
        MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()
        # Distance in condensed form
        pDisAux = squareform(MDisAux) #距离矩阵
        # --- To save unions and distances
        GUnions.append(unionAux)  # To save union
        MDisGUnions.append(pDisAux)  # To save distance matrix
    print("  --- End unions...")  # Ending

    # To perform Ripser computations
    print("Computing Vietoris-Rips complexes...")  # Beginning

    GVRips = []
    for jj in range(0, sizeWindow - 1):
        # print(jj)
        ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter) #查找多少个单纯形'
        GVRips.append(ripsAux)
    print("  --- End Vietoris-Rips computation")  # Ending

    # Shifting filtrations...
    print("Shifting filtrations...")  # Beginning
    GVRips_shift = []
    GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
    for kk in range(1, sizeWindow - 1):
        shiftAux = zzt.shift_filtration(GVRips[kk], NVertices * kk)
        GVRips_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    # To Combine complexes
    print("Combining complexes...")  # Beginning
    completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
    for uu in range(2, sizeWindow - 1):
        completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
    print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
    print("  --- End time")  # Beginning

    # To compute Zigzag persistence
    print("Computing Zigzag homology...")  # Beginning
    G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals) #mark
    print("  --- End Zigzag")  # Beginning
    for i in G_dgms:
        for j in i:
            print(j)
    # To show persistence intervals
    window_ZPD = []
    # Personalized plot
    for vv, dgm in enumerate(G_dgms):
        print("Dimension:", vv)
        if (vv < 2):
            matBarcode = np.zeros((len(dgm), 2))
            k = 0
            for p in dgm:
                matBarcode[k, 0] = p.birth
                matBarcode[k, 1] = p.death
                k = k + 1
            matBarcode = matBarcode / 2
            window_ZPD.append(matBarcode)

    # Timing
    print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str(
        (time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")

    return window_ZPD





# Zigzag persistence image
def zigzag_persistence_images(dgms, resolution = [100,100], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
    PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # Compute zigzag persistence image
    P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    if normalization:
        norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
    else:
        norm_output = output

    return norm_output

def creat_wdp_image(subject_path,i,sizeWindow):
    wdp_filename = os.path.join(subject_path, 'WDP_index{}'.format(i))
    nameFolderNet = subject_path + 'P_index_'
    zdp = zigzag_persistence_diagrams(nameFolderNet, i, NVertices, scaleParameter, maxDimHoles, sizeWindow, alpha)
    output = zigzag_persistence_images(zdp)
    np.save(wdp_filename, output)
    print('{} save successful'.format(wdp_filename))
def creat_one_subject(subject_path, sizeWindow, time_length=130):
    '''
    计算一个subject的ZPD
    :param subject_path:
    :param time_length:
    :return:
    '''
    pool = multiprocessing.Pool(8)

    start_time = time.time()
    for i in range(0,time_length-sizeWindow):
        # creat_wdp_image(subject_path,i,sizeWindow)
        pool.apply_async(creat_wdp_image,(subject_path,i,sizeWindow))
    pool.close()
    pool.join()
    end_time = time.time()
    print('运行时间 {}mins'.format((end_time - start_time) / 60))


if __name__ == '__main__':
    creat_one_subject(config.subject_dir,sizeWindow)



