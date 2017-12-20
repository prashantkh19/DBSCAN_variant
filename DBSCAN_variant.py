import numpy as np
import math

# function to read input data file
def read(filename):
    with open(filename) as fp:
        content = fp.readlines()  # read from the file

    content = [item.replace("\n", "") for item in content]
    content = [item.split(" ") for item in content]

    return content


# function to calculate euclidean distance between two points X and Y of dimension n
def euclidean_distance(X, Y, n):
    distance = 0
    for i in range(n):
        distance += math.pow((float(X[i]) - float(Y[i])), 2)
    distance = math.sqrt(distance)
    return distance


# function to cal Similarity Matrix
def getSimilarityMatrix(data):
    length = len(data)
    matrix = np.zeros(shape=(length, length))
    for m in range(length):
        for n in range(length):
            matrix.itemset((m,n),euclidean_distance(data[m],data[n],len(data[m])))
    return matrix


# function to get Sparsified Matrix from similarlity matrix
def get_Sparsified_Matrix(matrix,k):
    length = len(matrix[0])
    distances = []
    dummy = []
    sparsified_matrix = []
    for i in range(length):
        for j in range(length):
            distances.append(matrix.item(i, j))
            dummy.append(matrix.item(i, j))
        distances.sort()
        print(distances)
        neighbors = []
        for x in range(1, k + 1):
            val = dummy.index(distances[x]) + 1

            # checking for duplicates in dummy list
            val = check(val, neighbors, dummy, distances[x], 0)
            neighbors.append(val)
        neighbors.sort()
        sparsified_matrix.append(neighbors)
        distances = []
        dummy = []
    return sparsified_matrix


def check(val, neighbors, dummy, distance, i):
    if val in neighbors:
        i = i+1
        dummy2 = list(dummy)
        dummy2.pop(val-i)
        val = dummy2.index(distance) + 1 + i
        val = check(val, neighbors, dummy2, distance, i)
        return val
    else:
        return val


# function to obtain clusters
def get_clusters(d, sparsified_matrix):

    # making a graph with edges relation as (u,v) in Edges if W(u,v)>=e
    cluster_graph = Graph(len(d))

    # adding edges
    for i in core_pts:
        for j in core_pts:
            if i in range(1, len(d) + 1):
                if j in graph.vertex_vector[i - 1]:
                    w = len(list(set(sparsified_matrix[i - 1]).intersection(sparsified_matrix[j - 1])))
                    if w >= e:
                        cluster_graph.add_edge(i, j)

    # print(cluster_graph.vertex_vector)

    clusters = []
    visited = []
    for i in range(1, cluster_graph.no_of_vertex + 1):
        if cluster_graph.vertex_vector[i - 1] != [] and i not in visited:
            for j in cluster_graph.DFS(i):
                visited.append(j)
            # print(visited)
            clusters.append(cluster_graph.DFS(i))
    return clusters


# function to get core points
def getCorePoints(point_density,MinPts):
    core_pts = []
    for i in range(len(point_density)):
        if point_density[i]>=MinPts:
            core_pts.append(i+1)
    return core_pts


# function to get noise points
def getNoisePtIndices(core_pts,sparsified_matrix,e):
    noise = []
    for i in range(1, len(d)+1):
        if i not in core_pts:
            isQ = False
            for j in graph.vertex_vector[i-1]:
                w = len(list(set(sparsified_matrix[i - 1]).intersection(sparsified_matrix[j - 1])))
                if w >= e:
                    isQ = True
            if isQ == False:
                noise.append(i)
    return noise


# function to get border points
def getBorderPointIndices(noise_pts, core_pts):
    border_points=[]
    for i in range(1, len(d) + 1):
        if i not in core_pts and i not in noise_pts:
            border_points.append(i)
    return border_points


# function to add value after ref in list
def add_pt(list, ref, value):
    new_list=[]
    for i in list:
        new_list.append(i)
        if i == ref:
            new_list.append(value)

    return new_list


# function to update cluster with border points
def update_clusters(clusters, sparsified_matrix, border_pts):
    clusters_dummy = clusters
    for m in border_pts:
        max_val = 0
        max_i = 0
        max_k = 0
        for i in range(len(clusters)):
            for k in clusters[i]:
                w = len(list(set(sparsified_matrix[k - 1]).intersection(sparsified_matrix[m - 1])))
                if(max_val<w):
                    max_val = w
                    max_i = i
                    max_k = k
                # print(w)
        # print("max val ")
        # print(max_val)
        # print("max i ")
        # print(max_i)
        # print("max k ")
        # print(max_k)
        clusters_dummy[max_i] = add_pt(clusters_dummy[max_i],max_k,m)
        clusters_dummy[max_i].sort()
    return clusters_dummy

##########################################################################################################

#Input data
d = read('Test_Cases/t1.in')
values = d[0]
del d[0]

# Parameter from input
k = int(values[2])
e = int(values[3])
MinPts=int(values[4])

# print("Test data: ")
# print(d)

sim_matrix = getSimilarityMatrix(d)
print("Similarity Matrix: ")
print(sim_matrix)

sparsified_matrix = get_Sparsified_Matrix(sim_matrix,k)
print("Sparsified Matrix: ")
print(sparsified_matrix)

# Creating Shared Neighbour Graph
from graph_imp import Graph
import operator

# Initializing new graph
graph = Graph(len(d))

# Adding edges w.r.t. given relation
s_W = []
f_W_list = []
density = []

for i in range(1,len(d)+1):
    count = 0
    for x in range(k):
        val = sparsified_matrix[i-1][x]
        # checking val for condition
        if (i) in sparsified_matrix[val - 1]:
            w = len(list(set(sparsified_matrix[val - 1]).intersection(sparsified_matrix[i-1])))
            if w>=e:
                count=count+1
            s_W.append((val, w))
            s_W.sort(key=operator.itemgetter(1), reverse=True)
            graph.add_edge(i , val)
    f_W_list.append(s_W)
    s_W = []
    density.append(count)

# print("Edges of graph:")
# print(graph.edges())

print("List with entry as (p,w)")
print(f_W_list)

print("Point Density of each point is: ")
print(density)

core_pts = getCorePoints(density,MinPts)
print("Core Points: ")
print(core_pts)


clusters = get_clusters(d,sparsified_matrix)
print("Clusters are: ")
print(clusters)

noise_pts = getNoisePtIndices(core_pts, sparsified_matrix, e)
print("Noise Points are: ")
print(noise_pts)

border_pts = getBorderPointIndices(noise_pts,core_pts)
print("Border Points are: ")
print(border_pts)

updated_cluster = update_clusters(clusters,sparsified_matrix,border_pts)
print("Updated Clusters are: ")
print(updated_cluster)