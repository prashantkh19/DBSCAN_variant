from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


##########################################################################################################
# Test Data from input file
def read(filename):
    with open(filename) as fp:
        content = fp.readlines()  # read from the file
        # print(content)

    content = [item.replace("\n", "") for item in content]
    content = [item.split(" ") for item in content]

    return content


d = read('input.data')
values = d[0]
del d[0]

print("Test data: ")
print(d)

###########################################################################################################
# Similarity Matrix

matrix = np.asmatrix(euclidean_distances(d, d))
print("Similarity Matrix: ")
print(matrix)
# print(matrix[0].item(2))

##########################################################################################################3
# Sparsifying the matrix

# Parameter from input
k = int(values[2])
e = int(values[3])

length = len(d)
distances = []
dummy = []
sparsified_matrix = []
for i in range(length):
    for j in range(length):
        distances.append(matrix.item(i, j))
        dummy.append(matrix.item(i, j))
    # print("distances: ")
    # print(distances)
    # print("dummy: ")
    # print(dummy)
    distances.sort()
    # print("sorted: ")
    # print(distances)
    neighbors = []
    for x in range(1, k + 1):
        # print(distances[x])
        # print(dummy.index(distances[x]))
        neighbors.append(dummy.index(distances[x]) + 1)
    # print(neighbors)
    neighbors.sort()
    sparsified_matrix.append(neighbors)
    distances = []
    dummy = []

print("Sparsified Matrix: ")
print(sparsified_matrix)

# ##############################################################################################################
# Creating Shared Neighbour Graph

from networkx import Graph

if __name__ == "__main__":

    # Initializing new graph
    graph = Graph()

    # print("Edges of graph:")
    # print(graph.edges())

# Adding edges w.r.t. given relation
import operator

s_W = []
f_W_list = []
density = []

for i in range(length):
    count = 0
    for x in range(k):
        val = sparsified_matrix[i][x]
        # checking val for condition
        if (i + 1) in sparsified_matrix[val - 1]:
            w = len(list(set(sparsified_matrix[val - 1]).intersection(sparsified_matrix[i])))
            if w>=e:
                count=count+1
            s_W.append((val, w))
            s_W.sort(key=operator.itemgetter(1), reverse=True)
            graph.add_edge(i + 1, val)
    f_W_list.append(s_W)
    s_W = []
    density.append(count)


# print("Edges of graph:")
# print(graph.edges())

print("List with entry as (p,w)")
print(f_W_list)

#########################################################################################################
# Calculating Density

print("Point Density of each point is: ")
print(density)

##########################################################################################################
# Core points
MinPts=int(values[4])

core_pts = []
for i in range(len(density)):
    if density[i]>=MinPts:
        core_pts.append(i+1)
print("Core Points: ")
print(core_pts)

###########################################################################################################
# Forming Clusters
print("testing: ")

for i in core_pts:
    for j in core_pts:
       if (i,j) in graph.edges:
           w = len(list(set(sparsified_matrix[i-1]).intersection(sparsified_matrix[j-1])))
           if w>=e:
               print((i,j))

