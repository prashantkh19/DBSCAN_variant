
class Graph:
    no_of_vertex = 0
    no_of_edges = 0
    vertex_vector = []

    def __init__(self,vertex):
        self.no_of_vertex = vertex
        self.vertex_vector = []
        for i in range(1 , vertex+1):
            r = []
            self.vertex_vector.append(r)

    def add_edge(self,u,v):
        u=u-1
        r=[]
        r = self.vertex_vector[u]
        r.append(v)
        self.vertex_vector[u]=r

    def edges(self):
        edges = []
        for i in range(1, self.no_of_vertex+1):
            for j in range(len(self.vertex_vector[i-1])):
                 edges.append((i,self.vertex_vector[i-1][j]))
        return edges

    # A function used by DFS
    def DFSUtil(self, v, visited, related):

        # Mark the current node as visited and print it
        visited[v-1] = True
        related.append(v)
        # print(v)

        # Recur for all the vertices adjacent to this vertex
        for i in self.vertex_vector[v-1]:
            if visited[i-1] == False:
                self.DFSUtil(i, visited,related)

        # The function to do DFS traversal. It uses
        # recursive DFSUtil()

    def DFS(self, v):

        related = []
        # Mark all the vertices as not visited
        visited = [False] * (len(self.vertex_vector))

        # Call the recursive helper function to print
        # DFS traversal
        self.DFSUtil(v, visited,  related)
        return related