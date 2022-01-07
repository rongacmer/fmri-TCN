
from queue import PriorityQueue
class minimum_tree:
    def __init__(self,matrix):
        self.num_nodes = matrix.shape[0]
        self.matrix = matrix
        self.q = PriorityQueue()
        self.node = set() #点集

    def init_stage(self):
        self.node.add(0)
        start = 0
        for i in range(self.num_nodes):
            if i not in self.node:
                self.q.put_nowait((self.matrix[start,i],start,i))

    def create_tree(self):
        edges = []
        self.init_stage()

        while not self.q.empty():
            now_edges = self.q.get_nowait()
            if now_edges[2] in self.node:
                continue
            edges.append(now_edges[0])
            self.node.add(now_edges[2])
            for i in range(self.num_nodes):
                if i not in self.node:
                    self.q.put_nowait((self.matrix[now_edges[2],i],now_edges[2],i))
        edges.append(0)
        edges = list(edges)
        edges.sort()
        print('the numeber of edges:{}'.format(len(edges)))
        return edges
