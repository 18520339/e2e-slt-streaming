import numpy as np

class Graph:
    def __init__(self, layout='hand21', strategy='uniform', max_hop=1, dilation=1):
        '''The Graph to model the skeletons extracted by the DWPose
    
        Args:
            strategy (string): must be one of the follow candidates
            - uniform: Uniform Labeling
            - distance: Distance Partitioning
            - spatial: Spatial Configuration
            For more information, please refer to the section 'Partition Strategies' in our paper (https://arxiv.org/abs/1801.07455).
            
            layout (string): must be one of the follow candidates
            - hand: 21 joints as defined in the DWPose model
            - body: 9 joints as defined in the Body model
            - mouth: 8 joints as defined in the Mouth model
            - face: 18 joints as defined in the Face model

            max_hop (int): the maximal distance between 2 connected nodes
            dilation (int): controls the spacing between the kernel points
        '''
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A


    def get_edge(self, layout):
        if layout in ['left_hand', 'right_hand']:
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                [0, 1],  [1, 2],   [2, 3],   [3, 4],
                [0, 5],  [5, 6],   [6, 7],   [7, 8],
                [0, 9],  [9, 10],  [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
                
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                [0, 1], [0, 2], [0, 3], [0, 4],
                [3, 5], [5, 7], [4, 6], [6, 8],
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
            
        elif layout == 'mouth':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [[i, i + 1] for i in range(self.num_node - 1)] + \
                            [[self.num_node - 1, 0]]
            self.edge = self_link + neighbor_link
            self.center = 4
            
        elif layout == 'face':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [[i, i + 1] for i in range(16)] + \
                            [[17, i] for i in range(17)]
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1
            

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
            
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
            
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
                    
            A = np.stack(A)
            self.A = A
        raise ValueError('Do Not Exist This Strategy')


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    num_node = A.shape[0]
    Dl = np.sum(A, 0)
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0: Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)