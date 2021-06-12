from EMST import *
import numpy as np
import scipy
  
class EMSTBasedClustering():
    def __init__(self, emst):
        '''
        emst: EMST object
        '''
        self.emst = emst
        self.nbVertices = self.emst.nbVertices
        
        self.nbClusters = 1
        self.slicedTrees = None
    
    def cutEdges(self, **kwargs):
        pass
    
    def getComponents(self):
        '''
        get all components of the emst
        '''
        self.visited = np.zeros(self.nbVertices, dtype=bool)
        self.emst._transformTree(tree=self.slicedTrees)
        isRoot = np.ones(self.nbVertices, dtype=bool)
        
        # loop through all parent-child tuples, set False in the isRoot list to all child nodes
        for tup in self.slicedTrees.keys():
            isRoot[tup[1]] = False
        
        self.components = [[] for _ in range(self.nbClusters)]
        c = 0
        for i in range(self.nbVertices):
            if isRoot[i]:
                self.getSubTree(i, self.components[c])                
                c += 1
                
        self.labels = np.array([None]*self.nbVertices)      
        for i in range(len(self.components)):
            for node in self.components[i]:
                self.labels[node] = i
                
    def getSubTree(self, root, subTree):
        '''
        Find subtree from the root using the depth first search method
        
        ======
        INPUTS
        ======
        root: integer, index of the starting node
        subTree: list, store the elements of the tree
        '''

        # keep track of node visited  
        self.visited[root] = 1 
        subTree.append(root)  

        for iteration in self.emst.mst_node_neighb[root]:  
            if not self.visited[iteration]:
                self.getSubTree(iteration, subTree)
        return

    def fit(self, **kwargs):
        self.cutEdges(**kwargs)
        self.getComponents()




class HEMST(EMSTBasedClustering):
    def cutEdges(self, sig=1):
        '''
        hierarchical EMST clustering algorithm by Grygorash et al.
        sig: strictly positive number, number of standard diviation away from the average edge length 
        that an edge is considered to be cut
        '''
        self.slicedTrees = self.emst.mst_pair_weight.copy()
        
        edgeWeights = np.array(list(self.emst.mst_pair_weight.values()))
        avgDist = edgeWeights.mean()
        stdDist = edgeWeights.std()
        
        
        self.toBeRemoved = [ key for key, value in self.emst.mst_pair_weight.items() if value > avgDist + sig * stdDist ]
        self.nbClusters = len(self.toBeRemoved) + 1
        
        for key in self.toBeRemoved:
            del self.slicedTrees[key]
 



class FAREMST(EMSTBasedClustering):
    '''
    Fused adaptive ridge based on EMST
    
    Class parametres
    ----------------
    emst: EMST object, euclidean minimun spanning tree
    delta: float. tiny positive value that allows to prevent division by 0
    tol: float. convergence criterion
    max_iter: int. convergence criterion (maximum number of iterations)
    
    Class method(s)
    -------------
    fit(self, lmbda): cut unwanted edges and get labels
    
    Class attributes
    ----------------
    nbClusters: number of clusters
    labels: label of each data
    '''
    
    def __init__(self, emst, delta=1e-5, tol=1e-3, max_iter=100):
        '''
        ======
        INPUTS
        ======
        delta: float, tiny positive number to prevent division by zero
        tol: float, tiny positive number, convergence criterion
        max_iter: int, number of maximum iterations allowed if convergence criterion is not reached
        '''
        EMSTBasedClustering.__init__(self, emst=emst)
        self.delta = delta
        self.tol = tol
        self.max_iter = max_iter
        self.X = emst.data
        
        
        
    def initialise(self):
        '''
        initialise mu, w, prepare for iterations
        '''
        self.emst._transformTree()
        self.mu = self.X
        self.w = {pair:1 for pair in self.emst.mst_pair_weight.keys()}
        self.pairwise_centroid_dist = np.array(list(self.emst.mst_pair_weight.values()))
        
    def updateA(self, lmbda):
        # create the sparse matrix
        self.A = np.zeros((self.nbVertices, self.nbVertices))
        for k,val in self.emst.mst_node_neighb.items():
            for v in val:
                try:
                    self.A[k][v] = self.w[(k,v)]
                except KeyError:
                    self.A[k][v] = self.w[(v,k)]
                
        sumOfLine = [l.sum() for l in self.A]
        self.A *= -lmbda
        
        # add diagonal
        self.A += np.eye(self.nbVertices)
        self.A += lmbda*np.diag(sumOfLine)
        
    def updateMu(self):
        self.last_mu = self.mu.copy()
        self.mu = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(self.A), self.X)
        
    def updateW(self):    
        for k,(i,j) in enumerate(self.emst.mst_pair_weight.keys()):
            self.pairwise_centroid_dist[k] = np.linalg.norm(self.mu[i] - self.mu[j])**2   
            self.w[(i,j)] = 1/(self.pairwise_centroid_dist[k] + self.delta**2)
    
    def cutEdges(self, lmbda=5):
        '''
        cut unwanted edges in the EMST
        ======
        INPUTS
        ======
        lmbda: float. penalty coefficient. the larger the penalty, the less the number of clusters
        '''
        # initialisation
        self.initialise()
        
        # iterations
        for iteration in range(self.max_iter):
            self.updateA(lmbda=lmbda)
            self.updateMu()
            self.updateW()
            
            
            if np.max(np.linalg.norm(self.mu-self.last_mu)/np.linalg.norm(self.last_mu)) < self.tol:
                # print('Early break, number of iteration: %d' % iteration)
                break
        
        # remove edges that have little weight
        self.toBeRemoved = [ key for k,key in enumerate(self.emst.mst_pair_weight.keys()) if self.pairwise_centroid_dist[k] * self.w[key] > 0.99]
        self.slicedTrees = self.emst.mst_pair_weight.copy()
        self.nbClusters =  len(self.toBeRemoved) + 1
        for key in self.toBeRemoved:
            del self.slicedTrees[key]
        



class FAREMST_Label(EMSTBasedClustering):
    '''
    Fused adaptive ridge based on EMST using binary labels
    
    Class parametres
    ----------------
    emst: EMST object, euclidean minimun spanning tree
    delta: float. tiny positive value that allows to prevent division by 0
    tol: float. convergence criterion
    max_iter: int. convergence criterion (maximum number of iterations)
    
    Class method(s)
    -------------
    fit(self, lmbda): cut unwanted edges and get labels
    
    Class attributes
    ----------------
    nbClusters: number of clusters
    labels: label of each data
    '''
    
    
    def __init__(self, emst, delta=1e-5, tol=1e-3, max_iter=100):
        '''
        delta: float, tiny positive number to prevent division by zero
        tol: float, tiny positive number, convergence criterion
        max_iter: int, number of maximum iterations allowed if convergence criterion is not reached
        '''
        EMSTBasedClustering.__init__(self, emst=emst)
        self.delta = delta
        self.tol = tol
        self.max_iter = max_iter
        self.X = emst.data
        
        
    def initialise(self, y):
        '''
        initialise mu, w, prepare for iterations
        
        ======
        INPUTS
        ======
        y: array of shape [nbData,], binary label of each data
        '''
        self.emst._transformTree()
        # convert binary values to 0.1 and 0.9 (numeric consideration)
        self.y_cp = y * 0.8 + 0.1
        
        self.mu = np.log(self.y_cp / (1 - self.y_cp))
        self.w = {pair:1 for pair in self.emst.mst_pair_weight.keys()}
        self.pairwise_diff = np.array(list(self.emst.mst_pair_weight.values()))
        
        
    def _updateNegHessian(self, lmbda, logit):
        # Hessian matrix 
        # H[i,i] = logit[i] * (logit[i] - 1) - lmbda * sum(w_ij (mu_i - mu_j), for j in {neighbor_i})
        # H[i,j] = 0    if i and j are NOT neighbors
        #        = lmbda * w_ij    else
        # here the negative hessian is calculated
        # Rk: the Hessian matrix is not semi-definite, so we can't tell the convexity of the function
        
        H = np.zeros((self.nbVertices,self.nbVertices))
        
        # diag[i] = logit[i] * (1 - logit[i]) + lmbda * sum(w_ij, for j in {neighbor_i})
        diag = logit * (1 - logit)
        
        for i in range(self.nbVertices):
            for j in self.emst.mst_node_neighb[i]:
                try:
                    H[i,j] = -lmbda * self.w[(i,j)]
                    diag[i] += lmbda * self.w[(i,j)]
                except KeyError:
                    H[i,j] = -lmbda * self.w[(j,i)]
                    diag[i] += lmbda * self.w[(j,i)]
                    
        H += np.diag(diag)
        
        return H
        
     
    def _updateGradient(self, y, lmbda, logit):
        # gradient 
        # g[i] = y[i] - logit[i] - lmbda * tmp
        # tmp[i] = sum(w_ij (mu_i - mu_j), for j in {neighbor_i})
        tmp = np.zeros_like(y, dtype=float)
        for i in range(self.nbVertices):
            for j in self.emst.mst_node_neighb[i]:
                try:
                    tmp[i] += self.w[(i,j)] * (self.mu[i] - self.mu[j])
                except KeyError:
                    tmp[i] += self.w[(j,i)] * (self.mu[i] - self.mu[j])

        return y - logit - lmbda * tmp
        
    
    def updateMu(self, y, lmbda):
        self.last_mu = self.mu.copy()
        
        # Newton-Raphson loop
        #   20 iterations seem to be sufficient for now. Can be adjusted if needed.
        for _ in range(20):    
                
            # logit[i] = P(y_i = 1) 
            logit = 1 / (1 + np.exp(-self.mu))
            
            # get ready for the Newton-Raphson's algo: compute the gradient and the Hessian matrix
            grad = self._updateGradient(y, lmbda, logit)
            negHess = self._updateNegHessian(lmbda, logit)

            # update mu until convergence by the Newton's method
            self.mu += scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(negHess), grad)
        
        
    def updateW(self):    
        for k,(i,j) in enumerate(self.emst.mst_pair_weight.keys()):
            self.pairwise_diff[k] = np.abs(self.mu[i] - self.mu[j])  
            self.w[(i,j)] = 1/(self.pairwise_diff[k]**2 + self.delta**2)
    
    def cutEdges(self, y, lmbda=5):
        '''
        cut unwanted edges in the EMST
        ======
        INPUTS
        ======
        y: array of shape [nbData,], binary label of each data
        lmbda: float. penalty coefficient. the larger the penalty, the less the number of clusters
        '''
        # initialisation
        self.initialise(y)
        
        # iterations
        for iteration in range(self.max_iter):
            self.updateMu(y=y, lmbda=lmbda)
            self.updateW()
            
            if np.max(np.linalg.norm(self.mu-self.last_mu)/np.linalg.norm(self.last_mu)) < self.tol:
                # print('Early break, number of iteration: %d' % (iteration+1))
                break
        
        # remove edges that have little weight
        self.toBeRemoved = [ key for k,key in enumerate(self.emst.mst_pair_weight.keys()) if self.pairwise_diff[k]**2 * self.w[key] > 0.99]
        self.slicedTrees = self.emst.mst_pair_weight.copy()
        self.nbClusters =  len(self.toBeRemoved) + 1
        for key in self.toBeRemoved:
            del self.slicedTrees[key]
        