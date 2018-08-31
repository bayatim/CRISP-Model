


from __future__ import division

#import nest
import numpy as np
import pylab as pl
#import nest.topology as topo
#from sets import Set
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import random
import time
import copy
import scipy.stats
import scipy.misc
import scipy.signal
from matplotlib import cm
import cPickle as cPickle
import pickle as pickle
import matplotlib.image as mpimg
import matplotlib._png as png
import matplotlib.gridspec as gridspec
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy.optimize import curve_fit

import matplotlib.animation as animation

from sklearn.decomposition import PCA

from matplotlib.legend_handler import HandlerLine2D



#import distance.hamming as hammdis

#from cvxopt import matrix, solvers

#matplotlib Figure parameter
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'legend.handlelength': 1.})
matplotlib.rcParams.update({'legend.labelspacing': 0.1})


begin = time.time()

# useable functions#############################################################################################

def normalize(pattern): #normalize pattern over last dimension;

    dim =len(pattern.shape)

    if dim == 4:
        norms = np.sqrt(np.einsum('enpj,enpj->enp', pattern*1.,pattern*1.)).repeat(pattern.shape[-1]).reshape(pattern.shape)
        norms[norms == 0] =1
        pattern /= norms
        #I added these 2 lines
        #pattern = pattern/np.mean(np.sum(pattern, axis=1))

    if dim == 3:
        norms = np.sqrt(np.einsum('npj,npj->np', pattern*1.,pattern*1.)).repeat(pattern.shape[-1]).reshape(pattern.shape)
        norms[norms == 0] =1
        pattern /= norms
        #pattern = pattern/np.mean(np.sum(pattern, axis=1))        

    if dim ==2:
        pattern /= np.sqrt(np.einsum('pj,pj->p', pattern*1.,pattern*1.)).repeat(pattern.shape[1]).reshape(pattern.shape)
        #pattern = pattern/np.mean(np.sum(pattern, axis=1))

    if dim ==1:
        pattern/= np.sqrt(np.sum(pattern**2))
        #pattern /=np.mean(np.sum(pattern, axis=1))

def normalize_1norm(pattern):
    pattern -= np.einsum('ij -> i', pattern).repeat(pattern.shape[1]).reshape(pattern.shape)

def determineSparsity(pattern, binary = True): # return activity level of each pattern, patterns must be of dimension 2

    if binary:
        p = (pattern + 1)/2
    else:
        p = pattern
    activity = np.sum(p, -1)/(pattern.shape[-1]+0.0)
    return activity

def save_object(obj, path, info=True, compressed=True):
    ''' Saves an object to file.

    :Parameters:
        obj: object to be saved.
            -type: object

        path: Path and name of the file
             -type: string

        info: Prints statements if TRUE
             -type: bool

    '''
    if info == True:
        print '-> Saving File  ... ',
    try:
        if compressed:
            fp = gzip.open(path, 'wb')
            cPickle.dump(obj, fp)
            fp.close()
        else:
            file_path = open(path, 'w')
            cPickle.dump(obj, file_path)
            file_path.close()
        if info == True:
            print 'done!'
    except:
        print "-> File writing Error: "
        return None

def load_object(path, info=True, compressed=True):
    ''' Loads an object from file.

    :Parameters:
      path: Path and name of the file
           -type: string

      info: Prints statements if TRUE
           -type: bool

    :Returns:
        Loaded object
       -type: object

    
    if not os.path.isfile(path):
        if info is True:
            print "-> File not existing: " + path
        return None
    else:
	'''	
    if info is True:
		print '-> Loading File  ... ',
    #try:
    if compressed is True:
		fp = gzip.open(path, 'rb')
		obj = cPickle.load(fp)
		fp.close()
		if info is True:
			print 'done!'
		return obj
    else:
		file_path = open(path, 'r')
		obj = cPickle.load(file_path)
		file_path.close()
		if info is True:
			print 'done!'
		return obj
    #except:
	#	if info is True:
	#		print "-> File reading Error: "
	#	return None


class Random_walk(object):

    def __init__(self, width= None, height= None, step = 2.1):
        self.width = width
        self.height = height
        self.step = step
        self.current_position = [width*np.random.rand(),height*np.random.rand()]
        self.step_history = [0.0, 0.0]

    def get_next_position(self, momentum = None, reorientation = None):
        if np.random.rand()>reorientation:
            self.step_history[0]*=0
            self.step_history[1]*=0

        self.step_history[0] = momentum * self.step_history[0] + (1.0-momentum)*((np.random.rand() * 2.0) - 1.0)*self.step
        self.step_history[1] = momentum * self.step_history[1] + (1.0-momentum)*((np.random.rand() * 2.0) - 1.0)*self.step
        norm = np.sqrt(self.step_history[0] ** 2 + self.step_history[1] ** 2) / self.step
        self.step_history[0] /= norm
        self.step_history[1] /= norm
        self.current_position[0] += self.step_history[0]
        self.current_position[1] += self.step_history[1]
        if self.current_position[0] < 0.0:
            self.current_position[0] = 0.0
            self.step_history[0] *= 0.0
        if self.current_position[0] > self.width-1:
            self.current_position[0] = self.width-1
            self.step_history[0] *= 0.0
        if self.current_position[1] < 0.0:
            self.current_position[1] = 0.0
            self.step_history[1] *= 0.0
        if self.current_position[1] > self.height-1:
            self.current_position[1] = self.height-1
            self.step_history[1] *= 0.0
        return self.current_position


#save and load functions########################################################################


#####################################################################Networks########################################################################################################


class Network(object): # Generic network class

    def __init__(self, input_cells=None, cells=None, connectivity = None, learnrate= None, subtract_input_mean = None, subtract_output_mean = None, actFunction = None, number_winner=None, e_max = 0.1, active_in_env = None, n_e = 1, initMethod = None, initConnectionMethod = None,  weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5):

        '''
        abstract

        :param input_cells: number of input cells
        :type input_cells: int
        :param connectivity: proportion of input cells to which one outputcell is connected to
        :type connectivity: float in [0,1]
        :param learnrate: factor used for learning each pattern
        :type learnrate: float
        :param subtract_input_mean: determines whether input mean is subtracted by applying Hebbian learning+
        :param subtract_input_mean: bool
        :param subtract_output_mean: determines whether output mean is subtracted by applying Hebbian learning+
        :param subtract_output_mean: bool
        :param actFunction: How output is computed; e.g. activation Function
        :type actFunction: Network.getOutput method
        :param number_winner: number of firing neurons in the output in one pattern if actFunction is a WTA function
        :type number_winner: int
        :param e_max: Parameter for getOutput function ' getOutputEMax'. Determines activity threshold.
        :type weight_mean: float
        :param active_in_env: number of cells that are allowed to be active in one environment; if None all cells can fire in all environments
        :type active_in_env: int in [0, self.cells]
        :param n_e: Number of environments; only necessary if active_in_environnment != None
        :type n_e: int
        :param initMethod: How the weights are initialized
        :type initMethod: makeWeights method in :class:`Network`
        :param weight_sparsity: parameter needed of weight initMethod 'Network.makeWeightsSparsity'
        :type weight_sparsity: float in [0,1]
        :param weght_mean: Parameter for weight init function ' makeWeightsNormalDistributed'. Determines mean of the normal distribution how weights are initialized
        :type weight_mean: float
        :param weght_sigma: Parameter for weight init function ' makeWeightsNormalDistributed'. Determines sigma of the normal distribution how weights are initialized
        :type weight_mean: float
        '''

        self.no_presynapses = int(connectivity*input_cells)
        self.learnrate = learnrate
        self.number_winner = number_winner
        self.sparsity = self.number_winner/(cells*1.)
        self.cells = cells
        self.input_cells = input_cells
        self.subtract_input_mean = subtract_input_mean
        self.subtract_output_mean = subtract_output_mean
        self.getOutput = actFunction
        self.n_e = n_e
        self.e_max = e_max
        if active_in_env == None:
            active_in_env = self.cells
        self.active_in_env = active_in_env

        self.initActiveCells()

        initConnectionMethod(self)
        initMethod(self, sparsity=weight_sparsity, mean = weight_mean, sigma = weight_sigma)


        self.Cor = {} #Dictionarry of Corelation instances constisting of input and outputstatistics
        self.output_stored = None #stored output pattern
        self.noisy_output = None #reconstructed stored output when noisy input was given



    ####Initialize the connectivity matrix
    def initConnectionRandom(self, **kwargs):
        self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
        self.connection[np.mgrid[0:self.cells, 0:self.no_presynapses][0], np.array(map(random.sample, [range(self.input_cells)]*self.cells,  [self.no_presynapses]*self.cells))] = 1
    
    def initConnectionCa3Ca1Anisotropic(self, **kwargs):
    
            # Parameters
        N = 2500
        excR = 1.
        NE = int(excR*N)
        
        J = 2.
        
        J_ex = J
        
        layer_dim = 1.5
        dend_radius = 1.6
        neuron_type = 'mat2_psc_exp'
        synapse_model = 'static_synapse'
        
        # Functions
        def creatPositions():
            r = pl.rand(N)*layer_dim
            th= pl.rand(N)*2*pl.pi
            poss = [[r[i]*pl.cos(th[i]),r[i]*pl.sin(th[i])] for i in xrange(N)]
            e_poss = poss[:NE]
        
            ex_layer = topo.CreateLayer({"elements": neuron_type, "extent": [2*layer_dim, 2*layer_dim], "positions": e_poss})
        
            return pl.array(poss), ex_layer
        
        def connectAnisotropic(source_layer,target_layer,ax_thetas,ax_lengths,conn_model):
            tOffset = nest.GetChildren(target_layer)[0][0]
            sposses = pl.array(topo.GetPosition(nest.GetChildren(source_layer,None,False)[0]))
            s = sposses.transpose()[0] + sposses.transpose()[1]*1j
            tposses = pl.array(topo.GetPosition(nest.GetChildren(target_layer,None,False)[0]))
            z = tposses.transpose()[0] + tposses.transpose()[1]*1j
            for i, src in enumerate(nest.GetChildren(source_layer,None,False)[0]):
                zi = s[i]
                ax=ax_thetas[i]
                axl = ax_lengths[i]
                zd = (z-zi)*(pl.cos(ax)-1j*pl.sin(ax))
                tgts = [t for t in nest.GetChildren(target_layer)[0] if \
                    abs(zd[t-tOffset].imag) < dend_radius and \
                    zd[t-tOffset].real <= axl and \
                    zd[t-tOffset].real >= 0 and \
                    t != src]
                nest.DivergentConnect([src],tgts,model=conn_model)
        
        def ID2inx(InputID,ex_layer):
            ExOff = nest.GetChildren(ex_layer)[0][0]
           
            OutputInx = InputID
            #OutputInx[pl.where(InputID>NE+ExOff)] -= -ExOff
            #OutputInx -= ExOff
            OutputInx = OutputInx.astype(int)
            return OutputInx
        
        # program
        nest.ResetKernel()
        
        # create neurons
        poss, ex_layer = creatPositions()
        ax_thetas = pl.rand(NE,1)*2*pl.pi
        ax_lengths = pl.ones((NE,1))
        
        # connect excitatory neurons
        nest.CopyModel("static_synapse", "anisotropic",{"weight": J_ex})
        connectAnisotropic(ex_layer,ex_layer,ax_thetas,ax_lengths,"anisotropic")
        
        
        
        # get the adjacancy list
        conn = nest.GetConnections(source=None, target=None, synapse_model=None)
        conn = pl.array(conn)
        conn = conn[:,:2]
        conn = ID2inx(conn,ex_layer)
       
        
        
        C = pl.zeros([N,N]) # second one is for Ca1
        conn2 = conn[:,1] - 2 - 2500
        conn1 = conn[:,0] - 2
        C[conn1,conn2] = 1

        #pl.savetxt('ca3_ca1_wij.txt', C.T)
        #plt.imshow(C[1225].reshape(50,50))
        
        self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
        #self.connection = np.loadtxt("ca3_ca1_wij.txt")
        self.connection = C.T 
    
    
    def initConnectionCa3Ca1Isotropic(self, **kwargs):
        
        pl.ion()
        nest.ResetKernel()
        
        # generate list of 12 (x,y) pairs
        pos = [[random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)]
               for j in range(2500)]
        
        #l1 = topo.CreateLayer({'extent': [1.5, 1.5],
                               #'positions': pos, 
                               #'elements': 'iaf_neuron'})
        
        ca3 = topo.CreateLayer({'columns': 50, 'rows': 50, 
                                   'extent': [1.5, 1.5],
                                   'elements': 'iaf_neuron', 'edge_wrap': True})
        
        ca1 = topo.CreateLayer({'columns': 70, 'rows': 60, 
                                   'extent': [1.5, 1.5],
                                   'elements': 'iaf_neuron', 'edge_wrap': True})
        
        conndict_rect = { 'connection_type': 'divergent',
                     'mask': {'rectangular': {'lower_left': [-.45,-.45],
                                              'upper_right': [.45,.45]},
                              'anchor': [.0,0.]},'kernel':.3}
                              
                                                    
        conndict_circ = { 'connection_type': 'divergent',
                     'mask': {'circular': {'radius': .65},
                            'anchor': [.0,0.]},'kernel':.2}# .55, 0,0, 0.1
        
        
        
        topo.ConnectLayers(ca3, ca1, conndict_circ)
        
        
        conn = nest.GetConnections(source=None, target=None, synapse_model=None)
        
        conn = np.array(conn)
        
        #Conn_information = topo.DumpLayerConnections(l1,'static_synapse','conn.txt')
        
        C = np.zeros([2500,4200]) # second one is for Ca1
        conn2 = conn[:,1] - 3 - 2500
        conn1 = conn[:,0] - 2
        C[conn1,conn2] = 1
        #np.savetxt('ca3_ca1_wij.txt', C.T)
        
        self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
        #self.connection = np.loadtxt("ca3_ca1_wij.txt")
        self.connection = C.T 
 
    def initConnectionCa2Ca1(self, **kwargs):
        
        pl.ion()
        nest.ResetKernel()
        
        # generate list of 12 (x,y) pairs
        pos = [[random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)]
               for j in range(2500)]
        
        #l1 = topo.CreateLayer({'extent': [1.5, 1.5],
                               #'positions': pos, 
                               #'elements': 'iaf_neuron'})
        
        ca2 = topo.CreateLayer({'columns': 70, 'rows': 60, 
                                   'extent': [1.5, 1.5],
                                   'elements': 'iaf_neuron', 'edge_wrap': True})
        
        ca1 = topo.CreateLayer({'columns': 70, 'rows': 60, 
                                   'extent': [1.5, 1.5],
                                   'elements': 'iaf_neuron', 'edge_wrap': True})
        
        conndict_rect = { 'connection_type': 'divergent',
                     'mask': {'rectangular': {'lower_left': [-.5,-.5],
                                              'upper_right': [.5,.5]},
                              'anchor': [.0,0.]},'kernel':.1}
                              
                                                    
        conndict_circ = { 'connection_type': 'divergent',
                     'mask': {'circular': {'radius': .55},
                            'anchor': [.0,0.1]},'kernel':.3}# .55, 0,0, 0.1
        
        
        
        topo.ConnectLayers(ca2, ca1, conndict_circ)
        
        
        conn = nest.GetConnections(source=None, target=None, synapse_model=None)
        
        conn = np.array(conn)
        
        #Conn_information = topo.DumpLayerConnections(l1,'static_synapse','conn.txt')
        
        C = np.zeros([4200,4200]) # second one is for Ca1
        conn2 = conn[:,1] - 3 - 4200
        conn1 = conn[:,0] - 2
        C[conn1,conn2] = 1
        #np.savetxt('ca3_ca1_wij.txt', C.T)
        
        self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
        #self.connection = np.loadtxt("ca3_ca1_wij.txt")
        self.connection = C.T 
 
    def initConnectionCa3Ca1(self, **kwargs):
        self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
        self.connection = self.connection = np.loadtxt("ca3_ca1_wij.txt")
    
    
    def initConnectionNEST(self):
        self.connection = np.zeros([self.cells, self.input_cells], 'bool')
        self.connection = np.loadtxt("local_connectivity.txt")

    def initConnectionOfflineStructured(self):

        self.connection = np.zeros([self.cells, self.input_cells], 'bool')
        #self.connection = np.loadtxt("OfflineStructredConnections.txt")
        self.connection = load_object(path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/OfflineStructredConnections', info=True, compressed=False)

    def initConnectionplacecellstrained(self):

        self.connection = np.ones([self.cells, self.input_cells], 'bool')      

    '''
    def initConnectionMetric(self, centers, con_loc_param, **kwargs):
        self.connection = np.zeros([self.cells, self.input_cells], 'bool')
        self.connection = np.copy(self.calcDistanceMatrix(centers = centers))
        self.connection[self.connection > con_loc_param] = 0
        #self.connection[self.connection < 0.02] = 0
        self.connection[self.connection != 0.] = 1
    '''

    ###Determines which cells can be active in the environments
    def initActiveCells(self):
        self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')
        self.active_cells_vector = np.zeros([self.n_e, self.cells], 'bool')

        for h in range(self.n_e):
                self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
                self.active_cells_vector[h][self.active_cells[h]]=1

    #Methods for initializing the weights##################################################################
    # important for you
    def makeWeightsZero(self, **kwargs):
		'''
		abstract
		init weights as zero
		'''
		self.weights = np.zeros(self.connection.shape)

    def makeWeightsOne(self, **kwargs):
		'''
		init weights as one and normalize them
		'''
		self.weights = np.ones(self.connection.shape)
		self.weights *= self.connection
		normalize(self.weights)
        
    def makeWeightsOfflineStructured(self, **kwargs):
        '''
        init weights as one and normalize them
        '''  		
        #self.weights = np.loadtxt("OfflineStructredWeights.txt")
        self.weights = load_object(path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/OfflineStructredWeights', info=True, compressed=False)
        normalize(self.weights)
 
    def makeWeightsplacecellstrained(self, **kwargs):
        '''
        init weights as one and normalize them
        '''  		
        #self.weights = np.loadtxt("OfflineStructredWeights.txt")
        self.weights = load_object(path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/placecellstrainedWeights', info=True, compressed=False)   
        normalize(self.weights)     

    def makeWeightsexponential2(self, **kwargs):
		'''
		init weights as exponential kernel and normalize them
		'''
		self.weights = np.zeros(self.connection.shape)
		self.weights = np.loadtxt("weight.txt")
		self.weights *= self.connection
		normalize(self.weights)

    def makeWeightsEc_Equal_Ca3(self, **kwargs):
		'''
		init weights as exponential kernel and normalize them
		'''
		self.weights = np.zeros(self.connection.shape)
		self.weights = np.loadtxt("Ec_Equal_Ca3.txt")
		self.weights *= self.connection
		normalize(self.weights)


    '''
    def makeWeightsexponential(self, centers, **kwargs):
		
		#init weights as exponential kernel and normalize them
		
		self.weights = np.zeros(self.connection.shape)
		self.weights = np.exp(- self.calcDistanceMatrix(centers = centers)*5.)
		self.weights *= self.connection
		normalize(self.weights)
    
    def calcDistanceMatrix(self, centers):

		

		#calculates Distance Matrix given locations. Entry (i,j) is the eucledean distance between location i and j.

		#:param round_it:I f round_it != 0, distances are rounded. if round_it is an number, this number indicates the number of digits it is rounded to. If round it = 'fix', distances are rounded to next bin, specified by bins.
		#:type round_it: int or 'fix'
		#:param: bins: if round_it equals 'fix', distances are rounded up to their next bin in bins. If none, bins = np.linspace(0,1.5, 20)
		#:type bins: np.array
		
		locations = centers.reshape(1, centers.shape[1],2)
		rawDistances = abs(np.tile(locations, (1,locations.shape[1])).reshape(locations.shape[1], locations.shape[1],2) - np.tile(locations, (locations.shape[1],1)).reshape(locations.shape[1], locations.shape[1],2))
		rawDistances[rawDistances>=0.5] = 1. - rawDistances[rawDistances>=0.5]
		distance_matrix = np.sqrt(np.sum((rawDistances)**2, axis = -1)) # (i,j) = eucl distance loc_i loc_j
        
		return distance_matrix
    '''

    def makeWeightsUniformDistributed(self,**kwargs):
        '''
        abstract
        init weighs created over uniform distribution between [0,1] and normalize them;
        '''
        self.weights = np.random.uniform(0,1, self.connection.shape)
        self.weights*=self.connection
        normalize(self.weights)

    def makeWeightsNormalDistributed(self, mean = None, sigma = None, **kwargs):
        '''
        abstract
        init weighs created due to normal distribution with mean = mean and sigma = sigma. Finally, weights are normalized
        :param mean: Mean of distribution
        :type mean: float
        :param sigma: sigma of distribution
        :type sigma: float
        '''
        self.weights = np.random.normal(loc = mean, scale =sigma, size = self.connection.shape)
        self.weights*=self.connection
        normalize(self.weights)


    ##### Learning Methods ##################################################
    # These methods learn patterns by adjusting self.weights
    def hebbianLearning(self,input_pattern = None, output_pattern = None, learnrate = None):
        '''
        abstract
        adjusts weights according to the standard hebbian rule wij = k*p_i*q_j, k = learnfactor = self.learnrate; if self.subtract_input_mean, input_mean is subtracted from input before learning; similar if self.subtract_output_mean
        :param input_pattern: input to associate
        :type input_pattern: array of max 3 dimensions, last one must be self.connection.shape[1]
        :param output_pattern: output to associate
        :type output_pattern: array of max 3 dimensions, last one must be self.connection.shape[0]
        '''
        if learnrate == None:
            learnrate = self.learnrate
        if learnrate != 0:
            input_scaled = np.copy(input_pattern)*1.0
            output_scaled = np.copy(output_pattern)*1.0

            if len(input_pattern.shape) == 1:
                self.weights += learnrate * np.einsum('j,i,ij->ij', input_scaled, output_scaled, self.connection)
            if len(input_pattern.shape) == 2: #(pattern,cell)
                if self.subtract_input_mean:
                    input_mean =np.einsum('pi->i',input_scaled)/(input_scaled.shape[0]+0.0)
                    input_scaled -= input_mean
                if self.subtract_output_mean:
                    output_mean =np.einsum('pi->i',output_scaled)/(output_scaled.shape[0]+0.0)
                    output_scaled -= output_mean
                self.weights += learnrate * np.einsum('pj,pi,ij->ij', input_scaled, output_scaled, self.connection)

            if len(input_pattern.shape) == 3:#(environment, pattern, cell)
                if self.subtract_input_mean:
                    input_mean =np.einsum('epi->i',input_scaled)/(input_pattern.shape[0]*input_pattern.shape[1]+0.0)
                    #input_mean = np.sum(np.sum(input_scaled, 0), 0)/(input_pattern.shape[0]*input_pattern.shape[1]+0.0)
                    input_scaled -= input_mean
                if self.subtract_output_mean:
                    output_mean =np.einsum('epi->i',output_scaled)/(output_pattern.shape[0]*output_pattern.shape[1]+0.0)
                    output_scaled -= output_mean
                self.weights += learnrate * np.einsum('epj,epi,ij->ij', input_scaled, output_scaled, self.connection)


    #####################actFunction:############################################
    #what you need at the beginning
    def calcActivity(self,input_pattern=None):
        '''
        abstract
        calculates the output activity
        :param input_pattern: input_pattern
        :type input_pattern: numpy.array of arbritrary dimension
        :param return: Activity in the output
        :type return: numpy array. of same dimension as input_pattern, only last dimension differ if number of input cells is different to number of output cells.
        '''
        # 
        #print input_pattern.shape
        #print np.mean(input_pattern)
        dummynoise=0.00
        if dummynoise == 0.00:
            dummynoise = 0.0000001
        if input_pattern.shape[-1] == 1100:
            dummynoise = 0.0000001
      
        activity = np.tensordot(input_pattern+ np.random.normal(loc= 0, scale= dummynoise, size=input_pattern.shape), self.weights, (-1,-1))
        #print activity.shape
        return activity
        
        
        
    def KWTA(self, win=None, fire_rate= None, **kwargs): #returns the firing of the network given the input patterns based on k number in kwta method;
        #set K_WTA method
        kwta = 0.2
        size = list(np.shape(win)) #dimension of input
        if len(size) == 1:
            fire_rate[win[0:np.random.random_integers(0,int(self.number_winner *kwta))]] = 0
        if len(size) == 2: #env, pattern--- not possible; use len(size) = 1 instead and specify env.
            print 'len 2 not possible getoutput wta'
        if len(size) == 3:#env, pattern, cell
			for h in range(size[-2]):
				fire_rate[:,h,win[:,h,0:np.random.random_integers(0,int(self.number_winner *kwta))]] = 0
        if len(size) == 4:#env, noise, pattern,cells
			for h in range(size[-3]):
				for k in range(size[-2]):
					fire_rate[:,h,k,win[:,h,k,0:np.random.random_integers(0,int(self.number_winner *kwta))]] = 0 

        return fire_rate

    def getOutputWTA(self,input_pattern=None, env = None, **kwargs): #returns the firing of the network given the input patterns;
        '''
        abstract
        calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire; firing rate is either 1 or 0; Only cells that are allowed to be active in the enviroment are considered.
        :param input_pattern: input
        :type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]. Dimesnions are (environments, noise_level, patterns, cells)
        :param env: specifies current enviromnent if input dimension = 1
        param return: firing of the outputcells
        :param return: array
        '''
        size = list(np.shape(input_pattern)) #dimension of input
        size[-1] = self.weights.shape[0] # change to dimension of output

        #set activity of those cells to zero, that are not allowed to be active in the environment
        if len(size) == 1:
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]
        if len(size) == 2: #env, pattern--- not possible; use len(size) = 1 instead and specify env.
            print 'len 2 not possible getoutput wta'
        if len(size) == 3:#env, pattern, cell
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
        if len(size) == 4:#env, noise, pattern,cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)


        winner = np.argsort(activity)[...,-self.number_winner:size[-1]] 

        fire_rate = np.ones(size, 'bool')
        out_fire = np.zeros(size, 'bool')   
        
        fire_rate = self.KWTA(win= winner, fire_rate= fire_rate) 




        if len(size) ==1:#pattern
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:#env, pattern--- not possible
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3: #env, pattern, cells
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        return out_fire


    def getOutputWTA_Self_Inhibition(self,input_pattern=None, env = None, **kwargs): #returns the firing of the network given the input patterns;
        '''
        abstract
        calculates outputfiring given input_pattern; the highest self.number_winner activated neurons fire; To apply self inhibition, those neurons that are active in "input_patern" are not allowed to be active in "output_pattern". firing rate is either 1 or 0; Only cells that are allowed to be active in the enviroment are considered.

        :param input_pattern: input
        :type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]. Dimesnions are (environments, noise_level, patterns, cells)
        :param env: specifies current enviromnent if input dimension = 1
        param return: firing of the outputcells

        :param return: array
        '''


        size = list(np.shape(input_pattern)) #dimension of input
        size[-1] = self.weights.shape[0] # change to dimension of output

        #set activity of those cells to zero, that are not allowed to be active in the environment
        if len(size) == 1:
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]

        if len(size) == 2: #env, pattern--- not possible; use len(size) = 1 instead and specify env.
            print 'len 2 not possible getoutput wta'
        if len(size) == 3:#env, pattern, cell
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
        if len(size) == 4:#env, noise, pattern,cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)

        if "self_inhibition" in kwargs.keys():
			activity *= kwargs["self_inhibition"]

        winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

        fire_rate = np.ones(size, 'bool')
        out_fire = np.zeros(size, 'bool')
        fire_rate = self.KWTA(win= winner, fire_rate= fire_rate)


        if len(size) ==1:#pattern
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:#env, pattern--- not possible
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3: #env, pattern, cells
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        return out_fire


    def getOutputWTALinear(self,input_pattern=None, env = 0, **kwargs): #returns the firing of the network given the input patterns; output is the activity of the winners
        '''
        abstract
        Same as getOutputWTA, but now outputfiring is equal to activity of cell
        '''

        size = list(np.shape(input_pattern)) #dimension of input
        size[-1] = self.weights.shape[0] # change to dimension of output
        #print 'size getoutputwtalin'
        #print size

        if len(size) == 1:#cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]
        if len(size) == 2:
            print 'len 2 not possible getoutput wta'
        if len(size) == 3:#env, pattern, cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
        if len(size) == 4:#env, noise, pattern,cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)


        winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

        fire_rate = activity # activities < 0 are set to 0
        fire_rate[fire_rate< 0] = 0
        out_fire = np.zeros(size)
        fire_rate = self.KWTA(win= winner, fire_rate= fire_rate)

        if len(size) ==1:
            #print 'winner shape'
            #print winner.shape
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3:
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        normalize(out_fire)
        return out_fire


    def getOutputWTALinear_Self_Inhibition(self,input_pattern=None, env = 0, **kwargs): #returns the firing of the network given the input patterns; output is the activity of the winners
        '''
        abstract
        Same as getOutputWTA, but now outputfiring is equal to activity of cell. To apply self inhibition, those neurons that are active in "input_patern" are not allowed to be active in "output_pattern"
        '''


        size = list(np.shape(input_pattern)) #dimension of input
        size[-1] = self.weights.shape[0] # change to dimension of output
        #print 'size getoutputwtalin'
        #print size

        if len(size) == 1:#cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]
        if len(size) == 2:
            print 'len 2 not possible getoutput wta'
        if len(size) == 3:#env, pattern, cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
        if len(size) == 4:#env, noise, pattern,cells
            activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)

        if "self_inhibition" in kwargs.keys():
			activity *= kwargs["self_inhibition"]

        winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

        fire_rate = activity # activities < 0 are set to 0
        fire_rate[fire_rate< 0] = 0
        out_fire = np.zeros(size)
        fire_rate = self.KWTA(win= winner, fire_rate= fire_rate)

        if len(size) ==1:
            #print 'winner shape'
            #print winner.shape
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3:
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        normalize(out_fire)
        return out_fire





    ######################## Recall Function #########################################
    def recall(self, input_pattern = None, key = '', first = None):

        '''
        abstract
        calculates output given input cues and creates Correlation Classes comparing stored and recalled patterns and recalled with recalled ones.
        :param input_pattern: input
        :type input_pattern: array
        :param first: if first != None, only first 'first' patterns are considered for analysis (if first >0 ). If first <0 only last stored patterns are considered.
        :type first: integer
        '''

        if first == None:
            first = input_pattern.shape[-2]

        if self.output_stored == None: #if nothing is stored
            self.output_stored = np.zeros([1,1,first])

        if first >=0 :
            input_pattern = input_pattern[:,:,:first]
            self.noisy_output = self.getOutput(self,input_pattern)
            self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,:first], patterns_2 = self.noisy_output)
            self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)

        if first < 0:
            input_pattern = input_pattern[:,:,first:]
            self.noisy_output = self.getOutput(self,input_pattern)
            self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,first:], patterns_2 = self.noisy_output)
            self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)

    @classmethod
    def calcFireSparsity(cls, patterns = None):#patterns 0 (loc, pattern)

        '''
        abstract
        Help function to determine sparsity threshold for getOutputLinearthreshold
        '''
        enumerator = (np.sum(patterns*1., axis = -1)/patterns.shape[-1])**2
        denominator = np.sum((patterns*1)**2, -1)/patterns.shape[-1]
        return enumerator/denominator


class HeteroAssociation(Network):

    def learnAssociation(self,input_pattern = None, output_pattern = None, key = 'StoredStored', first = None): #Association of input pattern and output pattern
        '''
        abstract
        hebbian association input with output; self.input_stored becomes input_pattern; self.output_stored becomes outputpattern. Creates Correlation Class self.Cor[key] that analsizes the stored output
        :param input_pattern: input to associate
        :type input_pattern: array of max 4 dimensions, last one must be self.connection.shape[1]
        :param output_pattern: output to associate
        :type output_pattern: array of max 4 dimensions, last one must be self.connection.shape[0]
        :param first: if first != None, only the first 'first' stored pattern are considered for analysis. If negative, only the last ones are considered. However all patterns are stored.
        :type first: integer
        '''
        self.hebbianLearning(input_pattern = input_pattern, output_pattern = output_pattern)
        self.output_stored = output_pattern
        self.input_stored = input_pattern
        if first == None:
            first = input_pattern.shape[-2]
        if first >= 0:
            self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,:first])
        else:
            print 'first none'
            self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,first:])

class AutoAssociation(HeteroAssociation):

    def __init__(self, input_cells=None, cells=None, number_winner=None, connectivity = None, learnrate= None, subtract_input_mean = None, subtract_output_mean = None, initMethod = None, initConnectionMethod = None, weight_sparsity = None, actFunction = None, weight_mean = None, weight_sigma = None, cycles = None, external_force = None, internal_force = None, external_weights = None, active_in_env = None, n_e = None):


        '''
        abstract
        Network with recurrent dynamics. getOutputfunctions as in :class: 'Network' but now activation cycles are possible with external input clamped on.
        :param cycles: Number of activation cycles. In one cycle all neurons are updated synchronously.
        :type cycles: int
        :param external_force: Determines influence of external input during dynamics; if 0 no clapmed external input is considered.
        :type external_force: int
        :param internal_force: Determines influence of recurrent input during dynamics;
        :type internal_force: int
        :param external_weights: Weight matrix that connect external input to the network. Necessary when external_force != 0
        :type external weights: np.array of dimenstion (input_cells, cells)
        '''
        self.cycles = cycles
        self.external_force = external_force
        self.internal_force = internal_force
        self.external_weights = external_weights
        super(AutoAssociation, self).__init__(input_cells=input_cells, cells=cells, number_winner=number_winner, connectivity = connectivity, learnrate= learnrate, subtract_input_mean = subtract_input_mean, subtract_output_mean = subtract_output_mean, initMethod = initMethod, initConnectionMethod = initConnectionMethod, weight_sparsity = weight_sparsity, actFunction = actFunction, active_in_env = active_in_env, n_e = n_e, weight_mean = weight_mean, weight_sigma = weight_sigma)




    ### Helper Function; returns activity that arrives externally
    def calcExternalActivity(self, external_pattern = None):
        activity = np.tensordot(external_pattern, self.external_weights, (-1,-1))
        return activity

    ##### Output Function ################
    #As in Network class, but now implemented with recurrent dynamics
    def getOutputWTARolls(self,input_pattern=None, external_activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
        :param input_pattern: input
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        param return: firing of the outputcells
        :param return:  array possible dimensions (env, noise, pattern, cellfire)
        '''

        def calcOutput(internal_pattern = None, external_activity = None):

            internal_activity = self.calcActivity(input_pattern=internal_pattern)
            normalize(internal_activity)

            activity = internal_activity*self.internal_force  + external_activity*self.external_force

            size = list(np.shape(input_pattern)) #dimension of input
            size[-1] = self.weights.shape[0] # change to dimension of output
            winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

            max_activity_pattern = np.max(activity, -1)
            fire_rate = max_activity_pattern.repeat(activity.shape[-1]).reshape(activity.shape)
            #fire_rate = np.ones(size, 'bool')
            out_fire = np.zeros(size)


            if len(size) ==1:
                out_fire[winner] = fire_rate[winner]
            if len(size) ==2:
                out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
            if len(size) ==3:
                indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
                out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
            if len(size) ==4: # env, noise, time, pattern
                indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
                out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
            if len(size) > 4:
                    print 'error in input dimension in calckWTAOutput'


            return out_fire

        if external_activity != None:
            normalize(external_activity)

            out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
                if c == self.cycles -1:
                    print 'all ' +str(c) +' cycles used'
            return out_fire

        else: #rec dynamics without consistent input form the outside
            out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
            return out_fire

    def getOutputWTA(self,input_pattern=None, external_activity = None, env = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
        :param input_pattern: input
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        param return: firing of the outputcells
        :param return:  array possible dimensions (env, noise, pattern, cellfire)
        '''

        def calcOutput(internal_pattern = None, external_activity = None, env = None):

            internal_activity = self.calcActivity(input_pattern=internal_pattern)
            normalize(internal_activity)

            size = list(np.shape(input_pattern)) #dimension of input
            size[-1] = self.weights.shape[0] # change to dimension of output
            if len(size) == 1:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector[env]
            if len(size) == 2:
                print 'len 2 not possible getoutput wta'
            if len(size) == 3:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
            else:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)


            winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

            fire_rate = np.ones(size, 'bool')
            out_fire = np.zeros(size, 'bool')

            if len(size) ==1:
                out_fire[winner] = fire_rate[winner]
            if len(size) ==2:
                out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
            if len(size) ==3:
                indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
                out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
            if len(size) ==4: # env, noise, time, pattern
                indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
                out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
            if len(size) > 4:
                    print 'error in input dimension in calckWTAOutput'
            return out_fire


        if external_activity != None:
            normalize(external_activity)
            out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity, env = env)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity, env = env)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
                if c == self.cycles -1:
                    print 'all ' +str(c) +' cycles used'

        else: #rec dynamics without consistent input form the outside
            out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, env = env)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, env = env)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
        normalize(out_fire)
        return out_fire

    def getOutputWTALinear(self,input_pattern=None, external_activity = None, env = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
        :param input_pattern: input
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        param return: firing of the outputcells
        :param return: array possible dimensions (env, noise, pattern, cellfire)
        '''

        def calcOutput(internal_pattern = None, external_activity = None, env = None):

            internal_activity = self.calcActivity(input_pattern=internal_pattern)
            normalize(internal_activity)

            #activity = (internal_activity*self.internal_force  + external_activity*self.external_force)* self.active_cells_vector

            size = list(np.shape(input_pattern)) #dimension of input
            size[-1] = self.weights.shape[0] # change to dimension of output
            if len(size) == 1:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector[env]
            if len(size) == 2:
                print 'len 2 not possible getoutput wta'
            if len(size) == 3:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
            else:
                activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)


            winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

            activity[activity<0] = 0
            fire_rate = activity
            out_fire = np.zeros(size, 'bool')


            if len(size) ==1:
                out_fire[winner] = fire_rate[winner]
            if len(size) ==2:
                out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
            if len(size) ==3:
                indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
                out_fire[indices[0], indices[1], winner] = fire_rate[indices[0], indices[1], winner]
            if len(size) ==4: # env, noise, time, pattern
                indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
                out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
            if len(size) > 4:
                    print 'error in input dimension in calckWTAOutput'
            return out_fire



        if external_activity != None:
            normalize(external_activity)

            out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity, env = env)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity, env = env)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
                if c == self.cycles -1:
                    print 'all ' +str(c) +' cycles used'
            return out_fire

        else: #rec dynamics without consistent input form the outside
            out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete, env = env)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete, env = env)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
            return out_fire

    def getOutputLinearthreshold(self,input_pattern=None, external_activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
        :param input_pattern: input
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        param return: firing of the outputcells
        :param return: array possible dimensions (env, noise, pattern, cellfire)
        '''

        def calcOutput(internal_pattern = None):

            internal_activity = self.calcActivity(input_pattern=internal_pattern)
            normalize(internal_activity)

            activity = self.internal_force * internal_activity + self.external_force * external_activity

            if len(internal_pattern.shape) >= 2:
                a = Network.calcFireSparsity(activity)
                not_sparse = True
                i=0

                while not_sparse:

                    too_large_a_rows = a > self.sparsity
                    if not too_large_a_rows.any():
                        not_sparse = False

                        #print np.max(a,-1)
                        break

                    #print 'act'
                    #print activity
                    activity[activity == 0] = 10**5
                    min_activity_cell = np.argmin(activity, -1)
                    activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
                    activity[activity == 10**5] = 0
                    #print 'act after'
                    #print activity
                    a = Network.calcFireSparsity(activity)
                    i+=1
                    #time.sleep(5)


                i=0

                normalize(activity)

            if len(internal_pattern.shape) == 1:
                a = Network.calcFireSparsity(activity)
                not_sparse = True
                i=0

                while a > self.sparsity:
                    min_activity = np.min(activity[activity !=0])
                    activity[activity == min_activity] = 0
                    a = Network.calcFireSparsity(activity)
                    i+=1
                #print 'number cells reduced to zero'
                #print i
                normalize(activity)

            return activity



        if external_activity != None:
            normalize(external_activity)

            out_fire = calcOutput(internal_pattern = input_pattern)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = calcOutput(internal_pattern =out_fire)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
                if c == self.cycles -1:
                    print 'all ' +str(c) +' cycles used'
            return out_fire

        else: #rec dynamics without consistent input form the outside
            out_fire = super(AutoAssociation, self).getOutputLinearthreshold(input_pattern=input_pattern, discrete = discrete)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = super(AutoAssociation, self).getOutputLinearthreshold(input_pattern=out_fire, discrete = discrete)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
            return out_fire

    def getOutputId(self,  external_activity = None, input_pattern=None):


        def calcOutput(internal_pattern = None, external_activity = None):

            internal_activity = self.calcActivity(input_pattern=internal_pattern)
            normalize(internal_activity)

            activity = internal_activity*self.internal_force  + external_activity*self.external_force

            return activity



        if external_activity != None:
            normalize(external_activity)

            out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
                if c == self.cycles -1:
                    print 'all ' +str(c) +' cycles used'
            return out_fire

        else: #rec dynamics without consistent input form the outside
            out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete)
            for c in range(self.cycles):
                out_old = np.copy(out_fire)
                out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete)
                if (out_old == out_fire).all():
                    print 'stop after ' + str(c) +' cycles'
                    break
            return out_fire


        return activity


    #########Recall Function###########
    #As in Network; Now with additional external activity possible
    def recall(self, input_pattern = None, external_activity = None, key = '', first = None):
        if first == None:
            first = input_pattern.shape[-2]
        if first >= 0:
            input_pattern = input_pattern[:,:,:first]
            self.noisy_output = self.getOutput(self,input_pattern = input_pattern, external_activity = external_activity[:,:,:first])
            self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,:first], patterns_2 = self.noisy_output)
            self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)
        else:
            input_pattern = input_pattern[:,:,first:]
            self.noisy_output = self.getOutput(self,input_pattern = input_pattern, external_activity = external_activity[:,:,first:])
            self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,first:], patterns_2 = self.noisy_output)
            self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)


class OneShoot(Network):

	'''

	Network where input first triggers output with existing weights and then weights are adjusted. Different to Association networks, where a pair of patterns is associated.
	'''


	def __init__(self, **kwargs):


		super(OneShoot, self).__init__(**kwargs)
		self.co_factor = 1.0/self.cells # needed for neural gas; it decrases learning amount through time
	#one shoot learning methods

	def learnFiringDependent(self,input_pattern = None, output_pattern = None, cofactor_increase = None): # one shoot learning of patterns; hebbian association with input and outputfiring; outputpattern is opiional
		if output_pattern ==None:
			output_pattern = self.getOutput(self,input_pattern = input_pattern)
		self.weights += self.learnrate * np.einsum('j,i,ij->ij', input_pattern*1, output_pattern*1, self.connection)
		normalize(self.weights)

	def learnActivityDependent(self,input_pattern =None, output_pattern = None, cofactor_increase = 0): # one shoot version, here learning rate is dependent on output activity not firing; output_pattern is not needed
		activity = self.calcActivity(input_pattern = input_pattern)
		activity[np.argsort(activity)[:-self.number_winner]]=0
		self.hebbianLearning(input_pattern = input_pattern, output_pattern = activity)
		normalize(self.weights)

	def learnSiTreves(self,input_pattern =None, output_pattern = None, cofactor_increase = None):

		#input_mean = np.sum(input_pattern)*1./input_pattern.shape[0]


		input_mean = (np.dot(self.connection, input_pattern)*1./np.sum(self.connection, axis = -1)).repeat(input_pattern.shape[0]).reshape(self.connection.shape[0], input_pattern.shape[0])
		input_p = np.tile(input_pattern, (self.connection.shape[0],1)) -input_mean
		self.weights += self.learnrate * np.einsum('ij,i,ij->ij', input_p, output_pattern*1, self.connection)
		self.weights[self.weights< 0] = 0
		normalize(self.weights)


	def learnOneShootAllPattern(self,input_pattern = None, method = None, key = 'StoredStored', first = None, store_output = 1): # one shoot learning of all patterns using giving method; output activity during learning is stored

		print 'one shoot learning all patterns'
		normalize(self.weights)

		if store_output:
			output = np.zeros([input_pattern.shape[0],input_pattern.shape[1], self.cells])
			if self.learnrate != 0:
				for env in range(input_pattern.shape[0]):
					for p in range(input_pattern.shape[1]):
						output[env,p] = self.getOutput(self,input_pattern = input_pattern[env, p], env = env)
						method(self,input_pattern = input_pattern[env,p], output_pattern = output[env,p])
			else:
				output = self.getOutput(self,input_pattern = input_pattern)
			self.output_stored = output
			self.input_stored = input_pattern
			if first == None:
				first = input_pattern.shape[-2]
			if first >= 0:
				self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,:first])
			else:
				self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,first:])

		else:
			for env in range(input_pattern.shape[0]):
				for p in range(input_pattern.shape[1]):
					method(self,input_pattern = input_pattern[env,p], output_pattern = self.getOutput(self,input_pattern = input_pattern[env, p], env = env))

	def learnAccordingToRank(self, input_pattern = None, output_pattern = None, cofactor_increase = 0): # outputpattern not needed

		activity = self.calcActivity(input_pattern = input_pattern)
		activity_sort = np.argsort(activity)
		learnrate = calcExp(-self.co_factor*(np.arange(self.cells)))[activity_sort]
		self.weights += np.outer(learnrate, input_pattern) *self.connection
		normalize(self.weights)
		self.co_factor += cofactor_increase


class OneShootmehdi(Network):

	'''

	Network where input first triggers output with existing weights and then weights are adjusted. Different to Association networks, where a pair of patterns is associated.
	'''


	def __init__(self, **kwargs):


		super(OneShootmehdi, self).__init__(**kwargs)


	def learnFiringDependent(self,input_pattern = None, output_pattern = None, learnrate = None): # one shoot learning of patterns; hebbian association with input and outputfiring; outputpattern is opiional
		self.weights += learnrate * np.einsum('j,i,ij->ij', input_pattern*1, output_pattern*1, self.connection)
		#normalize(self.weights)

	def learnOneShootAllPattern(self,input_pattern = None, method = None, num_seq= None, seq_len= None): # one shoot learning of all patterns using giving method; output activity during learning is stored
		print 'one shoot learning all patterns'
		#normalize(self.weights)

		for env in range(input_pattern.shape[0]):
			for num in range(num_seq):
			    for seq in range(seq_len-1):
					p = seq + (num * seq_len)
					#learn_rate = .05
					learn_rate = .083333 - seq/(seq_len*10.)
					if seq>5: learn_rate = 0.04
					method(self,input_pattern = input_pattern[env,p], output_pattern = input_pattern[env,p+1], learnrate = learn_rate)
		normalize(self.weights)            


##################Paramter####################################################################

class Parameter():

    '''
    abstract
    Parameter Class. If some paramters are not given in the Simulation, usually paramters of this class are used instead of raising an error.
    '''

    #Parameter
    no_pattern = 100
    number_to_store = no_pattern
    n_e=1
    first  = None
    cells = dict(Ec = 1100, Dg = 12000, Ca3 =2500, Ca1 = 4200)#cell numbers of each region
    sparsity = dict(Ec = 0.35, Dg = 0.005, Ca3 = 0.032, Ca1 = 0.09)#activity level of each region (if WTA network)
    number_winner = dict(Ec = int(cells['Ec']*sparsity['Ec']), Dg = int(cells['Dg']*sparsity['Dg']), Ca3 = int(cells['Ca3']*sparsity['Ca3']), Ca1 = int(cells['Ca1']*sparsity['Ca1']) )
    connectivity = dict(Ec_Dg = 0.32, Dg_Ca3 = 0.0006, Ca3_Ec = 0.32, Ec_Ca3 =0.32, Ca3_Ca3 = 0.24, Ca3_Ca1 = 0.32, Ca1_Ec = 0.32, Ec_Ca1 = 0.32, Ca1_Sub = 0.32, Sub_Ec = 0.32, Ec_Sub = 0.32)# probability given cell is connected to given input cell
    learnrate = dict(Ec_Dg = 0.5, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 0.5, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
    initMethod = dict(Ec_Dg = Network.makeWeightsUniformDistributed, Dg_Ca3 = Network.makeWeightsUniformDistributed, Ec_Ca3 =Network.makeWeightsZero, Ca3_Ec =Network.makeWeightsZero, Ca3_Ca3 = Network.makeWeightsZero, Ec_Ca1 = Network.makeWeightsNormalDistributed, Ca3_Ca1 = Network.makeWeightsZero, Ca1_Ec =Network.makeWeightsZero)

    initConnectionMethod = dict(Ec_Dg = Network.initConnectionRandom, Dg_Ca3 = Network.initConnectionRandom, Ec_Ca3 =Network.initConnectionRandom, Ca3_Ec =Network.initConnectionRandom, Ca3_Ca3 = Network.initConnectionRandom, Ec_Ca1 = Network.initConnectionRandom, Ca3_Ca1 = Network.initConnectionRandom, Ca1_Ec =Network.initConnectionRandom)

    actFunctionsRegions = dict(Ec_Dg = Network.getOutputWTALinear, Dg_Ca3 = Network.getOutputWTA, Ca3_Ec = Network.getOutputWTALinear, Ec_Ca3 = Network.getOutputWTA, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputWTALinear, Ca1_Ec = Network.getOutputWTALinear, Ec_Ca1 = Network.getOutputWTALinear)
    active_in_env = dict(Ec = int(cells['Ec']), Dg =  int( cells['Dg']), Ca3 =int(cells['Ca3']), Ca1 = int(cells['Ca1']))

    radius_ca3_field = 0.2111
    radius_ca1_field = 0.2523

    active_env = 0
    if active_env: #only proportion of cells active in each env
        active_in_env = dict(Ec = cells['Ec'], Dg =  int(0.035 * cells['Dg']), Ca3 =int(0.2325 * cells['Ca3']), Ca1 = int(0.4625 * cells['Ca1'])) #number cells active in each
        sparsity = dict(Ec = 0.35, Dg = 0.0064/active_in_env['Dg']*cells['Dg'], Ca3 = 0.03255/active_in_env['Ca3']*cells['Ca3'], Ca1 = 0.0925/active_in_env['Ca1']*cells['Ca1'])#activity level given cell is active in env

    incrementalLearnMethod = OneShoot.learnFiringDependent#learns either activtiy dpendent, or firing dependent
    incremental_storage_mode = None # learns pattern either 'online', i.e. during learning statisitcs or 'offline', i.e. after learning input statistics
    no_incremental_times = None

    #rec dynamics
    external_force = 1
    internal_force = 3
    cycles = 15

    noise_levels = np.arange(0,cells['Ec']+1, int(cells['Ec']/8))
    noise_levels_ca3 =[0]
    noise_levels_ca1 =[0]

    subtract_input_mean = 1
    subtract_output_mean = 0 #=1 for all Ca3_Ca3 Autoassociations

    def __init__():
        pass

################################################ Analysis Classes #######################################################
class Corelations(object):

    def __init__(self, patterns_1=None, patterns_2=None, in_columns = False):
        '''
        abstract
        computes pearson correlations <a,b>/|a||b|; where a is element of patterns_1 and b of patterns_2: Copmutes the correlation pairwise for patterns in each noise level. Different Environments are lumped together and treated as one.
        :param patterns_1: original stored pattern
        :type patterns_1: array of dimension 2 (pattern, cellfire) or 3 (env, pattern, cellfire)
        :param patterns_2:  noisy_original pattern; if not given autocorrelation with patterns_1 is computed
        :type patterns_2: array of dimension 3 (noise, pattern, cellfire) or 4 (env, noise, pattern, cellfire)
        :param in_columns: If True, transpoes data given as (fire, pattern) or (noise, fire, pattern) into right dimension order.
        :type in_colums: Bool

         '''



        self.orig_vs_orig = None #
        self.orig_vs_other = None #average corelation of original pattern and noisy version of the original; array has length len(noise_levels)
        self.over_orig_vs_orig = None#overlaps
        self.over_orig_vs_other = None
        if len(patterns_1.shape) == 3:
            self.patterns_1= patterns_1.reshape(patterns_1.shape[0]*patterns_1.shape[1], patterns_1.shape[-1]) # env, pat, cell into env*pat,cell
        else:
            self.patterns_1= patterns_1
        if patterns_2 == None:
            self.patterns_2 = np.tile(self.patterns_1, (2,1,1))
            self.one_patterns = True
        else:
            self.one_patterns = False
            if len(patterns_2.shape) == 4:
                self.patterns_2= np.swapaxes(patterns_2, 0,1).reshape(patterns_2.shape[1], patterns_2.shape[0]*patterns_2.shape[2], patterns_2.shape[-1])
            if len(patterns_2.shape) == 3:#env, pattern, cell
                self.patterns_2 = np.tile(patterns_2.reshape(patterns_2.shape[0]*patterns_2.shape[1], patterns_2.shape[-1]), (2,1,1)) #2, env*patt, cell

        if in_columns: #if data is given as columns, transpose them into rows !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! not tested with env framework
            self.patterns_1 = np.transpose(self.patterns_1)
            self.patterns_2 = np.transpose(self.patterns_2, (0,2,1))


        self.distance_matrix = None # matrix that has at (i,j) the euclidean distance between pattern i and j as entry
        self.corelations = None # Corelation Matrix; (abc) is corelation of pattern b and c with noise a
        self.calculated_cor = False #indicates if self.corelation is yet computed, this way double computation is avoided
        self.overlaps = None# Matrix of overlaps;(abc) is overlap of pattern b and c with noise a
        self.calculated_over = False #indicates if self.overlaps is yet computed, this way double computation is avoided

        self.covariance = None
        self.covariance_ev = None
        self.covariance_ew = None
        self.number_relevant = None
        self.projection_matrix = None


        self.distances = None
        self.all_distances = None
        self.p_d =None
        self.p_fire_given_distance = None

        self.fire_fire_distances = []
        self.fire_distances = []
        self.silent_distances = []
        self.fire_silent_distances = []
        self.p_d_fire= []
        self.p_d_silent = []
        self.p_fire_given_fire_and_distance = []
        self.p_fire_given_silent_and_distance = []
        self.p_fire_given_fire_and_distance_weighted_distance = []
        self.p_fire_given_silent_and_distance_weighted_distance = []
        for i in range(self.patterns_1.shape[-1]):
            self.fire_fire_distances +=[None]
            self.fire_distances +=[None]
            self.silent_distances +=[None]
            self.fire_silent_distances +=[None]
            self.p_d_fire +=[None]
            self.p_d_silent +=[None]
            self.p_fire_given_fire_and_distance +=[None]
            self.p_fire_given_silent_and_distance +=[None]
            self.p_fire_given_fire_and_distance_weighted_distance +=[None]
            self.p_fire_given_silent_and_distance_weighted_distance +=[None]

    ######### calc Methods ####################
    def calcCor(self, noise_level = None, in_steps = False):
        '''
        abstract
        computes the pearson corelation matrix self.corelations; self.corelation[abc] = corleation of pattern b and  pattern c at noise_level a =  <b - <b>, c - <c>)/||b||*||c|
        :param noise_level: noise_level at which the matrix is computed; if None then it is computed for all levels
        :type noise_level: int in the intervall (0, len(noise_levels))
        :param in_steps: If True and noise_level != None, corrleations are calculated individually for each pair. This is necessary only for huge data sets, when memory demands are too high.
        :type in_steps: Bool
        '''
        if noise_level == None:
            if not self.calculated_cor:
                mean_subtracted_1 = self.patterns_1 - (np.einsum('pi->p', self.patterns_1*1)/(self.patterns_1.shape[-1]+0.0)).repeat(self.patterns_1.shape[-1]).reshape(self.patterns_1.shape)
                mean_subtracted_2 = self.patterns_2 - np.einsum('npi->np', self.patterns_2*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2.shape)/(self.patterns_2.shape[-1]+0.0)
                self.p1_norm_inverse = 1./np.sqrt(np.einsum('...ai, ...ai ->...a', mean_subtracted_1, mean_subtracted_1)) #input_norm_inverse[b]= 1/norm(b)
                self.p2_norm_inverse = 1./np.sqrt(np.einsum('...bi, ...bi ->...b', mean_subtracted_2, mean_subtracted_2))#p2_norm_inverse[a,b]= 1/norm(b(a)) ; b at noise level a
                self.corelations = np.einsum('pi, nqi, p, nq-> npq', mean_subtracted_1, mean_subtracted_2, self.p1_norm_inverse, self.p2_norm_inverse )#cor[abc] = at noise a over of patt b (noise =0) and c (noise = a)= <b - <b>, c - <c>)/||b||*||c||
                self.calculated_cor = True
        else:
            if self.corelations == None:
                self.corelations = np.zeros([self.patterns_2.shape[-3],self.patterns_2.shape[-2], self.patterns_2.shape[-2]])
                self.p2_norm_inverse = np.zeros([self.patterns_2.shape[-3], self.patterns_2.shape[-2]])
            if (self.corelations[noise_level] == 0).all():
                mean_subtracted_1 = self.patterns_1 - (np.einsum('pi->p', self.patterns_1*1)/(self.patterns_1.shape[-1]+0.0)).repeat(self.patterns_1.shape[-1]).reshape(self.patterns_1.shape)
                mean_subtracted_2 = self.patterns_2[noise_level] - np.einsum('pi->p', self.patterns_2[noise_level]*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2[0].shape)/(self.patterns_2.shape[-1]+0.0)
                self.p1_norm_inverse = 1./np.sqrt(np.einsum('...ai, ...ai ->...a', mean_subtracted_1, mean_subtracted_1)) #input_norm_inverse[b]= 1/norm(b)
                self.p2_norm_inverse[noise_level] = 1./np.sqrt(np.einsum('bi, bi ->b', mean_subtracted_2, mean_subtracted_2))#p2_norm_inverse[a,b]= 1/norm(b(a)) ; b at noise level a
                if in_steps:
                    for i in range(mean_subtracted_1.shape[0]):
                        for j in range(mean_subtracted_2.shape[0]):
                            self.corelations[noise_level][i][j] = np.dot(mean_subtracted_1[i],mean_subtracted_2[j])* self.p1_norm_inverse[i]* self.p2_norm_inverse[noise_level][j]

                else:
                    self.corelations[noise_level] = np.einsum('pi, qi, p, q-> pq', mean_subtracted_1, mean_subtracted_2, self.p1_norm_inverse, self.p2_norm_inverse[noise_level] )# cor[abc] = at noise a over of patt b (noise =0) and c (noise = a)= <b(0),






    #########################gets with Corelations######################
    def getCor(self, at_noise=None): #get correlations at_noise
        '''
        abstract
        get correlations at noise level at_noise
        '''
        self.calcCor(noise_level = at_noise)
        return self.corelations[at_noise]

    def getCorOrigOrig(self, at_noise=None):
        '''
        abstract
        get diagonal of self.corelation; These are the correlations of original patterns with reconstructed version; if at_noise = None, gets all noise_levels and gets array of dimension two (noise, diagonal)
        '''
        self.calcCor()
        if at_noise == None:
            corelations = np.copy(self.corelations)
        else:
            corelations = np.copy(self.corelations[at_noise])
        return np.diagonal(corelations, 0, -1,-2)

    def getCorOrigOther(self, at_noise=None, subtract_orig_orig = 1, pattern = None, column = False): #

		'''

		get correlations entries of self.corealtions at_noise_level; if at_noise = None, gets all noise_levels and gets array of dimension two (noise, entries). If self.one_pattern, only entries (i,j) with i<=j are returned, since the entry (i,j) is equal to the entry (j,i) here.

		:param pattern: if pattern != None, it gets only the correation of this pattern with all others.
		:type pattern: integer indicating index of pattern
		:param column: if pattern != None, it determines whether self.correlations[at_noise, pattern] (row) or self.correlations[at_noise, :, pattern] (column) is returned
		:type column: bool
		:param subtract_orig_orig: If True, only correlation(p_j, p_i) with i!= j are returned; i.e. not the entries at the diagonal
		:type subtract_orig_orig: Bool
		 orig_vs_other as array; if noise = None, we have additional noise dimension in return, if one_patterns this is triangle entries of self.corelations, else it is all entries; subtract_orig_orig determines, if diagonal is included (orig_orig)
		'''
		self.calcCor(noise_level = at_noise)
		if pattern == None: # all patterns
			if at_noise == None: #
				corelations = np.copy(self.corelations)
				if self.one_patterns:#triangle is sufficent since cor matrix is symetric, here we use lower triangle
					if subtract_orig_orig:#without diag
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						return_value = corelations[ind0, tril0, tril1]
					else: #with diagonal
						tril0 = np.tril_indices(corelations.shape[-1], 0)[0]
						tril1 = np.tril_indices(corelations.shape[-1], 0)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])

						return_value = corelations[ind0, tril0, tril1]
				else: #lower and upper differ and must be used both
					if subtract_orig_orig:
						#lower triangel
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_lower = corelations[ind0, tril0, tril1]

						#upper triangel
						tril0 = np.triu_indices(corelations.shape[-1], 1)[0]
						tril1 = np.triu_indices(corelations.shape[-1], 1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_upper = corelations[ind0, tril0, tril1]

						return_value = np.zeros([corelations.shape[0], corelations.shape[-1]**2 - corelations.shape[-1]])
						for i in range(corelations.shape[0]):
							return_value[i,:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower[i]
							return_value[i,(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper[i]
					else:
						for i in range(corelations.shape[0]):
							return_value[i] = np.ravel(corelations[i])
			else:
				corelations = np.copy(self.corelations[at_noise])
				if self.one_patterns:
					if subtract_orig_orig:
						return_value = corelations[np.tril_indices(corelations.shape[-1], -1)]
					else:
						return_value = corelations[np.tril_indices(corelations.shape[-1], 0)]
				else:
					if subtract_orig_orig:
						cor_lower = corelations[np.tril_indices(corelations.shape[-1], -1)]
						cor_upper = corelations[np.triu_indices(corelations.shape[-1], 1)]
						return_value = np.zeros(corelations.shape[-1]**2 -corelations.shape[-1])
						return_value[:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower
						return_value[(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper
					else:
						return_value = np.ravel(corelations)
		else: #one pattern
			if at_noise != None:
				if column:
					return_value = self.corelations[at_noise][:,pattern]# column pattern, i.e. StoredRecalled, REcalled pattern used
				else:
					return_value = self.corelations[at_noise][pattern]# Stored pattern used
			else:
				return_value = np.zeros([self.corelations.shape[0],self.corelations.shape[1]])
				for i in range(self.corelations.shape[0] ):
					if column:
						return_value[i] = np.ravel(self.corelations[i,:,pattern])# column pattern, i.e. StoredRecalled, REcalled pattern used
					else:
						return_value[i] = np.ravel(self.corelations[i,pattern])# Stored pattern used
		return return_value



    def getOrigVsOrig(self):

        '''
        abstract
        returns the average Corelation of the diagonal in self.correlations. If at_noise = None, at all noise_levels. Then return has dimension (noise_levels)
        '''
        if self.orig_vs_orig == None:
            self.orig_vs_orig = np.sum(self.getCorOrigOrig(), axis = -1)/(self.patterns_1.shape[0]+0.0)
        return self.orig_vs_orig

################################################Input Classes ######################################################
class Input(Corelations):

    def __init__(self, number_cells = Parameter.cells['Ec'], n_e = Parameter.n_e,  number_patterns= Parameter.no_pattern, store_indizes = None, number_to_store = None, inputMethod = None, noiseMethod = None, actFunction = None, sparsity = Parameter.sparsity['Ec'], noise_levels = Parameter.noise_levels, normed = 0, In = None, patterns = None):
        '''
        abstract
        Class that creates pattern and builts noisy versions of them.
        :param number_cells: number of cells
        :type number_cells: int
        :param n_e: Number of environments
        :type n_e: int
        :param number_patterns: number of total pattern this instance creates
        :type number_patterns: int
        :param store_indizes: indices of patterns that are considered for storing
        :type store_indizes: array of dimension 2 (environment, indices)
        :param number_to_store: number of patterns that are considered for storing
        :type number_to_store: int
        :param inputMethod: How patterns are created
        :type inputMethod: Input.makeInput method
        :param noiseMethod: How noise is created
        :type noiseMethod: Input.makeNoise method
        :param actFunction: How activation is transformed into firing
        :type actFunction: Input.getOutput method
        :param sparsity: Proportion of cells beeing active in each pattern; for getOutputWTA necessary
        :type sparsity: float
        :param noise_levels: levels of noise beeing applied on the pattern; mostly number of cells firing wrongly
        :type noise_levels: list or array of noise_levels
        :param normed: Whether patterns are normalized
        :type normed: bbol
        :param In: When given it uses its paramters
        :type In: Instance of Input
        :param patterns: When given, it uses these patterns as self.patterns
        :type patterns: array of dimension 3 (environment, pattern, cell fire)
        '''

        self.n_e = n_e
        self.number_patterns = number_patterns #per env
        self.number_to_store = number_to_store #per env
        self.inputMethod = inputMethod
        self.noiseMethod = noiseMethod
        self.actFunction = actFunction
        self.store_indizes = store_indizes
        self.cells = number_cells
        self.noise_levels = np.array(noise_levels)
        self.sparsity = sparsity
        self.number_winner = int(sparsity*self.cells)
        if number_to_store ==  None:
            self.number_to_store = self.number_patterns
        if In != None: #uses Parameter of given Input Instance
            self.n_e = In.n_e
            self.number_patterns = In.number_patterns #per env
            self.number_to_store = In.number_to_store #per env
            self.inputMethod = In.inputMethod
            self.noiseMethod = In.noiseMethod
            self.actFunction = In.actFunction
        if patterns == None: # uses given patterns as self.patterns
            self.patterns = np.zeros([self.n_e,self.number_patterns, self.cells])
            self.patterns_given = False
        else:
            self.patterns = patterns
            self.patterns_given = True

        self.input_stored = np.zeros([self.n_e, self.number_to_store, self.cells])#patterns that are considered for storing
        self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])#noisy versions of self.input_stored

        #creates self.patterns
        self.makeInput()
        if normed:
            normalize(self.patterns)
        #create self.input_stored and self.noisy_imput_stored
        self.choosePatternToStore(store_indizes = store_indizes)

        #Input Instance is a Corelation Instance at the same time.
        super(Input, self).__init__(patterns_1 = self.input_stored, patterns_2= self.noisy_input_stored)

    def makeInput(self):
        '''
        abstract
        creates the input; all input patterns are self.patterns. It has dimensions (envionment, pattern, cell firing)
        '''
        if not self.patterns_given:
            print 'make patterns'
            if self.inputMethod == None:
                print 'noinput Method gven'
                self.makeInputSparsityLevel()
            else:
                self.inputMethod(self)

        else:
            pass

    def makeNoise(self, env = None):
        '''
        abstract
        creates the noisy version of the input pattern self.patterns.
        :param env: Environment for which noise is created
        '''
        if self.noiseMethod == None:
            noise =self.makeNoiseRandomFire(pattern = self.patterns[env], noise_levels= self.noise_levels)
        else:
            noise = self.noiseMethod(self, pattern = self.patterns[env], noise_levels = self.noise_levels)
        return noise

    def choosePatternToStore(self, number_to_store = None, store_indizes = None):
        '''
        abstract
        sets the input patterns in self.patterns that are going to be stored. These patterns are then self.input_stored and their noisy version are in self.noisy_input_stored.
        :param number_to_sore: How many patterns are stored; if None, self.number_to_store is used
        :type number_to_store: int
        :param store_indizes: Indices of patterns that are stored. If None, store_indizes are created by self.makeStoreIndizes(store_indizes)
        :type store_indizes: array of dimension two; (environment, indizes)
        '''
        if number_to_store != None:
            self.number_to_store = number_to_store
        self.makeStoreIndizes(store_indizes)
        self.input_stored = np.zeros([self.n_e, self.number_to_store, self.cells])
        self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])
        for h in range(self.n_e):
            self.input_stored[h] = self.patterns[h][self.store_indizes[h]] #the pattern that are actually given as inputs; note that self.input_stored[i] corresponds to self.location[self.store_indized][i]
            self.noisy_input_stored[h] = self.makeNoise(h)[:,self.store_indizes[h]]

    def makeStoreIndizes(self, store_indizes=None):
        '''
        abstract
        creates store indizes randomly if store_indizes = None. If store_inidizes != None, these indizes are used.
        :param store_indizes: Indices of patterns that are stored. If None, store_indizes are created by self.makeStoreIndizes(store_indizes)
        :type store_indizes: array of dimension two; (environment, indizes)
        '''
        if self.number_to_store > self.number_patterns:
            print 'not enpugh input patterns'
        if store_indizes == None:
            print 'make store indizes'
            self.store_indizes = np.array(map(random.sample, [range(self.patterns.shape[1])]*self.n_e, [self.number_to_store]*self.n_e))
        else:
            self.store_indizes = store_indizes
            print 'store indizes given'

    def makeNewNoise(self, method = None):
        '''
        abstract
        creates noisy patterns, when Input Instance was already created. Old noise is overwritten
        :param method: Which method is used for creation of noise
        :type method: Input.makeNoise method.
        '''
        self.noiseMethod = method
        self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])
        for h in range(self.n_e):
            self.noisy_input_stored[h] = self.makeNoise(h)[:,self.store_indizes[h]]


    #################inputMethods#########################
    ##creates the input patterns self.patterns
    def makeInputNormalDistributed(self):
        '''
        abstract
        creates patterns, each cell activity is a sample of a normal distribution
        '''
        activity = np.random.normal(loc = 1, scale = 1, size = self.patterns.shape)
        self.patterns = self.actFunction(self, activity = activity)

    #####################noiseMethods##########################################
    # These methods are used to create noisy version of given patterns
    def makeNoiseAccordingSparsity(self,pattern=None, noise_levels=None):
		'''
		Each noise cell fires with P(fire) = self.sparsity. A cell that fires have valeu 1 others 0

		:param pattern: Patterns which are made noisy.
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of cells that fire wrongly
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		j=1
		for i in noise_levels[1:]:
			wrong = np.array(map(random.sample, [range(pattern.shape[-1])]*pattern.shape[-2], [i]*pattern.shape[-2]))
			noise[j][np.mgrid[0:pattern.shape[0], 0:i][0], wrong] = np.random.uniform(0,1, size =(wrong.shape)) <= self.sparsity
			j +=1
		return noise

    def makeNoiseRandomFire(self,pattern=None, noise_levels=None):
		'''

		Each noisy cell fires accroding to the rate of an arbritrayly chosen cell in that pattern

		:param pattern: Patterns which are made noisy.
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of cells that fire wrongly
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		j=1
		for i in noise_levels[1:]:
			wrong = np.array(map(random.sample, [range(pattern.shape[-1])]*pattern.shape[0], [i]*pattern.shape[0])) # for each pattern a random set of cells is chosen to fire wrongly
			noise[j][np.mgrid[0:pattern.shape[0], 0:i][0], wrong] = np.array(random.sample(np.ravel(pattern), i*pattern.shape[0])).reshape(wrong.shape)
			j +=1
		return noise

    def makeNoiseZero(self,pattern=None, noise_levels=None):
		'''

		Each noisy cell becomes silent.

		:param pattern: Patterns which are made noisy.
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here the noise_levels increase linearly from 0 to number_winner in noise_levels.shape[0] steps. Each level determines, how many cells that fired before are silent in the noisy pattern
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		max_noise = pattern.shape[-1]
		for p in pattern:
			max_noise = min(np.flatnonzero(p).shape[0], max_noise)
		self.noise_levels = np.array(np.linspace(0, max_noise-1, noise_levels.shape[0]), dtype = 'int')
		noise = np.tile(pattern, (len(noise_levels),1,1))
		level=1

		for i in self.noise_levels[1:]:
			for j in range(pattern.shape[0]):
				wrong = random.sample(np.flatnonzero(pattern[j]), i)
				noise[level,j][wrong] = 0
			level +=1

		return noise


	#later important
    def makeNoiseVector(self,pattern=None, noise_levels=None):
		'''

		For each noise_level a noise vector is added to the pattern. The noise vector is a random vector created by a normal distribution.

		:param pattern: Patterns which are made noisy.
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of noise vectors added.
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		rand_vectors = np.random.normal(size = (len(noise_levels), self.cells))
		normalize(rand_vectors)
		rand_vectors *= 10
		noise_vector = rand_vectors[0]
		for i in range(len(noise_levels)-1):
			noise[i+1] += noise_vector
			noise_vector += rand_vectors[i+1]
		return noise

    def makeNoiseCovariance(self,pattern=None, noise_levels=None):
		'''

		For each noise_level a noise vector is added to the pattern. The first noise vector is the eigenvector with the largest eigenvalue of the covariance matrix of self.patterns. The second is the sum of the first two and so on.

		:param pattern: Patterns which are made noisy.
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of noise vectors added.
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		Cor = Corelations(patterns_1 = self.patterns)
		ev = Cor.getEigenvectorsCovariance().T
		normalize(ev)
		ev *= 10
		noise_vector = ev[0]
		for i in range(len(noise_levels)-1):
			noise[i+1] += noise_vector
			noise_vector += ev[i+1]
		return noise

	###################### getOutputFunctions #################
    ### Similar as in class Network
    #important for you
    def getOutputWTA(self, activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given activity; the highest self.number_winner activated neurons fire;
        :param activtiy: activity
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        :param return: array
        '''
        size = activity.shape
        winner = np.argsort(activity)[...,-self.number_winner:size[-1]]


        fire_rate = np.ones(size, 'bool')
        out_fire = np.zeros(size, 'bool')

        if len(size) ==1:
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3:
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =1#fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        return out_fire

    def getOutputWTALinear(self, activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
        '''
        abstract
        calculates outputfiring  given activity; the highest self.number_winner activated neurons fire;
        :param activtiy: activity
        :type array of max 4 dimension, last one must be self.connection.shape[1]
        :param return: array
        '''
        size = activity.shape
        winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

        fire_rate = activity
        fire_rate[fire_rate<0] = 0
        out_fire = np.zeros(size)

        if len(size) ==1:
            out_fire[winner] = fire_rate[winner]
        if len(size) ==2:
            out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
        if len(size) ==3:
            indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
            out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
        if len(size) ==4: # env, noise, time, pattern
            indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
            out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
        if len(size) > 4:
                print 'error in input dimension in calckWTAOutput'
        return out_fire


    def getOutputId(self, activity=None, **kwargs):
		'''

		calculates outputfiring  given input_pattern; Firing rate of all neurons is equal to their activity level

		:param input_pattern: input
		:param return: firing of the outputcells;
		:type return: array
		'''
		return activity


class Spatial():
	
	'''
	
	Spatial firing statistics of cells.
	It finds place fields and calculates their sizes etc. 
	
	:param patterns: population activites that the class shall analysis, should be an array of dimension 2 (pattern, cell)
	:param In: If In != None, it analyses the pattern of that Input Class
	:param  min_rate: It is the minimal activity of the cell that is considered as an active location. It will be given as the fraction of the peak activity of the cell.
	:param min_size: minimum size of a field such that it is considered as place field. Given in cm
	:param max_size: maximum size of a field such that it is considered as place field. Given in cm
	:param si_criterion: criterion whether an field as an place field by si and Treves 2009
	'''
	
	def __init__(self, In = None, patterns = None, cage = [1,1], min_rate = None, centers = np.array([]),min_size = None, max_size = None, si_criterion = False, **kwargs):
		
		if In != None:
			patterns = In.getInput()
			centers = In.centers
			cage = In.cage
		
		self.patterns_1 = patterns #dim 2 (location, cell)
		if len(centers.shape) == 3:
			self.centers = centers.reshape(centers.shape[0]*centers.shape[1], centers.shape[2])#center of the place fields; if determined by hand
		else:
			 self.centers = centers
		self.cage = cage
		self.cluster_size = np.zeros([patterns.shape[-1], patterns.shape[0]], dtype = 'int') # (cell, cluster) = cluster_size in cm^2; since number of clusters is yet unkown, last dimension is number patterns.
		self.clusters_colored = np.zeros(list(np.shape(self.patterns_1)[::-1]))# locations that are considered as firing fields; (cell, location); different fields have differnt numeric value
		self.number_fields = np.zeros(self.patterns_1.shape[1], 'int')# no of fields per cell
		self.noise = np.zeros(self.patterns_1.shape[1], 'int') #no of firing locations of a cell that do not belong to a firing place field
		self.makeLocations(self.patterns_1.shape[0])
		#no pixel in x,y direction
		self.x_length = self.cage_proportion*self.space_res
		self.y_length = self.space_res
		
		self.pixel_to_cm2 = cage[0]*cage[1]*10000./self.patterns_1.shape[0]
		self.min_size = None
		self.max_size = None
		self.si_criterion = si_criterion
		if si_criterion:
			print 'si criterion'
		if min_size != None:
			self.min_size = min_size * 1./self.pixel_to_cm2 #min_size in pixel
			print 'min size to be pf', self.min_size, ' pixel and ' , min_size, ' cm^2'
		if max_size != None: #not implemented
			self.max_size = max_size * 1./self.pixel_to_cm2 #min_size in pixel
			print 'max size to be pf', self.max_size, ' pixel and ' , max_size, ' cm^2'
		if min_rate != None:
			print 'min rate ' , min_rate
			min_fire = np.max(self.patterns_1, axis = -2) * min_rate
			min_fire = np.tile(min_fire , (self.patterns_1.shape[0],1))

			self.patterns_1[self.patterns_1 < min_fire] = 0
		
		#self.x_length = np.sqrt(self.patterns_1.shape[0])# number of pixel for one horizontol row in the enviroment
		#self.locations = np.ravel((np.mgrid[0:self.x_length, 0:self.x_length] + 0.0)/self.x_length, order = 'F').reshape(self.patterns_1.shape[0], 2)# all locations in the enviroment
		self.patterns_2d = self.patterns_1.reshape(self.y_length, self.x_length, self.patterns_1.shape[1])
		self.patterns_white = preprocessing.scale(self.patterns_1)
	
	def makeLocations(self, number_patterns): #help function
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.sqrt(number_patterns*1./self.cage_proportion)

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]
				#self.locations = self.locations.reshape(number_patterns, 2)
	
	def getSpatial(self, cell = None):#help function
		'''
		
		returns spatial firing of given cell
		'''
		return np.copy(self.patterns_1[:,cell])
	
	def calcClusters(self, cells=None):#help function
		
		print 'start calculating number fields'
		begin = time.time()
		fields = []
		in_cluster = set([])
		max_size = 1 #side length of quader, two firing locations in that quader are considered connected.
		#!
		#if cells == None:
		cells = range(self.patterns_1.shape[1])
		for cell in cells:
			cell_fire = self.getSpatial(cell = cell)
			all_cluster_cell = [] #all clusters the cell have
			loc_fire = set(np.flatnonzero(cell_fire))
			not_visited = set(np.flatnonzero(cell_fire))
			while len(not_visited) >0:
				quader = set([])
				loc = not_visited.pop() # random fire loc, is called and removed

				ball_ind_x =np.arange(-max_size, max_size+1) + loc #x coordinates of quader
				y_row = int(loc/self.x_length) # y coordinated (row) of loc


				ball_ind_x = ball_ind_x[ball_ind_x - y_row*self.x_length >= 0] #cut ball if it goes over one row on the left

				ball_ind_x = ball_ind_x[ball_ind_x - y_row*self.x_length < self.x_length]#cut if it goes over the row on the right

				ball =np.tile(ball_ind_x , 2*max_size+1) + (np.arange(-max_size,max_size+1)*self.x_length).repeat(len(ball_ind_x)) #expand it in y direction
				
				ball = ball[ball >=0] # cut it if it goes below the first row
				ball= ball[ball < cell_fire.shape[0]]#cut it if it goes beyond the last one

				quader = set(ball).intersection(loc_fire)# the firing locations in the quader

				
				if len(quader) >=2: # if loc is not isolated
					ind_overlap = np.flatnonzero(np.array(map(quader.isdisjoint, all_cluster_cell))-1)
					if len(ind_overlap)== 0: #if no connection to any cluster
						all_cluster_cell.append(quader)
					else: #if quader has overlap with some other cluster, union those two
						all_cluster_cell[ind_overlap[0]]=all_cluster_cell[ind_overlap[0]].union(quader)
						for i in ind_overlap[1:][::-1]: # and if new quader connects two cluster that where disjoint before, union them as one
							all_cluster_cell[ind_overlap[0]]=all_cluster_cell[ind_overlap[0]].union(all_cluster_cell[i])
							all_cluster_cell.pop(i)
						#all_cluster_cell = list(np.delete(all_cluster_cell, ind_overlap[1:]))
					#not_visited -= quader
				else:
					self.noise[cell]+= len(quader)
			
			if self.si_criterion:
				max_fire = np.max(self.patterns_1)
				for i in np.arange(len(all_cluster_cell))[::-1]:
					av_fire_field = np.sum(self.patterns_1[:,cell][list(all_cluster_cell[i])])*1./(len(all_cluster_cell[i]))
					max_fire_field = np.max(self.patterns_1[:,cell][list(all_cluster_cell[i])])

					if (av_fire_field < 0.1 * max_fire) or max_fire_field < 0.15 * max_fire:
						print 'field removed'
						print av_fire_field
						print max_fire_field
						self.noise[cell]+= len(all_cluster_cell.pop(i))
			if self.min_size != None:
				for i in np.arange(len(all_cluster_cell))[::-1]:
					if len(all_cluster_cell[i]) < self.min_size:
						#print 'pf size < min size!'
						#print len(all_cluster_cell)
						#print all_cluster_cell[i]
						#print len(all_cluster_cell[i])
						self.noise[cell]+= len(all_cluster_cell.pop(i)) #deletes all_cluster_cell[i] and returns it to noise
						#print len(all_cluster_cell)
					else:
						if self.max_size!= None and len(all_cluster_cell[i]) > self.max_size:
							self.noise[cell]+= len(all_cluster_cell.pop(i)) #deletes all_cluster_cell[i] and returns it to noise
						

			for i in np.arange(len(all_cluster_cell))[::-1]:
				self.clusters_colored[np.array([cell]*len(all_cluster_cell[i]), 'int'),np.array(list(all_cluster_cell[i]), 'int')] = 1*(1+10*(i+1))
				self.cluster_size[cell][i] = len(all_cluster_cell[i])*self.pixel_to_cm2
				#print self.cluster_size[cell][i]
			self.number_fields[cell] = copy.copy(len(all_cluster_cell))
		tim = time.time() - begin
		print 'finished in '+str(int(tim/3600)) + 'h '+str(int((tim-int(tim/3600)*3600)/60)) + 'min ' +str(int(tim - int(tim/3600)*3600- int((tim-int(tim/3600)*3600)/60)*60)) + 'sec'
	
	def getDistanceOfMin(self, activity_map = None, ref_point = None):#help function
		#activity_map = self.getSpatial
		loc_min = self.location[np.argmin(activity_map)]
		min_dis = self.getDistanceMatrix(locations = self.locations)[loc_min, ref_point]
		return min_dis
	
	
	################################## average number of fields in certain populations #############################################
	def getAverageFieldNumber(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.number_fields)*1./(self.patterns_1.shape[1])
		
	def getAverageFieldNumberActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.number_fields)*1./len(np.flatnonzero(self.number_fields+self.noise))
	
	def getAverageFieldNumberActiveCellsWithField(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.sum(self.number_fields)*1./len(np.flatnonzero(self.number_fields))
	
	def getAverageFieldNumberActiveCellsWithFieldStd(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.std(self.number_fields)
	
		
	################################## average sizes of fields in certain populations #############################################
	def getAverageFieldSize(self): # returns average field size of all fields;
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.sum(self.cluster_size)*1./np.sum(self.number_fields)
		
	def getAverageFieldSizeStd(self): # returns average field size of all fields;
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.std(self.cluster_size)
		
	def getAverageFieldSizeCell(self): #returns averge field size of each cell (cell); a cell without have field has av size 0
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1)/(self.number_fields[non_z]+0.0)
		return np.round(av_size,3)
		
	def getAverageCoverActiveCellsWithField(self): #average cover of pc, (sum pf coverage of all fields the cell has)
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1)
		non_z = np.flatnonzero(av_size).shape[0]*1.
		return np.sum(av_size)/non_z
		
	def getAverageCoverActiveCells(self): #average coverage of a active cell
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields+self.noise)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1) + np.sum(self.cluster_size[non_z], axis = -1)
		non_z = np.flatnonzero(av_size).shape[0]*1.
		return np.sum(av_size)/non_z
		
		
	################################## other helpful statistics #############################################		
	def getAverageNoiseActiceCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.noise)/(len(np.flatnonzero(self.number_fields+self.noise))+0.0)
		
	def getNumberActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return len(np.flatnonzero(self.number_fields+self.noise))+0.0
		
	def getNumberCellsWithField(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return len(np.flatnonzero(self.number_fields))+0.0
		
	def getActiveCell(self, no_cells = 1): #return indizes of active cells
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.flatnonzero(self.number_fields+self.noise)[:no_cells]
	
	def getPlaceCell(self, no_cells = 1): #return indizes of place cells
		if (self.cluster_size == 0).all():
			self.calcClusters()
		cells = np.flatnonzero(self.number_fields)
		np.random.shuffle(cells)
		return cells[:no_cells]
		
	def getProportionActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return self.getNumberActiveCells()/self.patterns_1.shape[-1]
		
	def getProportionCellsWithField(self):
		return self.getNumberCellsWithField()/self.patterns_1.shape[-1]
		
	def getSizesOverNumberFields(self):#returns field sizes dependent of number fields the cell has, (number_fields, sizes)
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_fields = np.max(self.number_fields)

		sizes = []
		for i in range(1,max_fields+1):
			#sizes1 = np.ravel(self.cluster_size[self.number_fields == i])
			#if size1s.shape[0]>=0:
			sizes.append(np.ravel(self.cluster_size[self.number_fields == i][self.cluster_size[self.number_fields == i]>0]))
		return sizes
		
	def getCellsMaxFieldSize(self, cells = 1):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_fields = np.max(self.cluster_size, axis = -1)
		return np.argsort(max_fields)[-cells:]
		
	def getfieldSizes(self):#returns the field sizes considered as fields
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return self.cluster_size[self.cluster_size!= 0]
		
	def getfieldSizesSorted(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sort(self.cluster_size[self.cluster_size!= 0])
		
	def getCellsWithFieldSize(self, size, cells =1):# return cell indizes (cells many) that have place field size just above size; if there are not enough many of that kind, returns indizes with max size
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_size_cell = np.max(self.cluster_size, axis = -1)
		argsort = np.argsort(max_size_cell)
		return_cell = argsort[max_size_cell[argsort]>=size]
		if return_cell.shape[0] < cells:
			return_cell = self.getCellsMaxFieldSize(cells)
		else:
			return_cell = return_cell[:cells]
		return return_cell
		
	def getMaxFieldSizeCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.max(self.cluster_size, axis = -1)
	
	def getAverageInsideOutsideWrong(self, noisy_patterns):
		
		fire_cells = np.flatnonzero(self.number_fields)
		fire_locs = np.copy(self.clusters_colored[fire_cells])
		noisy_fire_locs = np.copy(noisy_patterns).T[fire_cells] #cells, locations
		fire_locs[fire_locs!=0] = 1

		noisy_fire_locs[noisy_fire_locs!=0] = 1
		wrong = fire_locs- noisy_fire_locs
		
		inside_wrong = np.copy(wrong)
		inside_wrong[inside_wrong == -1] = 0
		inside_wrong = np.sum(inside_wrong, axis =-1)*1./np.sum(fire_locs, axis = -1)

		
		
		outside_wrong = np.copy(wrong)*-1
		outside_wrong[outside_wrong == -1] = 0
		outside_wrong = np.sum(outside_wrong, axis =-1)*1./(self.patterns_1.shape[0]-np.sum(fire_locs, axis = -1))

		return [inside_wrong,outside_wrong]
		
	def getAverageProportionWrong(self, noisy_patterns):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		wrong = np.zeros([noisy_patterns.shape[0], len(np.flatnonzero(self.number_fields))]) #noise; cells with field
		for noise in range(noisy_patterns.shape[0]):
			inside, outside = self.getAverageInsideOutsideWrong(noisy_patterns[noise])
			wrong[noise] = (inside +outside)/2.
		return np.mean(wrong, axis = -1)
		
	def getAverageProportionWrongInsideOutside(self, noisy_patterns):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		wronginside = np.zeros([noisy_patterns.shape[0], len(np.flatnonzero(self.number_fields))]) #noise; cells with field
		wrongoutside = np.zeros([noisy_patterns.shape[0], len(np.flatnonzero(self.number_fields))]) #noise; cells with field
		for noise in range(noisy_patterns.shape[0]):
			wronginside[noise], wrongoutside[noise] = self.getAverageInsideOutsideWrong(noisy_patterns[noise])
			
		return np.mean(wronginside, axis = -1), np.mean(wrongoutside, axis = -1)		
		
	def getAverageRadiusCell(self): #the average radius of one cells place fields
		return np.sqrt(self.getAverageFieldSizeCell()/(np.pi*self.patterns_1.shape[0])) # no_pixel = space_res**2 * pi * r**2
	
	def getAverageRadius(self): #the average radius of all place fields in the population
		return np.sqrt(self.getAverageFieldSize()/(np.pi*self.patterns_1.shape[0])) # no_pixel = space_res**2 * pi * r**2
	
	def getSparsityCell(self): #the sparsity level of each cell
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return (np.sum(self.cluster_size, axis = -1) + self.noise)/(self.patterns_1.shape[0]*1.)
		
	def getSparsityLocation(self): #the sparsity level of each location
		return np.sum(self.patterns_1, -1)*1./self.patterns_1.shape[-1] 
		
	def getAverageSparsityLocation(self): #the average sparsity level over locations
		sp = self.getSparsityLocation() #the sparsity level of each location
		return np.sum(sp)/sp.shape[0]
	def plotBump(self, pattern = None):
		

		fig = plt.Figure()
		plt.jet()
		plt.scatter(self.centers[:,0], self.centers[:,1], c= 'w', faceted = 1)
		plt.scatter(self.centers[:,0][np.nonzero(self.patterns_1[pattern])], self.centers[:,1][np.nonzero(self.patterns_1[pattern])], c= 'r', ec = 'none')
		plt.scatter(self.locations[:,0][pattern], self.locations[:,1][pattern], c= 'b', ec = 'none', s = 60)
		#plt.colorbar()
		return fig
	
	def plotCellFiringClusters(self, ax = None, cell = None): #plot the cells spatial firing, each detected place field has different color
		
		if (self.cluster_size == 0).all():
			self.calcClusters()
		
		ax.set_xlim(0,self.cage[0])
		ax.set_ylim(0,self.cage[1])
		#plt.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])], self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= 'w', faceted = 1)
		#plt.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])], self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= 'g', alpha = 0.5, faceted = False)
		ax.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])] ,self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= self.clusters_colored[cell][np.flatnonzero(self.clusters_colored[cell])], alpha = 0.5, faceted = False)
		ax.text(0,1, ('no_field = '+str(self.number_fields[cell]) +'\ns '+ str(self.cluster_size[cell][self.cluster_size[cell]!=0])))
	
	def plotCellFiring(self, cell = None):#just plots the spatial firing

		plt.figure()
		plt.jet()
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.scatter(self.locations[:,0][np.flatnonzero(self.patterns_1[:,cell])], self.locations[:,1][np.flatnonzero(self.patterns_1[:,cell])], c= self.patterns_1[:,cell][np.flatnonzero(self.patterns_1[:,cell])], ec = 'none')
		plt.colorbar()
	
	def plotCellFiring2D(self, cell = None):#just plots the spatial firing
		
		patterns_2d = self.patterns_1.reshape(self.x_length, self.y_length, self.patterns_1.shape[1])[:,:,cell]
		plt.figure()
		plt.jet()
		plt.xlim(0,self.cage[0])
		plt.ylim(0,self.cage[1])
		plt.scatter(np.nonzero(patterns_2d)[0]/(self.x_length*1.), np.nonzero(patterns_2d)[1]/(self.y_length*1.), c= patterns_2d[np.nonzero(patterns_2d)], ec = 'none')
	
	def calcSpatialAutocorrelation2(self, cell = None, mode = 'full'): #calc spatial autocorrelation (Pearson Coefficient); border effects are corrected to 1, -1
		mode = 'full'
		self.number_evaluations = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), mode)
		#r = np.zeros(self.number_evaluations.shape)
		xy = scipy.signal.correlate(self.patterns_2d[:,:,cell]*1, self.patterns_2d[:,:,cell]*1, mode)/(self.number_evaluations*1.)
		
		mean = np.sum(self.patterns_1[:,cell])*1./self.patterns_1.shape[0] #mean of firing map
		std = np.sqrt(mean - mean**2)#std of firing map
		
		mean_shifted_map = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), self.patterns_2d[:,:,cell]*1, mode)/(self.number_evaluations*1.) #means of the shifted maps; differs form mean, since not all locations are considered if map is shifted and zero padded (with zeros extended at the border)
		std_shifted_map = np.sqrt(mean_shifted_map - mean_shifted_map**2)
		
		#r[std_shifted_map!=0] = (xy[std_shifted_map!=0] - mean * mean_shifted_map[std_shifted_map!=0])/(std * std_shifted_map[std_shifted_map!=0])
		r = (xy - mean * mean_shifted_map)/(std * std_shifted_map) # pearson coefficeint
		
		#map corrected outsiders
		#plt.figure()
		#if mode == 'full':
			#grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)

		#if mode == 'same':
			#grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		#plt.scatter(grid[0][r**2>1],grid[1][r**2>1], r[r**2>1])
		#plt.jet()
		#plt.colorbar()
		r[r>1] = 1 # correct outsiders
		r[r<-1] = -1
		return r.T
		
	def calcSpatialAutocorrelation(self, cell = None, mode = 'full'): #calc spatial autocorrelation (Pearson Coefficient); border effects are corrected to 1, -1

		
		cell_firing = self.patterns_white[:,cell].reshape(self.y_length, self.x_length)
		xy = scipy.signal.correlate2d(cell_firing,cell_firing, mode = mode)
		return xy		
		
	def getSpatialAutocorrelation(self, cell = None, mode = 'full'):#plot spatial autocorrelation of cell, or if cell = None, the average of spatial corrleations of all cells
		
		if cell ==None:
			auto_corr = self.calcSpatialAutocorrelation(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutocorrelation(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.* self.patterns_1.shape[0]

		
		else:
			auto_corr = self.calcSpatialAutocorrelation(cell = cell, mode = mode)/(1.* self.patterns_1.shape[0])
		
		#if mode == 'full':
			#grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)

		#if mode == 'same':
			#grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		return auto_corr
	
	def plotSpatialAutocorrelation(self, cell = None, mode = 'full'):#plot spatial autocorrelation of cell, or if cell = None, the average of spatial corrleations of all cells
		
		if cell ==None:
			auto_corr = self.calcSpatialAutocorrelation(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutocorrelation(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.

		else:
			auto_corr = self.calcSpatialAutocorrelation(cell = cell, mode = mode)
		
		plt.figure()
		plt.jet()
		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[1]+1:self.patterns_2d.shape[1]]/(self.x_length+0.0)

		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[1]/2:self.patterns_2d.shape[1]/2]/(self.x_length+0.0)
		plt.scatter(grid[0],grid[1],c =auto_corr)
		plt.colorbar()
		
	def getSpatialAutoOverlap(self, cell = None, mode = 'full'):#calc overlap with original and shifted map; normalized; border effects are corrected
		mode = 'full'
		
		def calc():
			no_firing_times = np.sum(self.patterns_1[:,cell])
			if no_firing_times == 0:
				r = np.ones(np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]][0].shape)			
			else:
				self.number_evaluations = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[0]]), np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[0]]), mode)#number of locations that are compared (less shift many locations are compared, great shift, just little many)
				r = scipy.signal.correlate(self.patterns_2d[:,:,cell]*1, self.patterns_2d[:,:,cell]*1, mode)*self.patterns_1.shape[0]/(self.number_evaluations*1.*no_firing_times)#normalized by number of locations considered and 1/a
				r[r>1] = 1 # correct outsiders
			return r.T
			
			
		if cell == None:
			cell = 0
			r = calc()
			for i in range(1,self.patterns_1.shape[1]):
				cell +=1
				r += calc()
			r /= self.patterns_1.shape[1]*1.
		else:
			r = calc()
		return r
		
	def plotSpatialAutoOverlap(self, cell = None, mode = 'full'): #plots  spatial overlap of cell, or average of all cells
		
		
		if cell ==None:
			auto_corr = self.calcSpatialAutoOverlap(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutoOverlap(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.
		else:
			auto_corr = self.calcSpatialAutoOverlap(cell = cell, mode = mode)
		
		plt.figure()
		plt.jet()
		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)
		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		plt.scatter(grid[0],grid[1],c =auto_corr)
		plt.xlim([-1,1])
		plt.ylim([-1,1])
		plt.colorbar()
		
	def getAutoCorrelationPoints(self, mode = 'full'): #returns points to plot autocorrelation scatterplot

		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)
		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		return grid

	def getSpatialInformation(self):
		mean_cell = np.sum(self.patterns_1, -2)
		non_zero_cells = np.flatnonzero(mean_cell != 0)
		mean_cell = np.tile(mean_cell, (self.patterns_1.shape[0],1))/(self.patterns_1.shape[0]*1.)
		mean_cell[mean_cell==0] = 1

		terms = self.patterns_1*1./mean_cell
		terms[terms == 0.0] = 10**-10
		r = np.zeros(self.patterns_1.shape[-1])

		try: 
			r[non_zero_cells] =np.sum(terms[:,non_zero_cells] * np.log2(terms[:,non_zero_cells]), -2) *1./self.patterns_1.shape[0]
		except RuntimeWarning:
			print 'mins', np.min(terms),'\t',np.min(self.patterns_1),'\t',np.min(mean_cell)
		#print 'mins', np.min(terms),'\t',np.min(self.patterns_1),'\t',np.min(mean_cell)
		return r




class Grid(Input, Spatial):
	'''
	
	Input Class that creates Grid pattern.
	
	:param grid_mode: how grid parameter are set up. If 'linear' spacings increase linearly from 30-50cm, orientations are random. If 'modules' it has 4 Modules with similar spacings and orientations. Phase is in both cases randomly chosen
	:type grid_mode: 'linear' or 'modules'
	:param rat: When grid_mode is 'modules' it determines how modules are split up. rat = 0 it has only modules 1 and 2, rat=1 it has all 4, rat=2 it has only modules 3 and 4.
	:type rat: 0,1 or 2
	:param space: If grid_mode = 'linear' it uses space as spacings, rather than 30-50cm
	:type space: array with lenght equal to number_cells
	'''

	def __init__(self,  grid_mode = 'modules', rat =1, spacings = None, peak = None, theta = None, phase = None, cage = [1,1], r_to_s = 0.32,**kwargs):
		


		#Grid Attributes
		self.spacing = spacings
		self.peak_firing_rate = peak
		self.theta = theta
		self.phase = phase

		
		self.theta_given = 0
		self.spacing_given = 0
		self.peak_given = 0
		self.phase_given = 0
		if spacings != None:
			self.spacing_given = True
		if peak != None:
			self.peak_given = True
		if theta != None:
			self.theta_given = True
		if phase != None:
			self.phase_given = True


		self.grid_mode = grid_mode
		self.rat = rat
		self.cage = cage
		self.r_to_s =r_to_s # radius of field / spacing of cell; Defining field as everthing above 0.2 peak rate, then in Knierim r_to_s = .17. Using Hafting r_to_s is in [0.3, 0.32]; Supposing that at 0.21 * spacing fire rate is 0.5 * peak (as in deAlmeida), r_to_s = 0.32.
		self.k = self.r_to_s**2/np.log(5) #see calcActivity
		
		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)


		
	def makeLocations(self, number_patterns):
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.int(np.sqrt(number_patterns/self.cage_proportion))

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
				self.x_length= self.cage_proportion*self.space_res
				self.y_length = self.space_res
			
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)

				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]
				self.y_length= self.cage_proportion*self.space_res
				self.x_length = self.space_res
				
		
	

	
	def makeInput(self,**kwargs):
		print 'kwargs in makeinput Grid', kwargs

		self.makeGrid()
		self.patterns = self.calcGridFiring()

	def makeGrid(self): # makes grid; set ups the parameter
		
		self.n_grid = self.cells 
		ec_firing = np.zeros([self.n_e, self.space_res**2, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		if not self.spacing_given:
			self.spacing = np.zeros([self.n_e,self.n_grid]) +0.0	# space between two peaks in one grid cell
		if not self.theta_given:
			self.theta = np.zeros([self.n_e,self.n_grid])		# Orientation of the grid			
		if not self.peak_given:
			self.peak_firing_rate = np.ones([self.n_e,self.n_grid, 50, 50])# Peak firing rate
		if not self.phase_given:
			self.centers = np.zeros([self.n_e, self.n_grid, 2])	# Origin of the grid
		else:
			self.centers = np.ones([self.n_e, self.n_grid, 2]) * self.phase	# Origin of the grid
		self.field_size = np.zeros([self.n_e,self.n_grid])	# Field size in one grid cell
		self.theta_arc =np.zeros([self.n_e,self.n_grid])	#Orientation of the grid presented in Pi coordinates
		self.rotation = np.zeros([self.n_e,self.n_grid,2,2]) #Rotation Matrices; We first bulit always a grid with 0 orientation and then rotate the whole thing to the true orientation via these matrices


		# for each enviromnent define grid parameters similar found in literature; linear
		if self.grid_mode == 'linear':
			self.gridLinear()
		if self.grid_mode == 'modules':
			self.gridModules()
		
		# parameters needed for calculating the rate:
		self.cos_spacing = self.spacing*np.cos(np.pi/3) # how far i go to the right-left direction, when I move one grid point upwards
		self.sin_spacing = self.spacing*np.sin(np.pi/3) # how far i go to the up-down direction, when I move one grid point to the right
		#self.std = self.field_size**2/np.log(5) #modulates the Gauss according to place field size; 
		
	
	def gridLinear(self):
		'''
		
		Grid is made according to Hafting 2005:  with random origin (phase), random orientation, spacing increasing from 30-50cm.
		'''
		self.modules = np.array(np.linspace(0, self.n_grid, 5), 'int')
		for h in range(self.n_e):
			theta_change = np.random.uniform(0,60)
			origin_change = np.random.uniform(-.25, 0.25, size = 2)
			for i in range(self.n_grid):
				noise_spacing = 0#random.gauss(0, 0.01) #noise mean 0, varianz=1cm
				noise_field_size = 0#random.gauss(0, 0.01)#noise varianz=1cm
				if not self.peak_given:
					self.peak_firing_rate[h][i] = 1
				if h == 0:
					if not self.spacing_given:
						self.spacing[h][i] = 0.3 + (0.2/self.n_grid)*i + noise_spacing 	#Generate a baseline spacing, starting at 0.30m and increasing linearly to0.50m
					#if self.fields == None:
						#self.field_size[h][i] = 0.0977 + 0.0523*(1.0/self.n_grid)*i+ noise_field_size 	#Generate a baseline field size, starting at 9,77cm and increasing linearly 15cm; field size in radius; (area from 300-700cm**2)
					#else:
						#self.field_size[h][i] = self.fields[i]
					if not self.theta_given:
						self.theta[h][i] = random.uniform(0, 360)
					if not self.phase_given:
						self.centers[h][i] = np.array([random.uniform(0, 1),random.uniform(0, 1)])
				else:
					self.spacing[h][i] = self.spacing[0][i] + noise_spacing
					self.field_size[h][i] = self.field_size[0][i] + noise_field_size
					self.theta[h][i] = self.theta[0][i] + theta_change #+ random.gauss(0,2)
					self.centers[h][i] = self.centers[0][i] + origin_change
				self.theta_arc[h][i] = (2*np.pi*self.theta[h][i])/360.
				self.rotation[h][i] =[[np.cos(self.theta_arc[h][i]), np.sin(self.theta_arc[h][i])], [-np.sin(self.theta_arc[h][i]), np.cos(self.theta_arc[h][i])]]


	def gridModules(self):
		'''
		
		Grid is made according to Solstad 2012: It consists of 4 Modules, each having similar orientation and spacing. Phase is random.
		'''
		
		if self.rat ==0: #m1 and m2 only
			m1 = 19 + 2.5
			m2 = 30+2.5
			m3 = 0
			m4 = 0
			thres = np.array([m1, m1+m2, m1+m2+m3, 54.0001])/54.
			spacing_mean = [38.8, 48.4, 65., 98.4]
			spacing_var = [8,8,6,16]
			orient_mean = [-5, -5, 5, -5]
			orient_var = [3,3, 6, 3]
		
		
		if self.rat ==1:
			m1 = 19 + 2.5 #34.7%
			m2 = 30+2.5 #52.4%
			m3 = 5 #8%
			m4 = 3 #4.8%
			thres = np.array([m1, m1+m2, m1+m2+m3, 62.0001])/62.
			#spacing_mean = [38.8, 48.4, 65., 98.4]
			spacing_mean = [48.4, 65., 98.4, 140.8]
			#spacing_var = [8,8,6,16]
			spacing_var = [8,8,8,8]
			#orient_mean = [-5, -5, 5, -5]
			#orient_mean = np.random.uniform(0,60, size = 4)
			orient_mean = np.array([15,30,45,60])
			#orient_var = [3,3, 6, 3]
			orient_var = [3,3, 3, 3]
			
		if self.rat ==2:#only m3 and m4
			m1=0
			m2=0
			m3 = 5
			m4 = 3
			thres = np.array([m1, m1+m2, m1+m2+m3, 8.0001])/8.
			spacing_mean = [38.8, 48.4, 65., 98.4]
			spacing_var = [8,8,6,16]
			orient_mean = [-5, -5, 5, -5]
			orient_var = [3,3, 6, 3]
			#!!!!
			#orient_mean = [-5, -5, -5, -5]
			#orient_var = [3,3, 3, 3]
		
		if self.rat ==3:
			#m1 = 19 + 2.5 #34.7%
			#m2 = 30+2.5 #52.4%
			#m3 = 5 #8%
			#m4 = 3 #4.8%
			m1 = 10 #34.7%
			m2 = 10 #52.4%
			m3 = 22 #8%
			m4 = 20 #4.8%
			print "NEw SPACCCCCCCCCCCCCCCCCCCCCCCCCCING RAAAAAAT ONE"
			print m1, m2, m3, m4
						
			thres = np.array([m1, m1+m2, m1+m2+m3, 62.0001])/62.
			#spacing_mean = [38.8, 48.4, 65., 90.4]
			spacing_mean = [48.4, 65., 98.4, 140.8]
			#spacing_var = [8,8,6,16]
			spacing_var = [4,4,4,4]
			#orient_mean = [-5, -5, 5, -5]
			#orient_mean = np.random.uniform(0,60, size = 4)
			orient_mean = np.array([15,30,45,60])
			#orient_var = [3,3, 6, 3]
			orient_var = [4,4, 4, 4]		
		
		
		
			
		self.modules = np.zeros(5, 'int')	
		choice = np.random.uniform(low = 0, high =1, size = self.n_grid) + 0.000001
		for i in range(1,self.modules.shape[0]):
			self.modules[i] = np.flatnonzero(choice <= thres[i-1]).shape[0]
			self.spacing[0][self.modules[i-1]:self.modules[i]] = np.random.normal(loc = spacing_mean[i-1], scale = np.sqrt(spacing_var[i-1]), size = self.modules[i]- self.modules[i-1])/100.
			self.theta[0][self.modules[i-1]:self.modules[i]] = np.random.normal(loc = orient_mean[i-1], scale = np.sqrt(orient_var[i-1]), size = self.modules[i]- self.modules[i-1])
		self.centers[0] = np.random.uniform(low =0, high = 1, size=(self.n_grid, 2))
		self.theta_arc[0] = (2*np.pi*self.theta[0])/360.
		for i in range(self.n_grid):
			self.rotation[0][i] =[[np.cos(self.theta_arc[0][i]), np.sin(self.theta_arc[0][i])], [-np.sin(self.theta_arc[0][i]), np.cos(self.theta_arc[0][i])]]
		
		if self.n_e >1:
			for h in range(1, self.n_e):
				theta_change = np.random.uniform(0,60, size = 4)
				origin_change = np.random.uniform(-.25, 0.25, size = (4,2))
				noise_spacing = 0#random.gauss(0, 0.01) #noise mean 0, varianz=1cm
				noise_field_size = 0#random.gauss(0, 0.01)#noise varianz=1cm
				for m in range(1,5):
					for i in range(self.modules[m-1],self.modules[m]):
						if not self.peak_given:
							self.peak_firing_rate[h][i] = 1
						self.spacing[h][i] = self.spacing[0][i] + noise_spacing
						self.field_size[h][i] = self.field_size[0][i] + noise_field_size
						self.theta[h][i] = self.theta[0][i] + theta_change[m-1]
						self.centers[h][i] = self.centers[0][i] + origin_change[m-1]
						self.theta_arc[h][i] = (2*np.pi*self.theta[h][i])/360.
						self.rotation[h][i] =[[np.cos(self.theta_arc[h][i]), np.sin(self.theta_arc[h][i])], [-np.sin(self.theta_arc[h][i]), np.cos(self.theta_arc[h][i])]]



	def calcActivity(self, h, location):
		'''
		 calculates and returns the activity in environment h at location location
		'''
		
		t_loc = np.einsum('ijk,ik->ij' ,self.rotation[h], (location - self.centers[h])) # shift and rotate location into coordinates, where the grid has origin at 0 and rotation 0
		k= np.floor(t_loc[:,1]/(self.sin_spacing[h])) #nearest vertex in up-down (y)  direction is the kth or k+1th one
		y1 = (t_loc[:,1] - k*self.sin_spacing[h]) # assume it is the kth one
		y2 = (t_loc[:,1] - (k+1)*self.sin_spacing[h]) # assume it is the k +1th one
		kx1 = np.round((t_loc[:,0]- k*self.cos_spacing[h])/(self.spacing[h])) 
		kx2 = np.round((t_loc[:,0]- (k+1)*self.cos_spacing[h])/(self.spacing[h]))
		x1 = (t_loc[:,0]-k*self.cos_spacing[h]- kx1*self.spacing[h])# nearest vertex in x direction for y1 (if it is the kth one)
		x2 = (t_loc[:,0]-(k+1)*self.cos_spacing[h]- kx2*self.spacing[h])# nearest vertex in x direction for y2
		#dis = np.minimum((x1*x1 + y1*y1), x2*x2+y2*y2) # = euclidean distance**2 to nearest peak
		
		arg = np.argmin(np.array([x1*x1 + y1*y1, x2*x2+y2*y2]), axis = -2)
		vertX = np.array([kx1, kx2], 'int')[arg, np.arange(arg.shape[0])]
		vertY = np.array([k, k+1], 'int')[arg, np.arange(arg.shape[0])]
		dis = np.array([[x1*x1 + y1*y1], [x2*x2+y2*y2]])[arg, np.zeros(arg.shape[0], 'int'), np.arange(arg.shape[0])]

		#activity= np.exp(-dis/self.std[h])*self.peak_firing_rate[h] # rate is calcualted acording to the distance
		activity = np.exp(-dis/(self.spacing[h]**2 * self.k))*self.peak_firing_rate[h][np.arange(vertY.shape[0]), vertX, vertY] # a = exp(-d**2/(s**2*k) as in Knierim, where k = r_to_s**2/log(5) defining everythin above 0.2 is in field


		return activity

		
	def calcGridFiring(self):
		'''
		
		returns population activity of grid over all environments and locations
		'''
		ec_firing = np.zeros([self.n_e, self.number_patterns, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		for h in range(self.n_e): # calc firing rate at each location in each enviroment
			activity = np.zeros([self.number_patterns, self.cells])
			activity[:, :self.n_grid] = np.array(map(self.calcActivity, [h]*self.number_patterns, self.locations))
			ec_firing[h] = self.actFunction(self, activity = activity)
		return ec_firing
	
	def getTrueRadius(self): #to avoid the smaller sizes aat the boarder, we compute the radius only of the largest field assuming all other have the same size
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_field_size_cell = np.max(self.cluster_size, axis = -1)
		return np.sqrt(max_field_size_cell/(self.x_length**2*np.pi))
		
	def getTrueRadiusAverage(self):
		return np.sum(self.getTrueRadius())/(self.cells+0.0)
		
	def getStoredlocations(self):
		'''
		
		returns the locations in which pattern are stored
		'''

		return self.locations[self.store_indizes]
	

class PlaceFields(Input):


	'''

	Input Class that creates pattern with place cells.

	:param no_fields: Number place fields per cell
	:type no_fields: int
	:param field_size: size of each field; Note if size of fields depend on self.sparsity for certain actFunction methods!
	:type field_size: int
	:param ordered: determines whether field centers of the cells are ordered or randomly distributed.
	:type ordered: bool
	:param active_in_env: Determines how many cells are allowed to be active in one environment
	:type active_in_env: int
	'''

	def __init__(self, no_fields = None, field_size = None, centers = None, ordered = 0, active_in_env = None, peak = None, cage = [1,1], center_cage = None, **kwargs):

		#Place Fields Attributes
		self.ordered = ordered #whether centers of place fields are evenly distrivuted or random
		self.centers =centers # place field center
		
		
		self.no_fields = no_fields #no fields per cell
		self.field_size = field_size # radius of one field; if actFunction is for example getOutputWTA size is only determined by number winner; otherwise, activity is set to 0 outside of the field, inside it has a guassian fire rate disribution
		self.n_e = kwargs['n_e']
		self.cells = kwargs['number_cells']
		self.peak_firing_rate = peak

		if no_fields == None:
			no_fields = np.ones([self.n_e, self.cells], 'int')*1
		if peak == None:
			print [self.n_e, self.cells, np.max(no_fields)]
			self.peak_firing_rate = np.ones([self.n_e, self.cells, np.max(no_fields)])
		self.active_in_env = active_in_env
		if self.active_in_env == None:
			self.active_in_env = self.cells
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')#which cells are active in which envrionment
		self.no_fields = np.zeros([self.n_e, self.cells],'int')
		for h in range(self.n_e):
				self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
				self.no_fields[h][self.active_cells[h]] = no_fields[h][self.active_cells] #no fields per cell
		self.max_fields = np.max(self.no_fields, axis = 1) #max field number in each env.
		if self.field_size == None:
			field_size = np.random.normal(loc = .1400, scale = .01, size = (self.n_e, self.cells, max(self.max_fields)))
			self.field_size = np.sqrt(field_size/np.pi)



		self.locations = None #x and y coordinates of all pixel
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.cage = cage
		if center_cage == None: # locations of possible field centers; can be bigger or smaller than cage
			self.center_cage = np.array(self.cage)
		else:
			self.center_cage = np.array(center_cage)
		#self.number_patterns= kwargs['number_patterns']
		#if kwargs['sparsity'] == None:
			#sparsity =
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)
		#Spatial.__init__(self, patterns = self.input_stored[0], cage = cage, **kwargs)


	def makeInput(self,**kwargs):
		self.number_winner = min(self.number_winner, self.cells)
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')#which cells are active in which envrionment
		if self.active_in_env != self.cells:
			print 'not all cells active in envoronment!!!!'
			for h in range(self.n_e):
					self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
		else:
			for h in range(self.n_e):
				self.active_cells[h] = np.arange(self.cells)
		self.makeFields()
		self.patterns = self.calcFiring()

	def makeFields(self):
		if self.centers == None:
			self.centers = np.zeros([self.n_e, self.cells, np.max(self.no_fields), 2])
			if self.ordered:
				for h in range(self.n_e):
					self.centers[h][:int(np.sqrt(self.active_in_env))] = np.ravel((np.mgrid[0:int(np.sqrt(self.active_in_env)), 0: int(np.sqrt(self.active_in_env))] + 0.0)/np.sqrt(self.cells), order = 'F').reshape(self.cells,1, 2)
					self.centers[h][int(np.sqrt(self.active_in_env)):] = np.random.sample(size = [self.active_in_env -int(np.sqrt(self.active_in_env)) ,1, 2])
			else:
				for h in range(self.n_e):
					self.centers[h] = np.random.sample(size = [self.cells, self.max_fields[h], 2]) * np.array(self.cage)*self.center_cage - (0.5*(self.center_cage-1)) * np.array(self.cage)
					for cell in range(self.cells):
						self.centers[h, cell, self.no_fields[h, cell]:] = -10000 #center of fields that exceed number of fields of a cell are put far away
		self.std = self.field_size**2/np.log(5) #modulates the Gauss according to place field size;

	def calcActivity(self, env ,location, field_number): #calc rate map for all cells at location for field_number
		loc_diff = self.centers[env, :, field_number] - location# diff loation to all fields (cells, (x,y))
		dis = np.sum(loc_diff * loc_diff, axis = -1)# eucledan distance squared to each field; (cell, x**2 +y**2)
		activity= np.exp(-dis/self.std[env, :, field_number])*self.peak_firing_rate[env,:,field_number] # rate is calcualted acording to the distance; activity = 0.2* peak rate if dis = size

		return activity #env ,pattern, cell_activity (with field)

	def calcFiring(self):
		#firing = np.zeros([self.n_e,self.number_patterns, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		activity = np.zeros([self.n_e, self.number_patterns, self.cells])
		for h in range(self.n_e):
			for i in range(self.max_fields[h]): # calc rate map for each field and sum up
				activity[h] += np.array(map(self.calcActivity, [h]*self.number_patterns, self.locations, [i]*self.number_patterns))
		indices = np.mgrid[0:self.n_e,0:self.number_patterns,0:self.active_in_env]
		#act_cells = self.active_cells.repeat(self.number_patterns, axis = 0).reshape(self.n_e, self.number_patterns, self.active_in_env)
		#firing[indices[0], indices[1], act_cells] = self.actFunction(self,activity = activity)
		firing = self.actFunction(self, activity = activity)
		return firing
		#return activity

	def getTrueRadius(self): #in a constructed PlaceField Code with known number of fields, we can calculate the radius in antoherway. To avoid smaller fieled sizes at border,we look at the cell with the largest Place fields and calculate the radius.
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_field_size_cell = np.max(self.getAverageFieldSizeCell())/(self.no_fields+0.0)
		return np.sqrt(max_field_size_cell/(self.x_length**2*np.pi))

	def getStoredlocations(self):
		return self.locations[self.store_indizes]

	def makeLocations(self, number_patterns):
		#if number_patterns == None:
			#number_patterns = self.number
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.sqrt(number_patterns*1./self.cage_proportion)

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]

##################################################################

def PlaceFieldsTest():
	fig = plt.figure()
	cells = 2500
	res = 2500



	#field_sizes = np.random.normal(loc = 1775, scale = 1500, size = cells*10**2)
	#field_sizes = field_sizes[field_sizes >= 200]
	#field_sizes = field_sizes[:cells*1].reshape(1,cells , 1)
	#no_fields = np.ones([1,cells])*1
	#field_in_r = np.sort(np.sqrt(field_sizes/np.pi)/100.)

	In = PlaceFields(number_cells = cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =res ,n_e =1,noise_levels = [0], normed = 0, sparsity = 0.033)

	ax = fig.add_subplot(111)
	loc = In.locations[In.store_indizes]
	patterns = In.input_stored
	cell = 0
	env = 0
	s = ax.scatter(loc[env][:,0][np.flatnonzero(patterns[env][:,cell] != 0)], loc[env][:,1][np.flatnonzero(patterns[env][:,cell] != 0)], c = patterns[env][:,cell][np.flatnonzero(patterns[env][:,cell] != 0)], faceted = False,cmap=cm.jet)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	fig.colorbar(s)


def HammingDistance(pattern_1, pattern_2, number_winner, num_seq, seq_len, len_noise_levels):
    pattern_1[pattern_1>0] = 1
    pattern_2[pattern_2>0] = 1
    f = pattern_1 + pattern_2
    f[f != 1] = 0.
    a = np.sum(f, axis=-1)
    c = a[0].reshape(len_noise_levels, num_seq, seq_len)
    return c

def GeneratePatternsOutofsomePatterns(input_patterns = None): 
    '''
    :input_patterns: input patterns with the shape of (n_e, num_patterns, num_cells)
    '''
    num_output_patterns = 9
    output_patterns = np.zeros([num_output_patterns, input_patterns.shape[-1]], 'bool')
    number_winner= int(2500 * 0.053)
    input_patterns = input_patterns.reshape(input_patterns.shape[1:])
    sum_patterns = np.zeros(input_patterns.shape[1])
    for i in range(input_patterns.shape[0]):
        sum_patterns = sum_patterns + input_patterns[i]
    sum_patterns[sum_patterns>1] = 1
    nonZeroIndexes = np.array(np.where(sum_patterns == 1))#.reshape(977)
    nonZeroIndexes = nonZeroIndexes.reshape(nonZeroIndexes.shape[1])
    #random.shuffle(nonZeroIndexes)
    for i in range(num_output_patterns):
        changed_indexes = random.sample(nonZeroIndexes,number_winner)
        output_patterns[i,changed_indexes] = 1
    return  output_patterns.reshape(1, num_output_patterns, 2500)   
    

###########################  Lyapunov Exponent ########################################
def RGSTP(initConnectionMethod= None, initMethod = None, in_weight= None, plastic= None):
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 25
    seq_len = 9
    no_realization = 1
    num_all_pat = (num_seq * seq_len)
    noise_levels=np.arange(0, 2501, 500)
    realization_matrix_Ca3 = np.zeros([len(noise_levels), seq_len, no_realization])
    HammingDistance_matrix_Ca3 = np.zeros([len(noise_levels), seq_len, no_realization])


    for realization in range(no_realization):

		centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.

		#centers = np.random.uniform(0,1, size=(1,2500,1,2))


		#store_indizes = np.arange(num_seq).reshape(1,num_seq)
		store_indizes = None
		Ca3 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = 2500, number_patterns = 400, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = np.arange(0, 2501, 500), centers = centers, sparsity = 0.053)

		#store_indizes = None
		#Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =num_seq, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers)

		#################################################################

		#Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA_Self_Inhibition, number_winner= int(2500 * 0.033), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsexponential, initConnectionMethod = Network.initConnectionMetric, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = Ca3.centers)

		#plastic
		Ca3_Ca3 = OneShootmehdi(input_cells = 2500, cells = 2500, number_winner = int(2500*0.053), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = initMethod, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = initConnectionMethod, active_in_env = 2500, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers)

		#static
		#Ca3_Ca3 = OneShootmehdi(input_cells = 2500, cells = 2500, number_winner = int(2500*0.053), connectivity = 0.999, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = Network.makeWeightsexponential, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionMetric, active_in_env = 2500, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = Ca3.centers)



		#################################################################

		ca3_input = np.zeros([1, num_all_pat, 2500])
		for j in range(num_seq):
		    inhibition = np.copy(Ca3.input_stored[0][j]) # it works just for one sequence!!!!!!!!!!!!!!!!!!!!!!
		    self_inhibition = np.ones(Ca3.input_stored[0][j].shape)


		    #inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
		    #self_inhibition *= abs(inhibition - 1.)

		    inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
		    inhibition *= in_weight
		    inhibition[inhibition==0.] = 1.
		    self_inhibition *= inhibition


		    ca3_input[0][j*seq_len] = Ca3.input_stored[0][j]
		    for i in range(1, seq_len):

				ca3_input[0][i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca3_input[0][i+(j*seq_len)-1], self_inhibition=self_inhibition)


				inhibition = np.copy(ca3_input[0][i+(j*seq_len)]) # it works just for one sequence!!!!!!!!!!!!!!!!!!!!!!

				#inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
				#self_inhibition *= abs(inhibition - 1.)

				inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
				inhibition *= in_weight
				inhibition[inhibition==0.] = 1.
				self_inhibition *= inhibition


		if plastic:
			Ca3_Ca3.learnOneShootAllPattern(input_pattern = ca3_input, method = OneShootmehdi.learnFiringDependent, seq_len = seq_len, num_seq= num_seq)

		#################################################################

		ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, 2500])

		for j in range(num_seq):
		    inhibition = np.copy(Ca3.noisy_input_stored[:, :,j])#
		    self_inhibition = np.ones(Ca3.noisy_input_stored[:, :,j].shape) #

		    #inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
		    #self_inhibition *= abs(inhibition - 1.)

		    inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
		    inhibition *= in_weight
		    inhibition[inhibition==0.] = 1.
		    self_inhibition *= inhibition


		    ca1_noisy_input_pattern[:,:,j*seq_len] = Ca3.noisy_input_stored[:, :,j]#
		    for i in range(1, seq_len):
				ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1], self_inhibition=self_inhibition)

				inhibition = np.copy(ca1_noisy_input_pattern[:, :,i+(j*seq_len)]) #

				#inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
				#self_inhibition *= abs(inhibition - 1.)

				inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
				inhibition *= in_weight
				inhibition[inhibition==0.] = 1.
				self_inhibition *= inhibition

		#########################################################################

		ca3_input_output_correlation = Corelations(patterns_1=ca3_input, patterns_2=ca1_noisy_input_pattern, in_columns = False)
		cor = ca3_input_output_correlation.getCorOrigOrig().reshape(len(Ca3.noise_levels), num_seq, seq_len)
		HammingDistance1 = HammingDistance(pattern_1 = ca3_input, pattern_2= ca1_noisy_input_pattern, number_winner = Ca3_Ca3.number_winner, num_seq= num_seq, seq_len= seq_len, len_noise_levels= len(Ca3.noise_levels))



		#for i in range(seq_len):
			#for j in range(len(Ca3.noise_levels)):
				#realization_matrix_Ca3[i, j, realization] = np.mean(cor[j, : , i])
				#HammingDistance_matrix_Ca3[i, j, realization] = np.mean(HammingDistance1[j, : , i])
		
		for i in range(len(noise_levels)):	
			for j in range(seq_len):
				realization_matrix_Ca3[i, j, realization] = np.mean(cor[i, : , j])
								
				
		print realization		

	#fig1 = plt.figure(1)
	#ax = fig1.add_subplot(121)
	#s = ax.imshow(np.mean(realization_matrix_Ec, axis=-1)[::-1, ::-1], extent=[0,1,1,20] ,vmin = 0, vmax = 1, aspect='auto')
	#fig1.colorbar(s)
	#plt.xlabel('Cue strength', fontsize=20)
	#plt.ylabel('patern_index', fontsize=20)
	#plt.legend()
	#plt.title("Ec_EC-mean, initialized in Ca3", fontsize=20)
	#ax = fig1.add_subplot(122)
	#p = ax.imshow(np.std(realization_matrix_Ec, axis=-1)[::-1, ::-1],extent=[0,1,1,20], aspect='auto')
	#fig1.colorbar(p)
	#plt.xlabel('Cue strength', fontsize=20)
	##plt.ylabel('patern_index', fontsize=20)	
	#plt.legend()
	#plt.title("Ec_Ec-std, initialized in Ca3", fontsize=20)
    
    
    '''
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(211)
    data = np.mean(realization_matrix_Ca3, axis=-1)[::-1, ::-1]
    data = np.transpose(data)
    data = data[::-1]
    data = np.transpose(data)
    s = ax.imshow(data , origin='lower', extent=[1,seq_len,0,1] ,vmin = 0, vmax = 1, aspect='auto')
    fig2.colorbar(s)
    plt.xlabel('Position in the sequence', fontsize=20)
    plt.ylabel('Cue quality', fontsize=20)
    plt.legend()
    plt.title("Local connectivity, Mean correlation, iteration = 30", fontsize=20)
    ax = fig2.add_subplot(212)
    data = np.std(realization_matrix_Ca3, axis=-1)[::-1, ::-1]
    data = np.transpose(data)
    data = data[::-1]
    data = np.transpose(data)
    p = ax.imshow(data, origin='lower', extent=[1,seq_len,0,1], aspect='auto')
    fig2.colorbar(p)
    plt.xlabel( 'Position in the sequence', fontsize=20)
    plt.ylabel('Cue quality', fontsize=20)
    plt.legend()
    plt.title("Standard deviation", fontsize=20)
    '''
    
    
    
 

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.title('Time (pattern unit)')
    
    def ca3_anim(j):
		return ca1_noisy_input_pattern[0,0,j]
		
    im = ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]), edgecolors ='none', cmap = 'YlOrRd' )
    j=0
    def updatefig(*args):
		global j
		im.set_array(ca3_anim(j))
		j += 1
		return im,
		
    ani = animation.FuncAnimation(fig, updatefig, interval=1000)
    #ani.save('fghj.mp4', fps=30, codec='libx264')
    '''

 
    fig1 = plt.figure(1)
    for j in range(len(Ca3.noise_levels)):
		ax = fig1.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, j+1)
		ax.plot(np.array(range(seq_len)), np.mean(realization_matrix_Ca3, axis=-1)[j,:])
		#print(cor[j][i])
		ax.set_ylim(-0.05, 1.05)
		#plt.legend()
		#plt.title("Ca3_Ca3, C = %.3f" % (Ca3.getOrigVsOrig()[j]))
		ax.text(.7,.6,"CS = %.3f" % (Ca3.getOrigVsOrig()[j]), horizontalalignment='center', transform=ax.transAxes)
    plt.suptitle(r'$\alpha =$' " %.2f, plasticity = %d " % (in_weight, plastic))
    # Set common labels
    fig1.text(0.5, 0.04, 'Pattern position in Seq.', ha='center', va='center')
    fig1.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')

    #fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


        
    colors = cm.rainbow(np.linspace(0, 1, len(Ca3.noise_levels)))
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    y1 =  cor
    for i in range(len(Ca3.noise_levels)):
		for j in range(num_seq):
			x1 = y1[i, j, :-1]
			x2 = y1[i, j, 1:]
			ax.plot(x1, x2, 'o',color= colors[i])#, markersize= (len(Ca3.noise_levels) - i))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    x1 =  np.arange(-0.1,1.1,.1)
    ax.plot(x1, x1)
    plt.xlabel(r'$<Corr_t>$')
    plt.ylabel(r'$<Corr_{t+1}>$')
    plt.suptitle(r'$\alpha =$' " %.2f, plasticity = %d " % (in_weight, plastic))
    
    fig3 = plt.figure(3)
    for j in range(len(Ca3.noise_levels)):
        ax = fig3.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, cor[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.05, 1.05)
        plt.legend()
        ax.text(.7,.6,"CS = %.3f" % (Ca3.getOrigVsOrig()[j]), horizontalalignment='center', transform=ax.transAxes)

    plt.suptitle(r'$\alpha =$' " %.2f, plasticity = %d " % ('1 - in_weight', plastic))
    # Set common labels
    fig3.text(0.5, 0.04, 'Pattern position in Seq.', ha='center', va='center')
    fig3.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')
       
    #fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

 


    
    fig4 = plt.figure(4)
    for j in range(seq_len):
        ax = fig4.add_subplot(int(seq_len/4.)+1, 4, j+1)
        #ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]*2 + ca1_noisy_input_pattern[0,10,j]), edgecolors ='none', cmap = 'YlOrRd')
        ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]), edgecolors ='none', cmap = 'YlOrRd' )
        ax.text(.5,.9,"t = %d" % (j+1), horizontalalignment='center', transform=ax.transAxes)
        plt.legend()
        #plt.title("Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
    plt.suptitle(r'$\alpha =$' " %.2f, plasticity = %d " % (1 - in_weight, plastic))    
    
    
    fig5 = plt.figure(5)
    ax = fig5.add_subplot(1,1,1)
    #ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]*2 + ca1_noisy_input_pattern[0,10,j]), edgecolors ='none', cmap = 'YlOrRd')
    ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,0]), edgecolors ='none', cmap = 'cool' )
    ax.text(.5,.9,"t = %d" % (j+1), horizontalalignment='center', transform=ax.transAxes)
    plt.legend()
    #plt.title("Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
    plt.suptitle(r'$\alpha =$' " %.2f, plasticity = %d " % (1 - in_weight, plastic))     

 
    

    '''
    colors = cm.rainbow(np.linspace(0, 1, len(Ca3.noise_levels)))
    fig4 = plt.figure(4)
    #ax = fig4.add_subplot(111)
    #y1 =  np.mean(realization_matrix_Ca3, axis=-1)
    y1 =  np.mean(HammingDistance_matrix_Ca3, axis=-1)
    x1 =  np.arange(0,120.1,1)
    for i in range(len(Ca3.noise_levels)):
        ax = fig4.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, i+1)
        x1 = y1[:-1, i]
        x2 = y1[1:, i]
        ax.plot(x1, x2, '*', color= colors[i])
        ax.plot(x1, x1)
        ax.set_xlim(0., 120)
        ax.set_ylim(0., 120)

    ##plt.xlabel('Pattern position in Seq.')
    ##plt.ylabel('Lyapunov Exponent')


    colors = cm.rainbow(np.linspace(0, 1, len(Ca3.noise_levels)))
    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    #y1 =  np.mean(realization_matrix_Ca3, axis=-1)
    y1 =  np.mean(HammingDistance_matrix_Ca3, axis=-1)
    for i in range(len(Ca3.noise_levels)):
        #ax = fig4.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, i+1)
        x1 = y1[:-1, i]
        x2 = y1[1:, i]
        ax.plot(x1, x2, 'o',color= colors[i])#, markersize= (len(Ca3.noise_levels) - i))
        ax.set_xlim(0., 120)
        ax.set_ylim(0., 120)
    x1 =  np.arange(0,120.1,1)
    ax.plot(x1, x1)
    ##plt.xlabel('Pattern position in Seq.')
    ##plt.ylabel('Lyapunov Exponent')
    
    
    
    fig7 = plt.figure(7)
    for j in range(4):
        #ax = fig7.add_subplot(int(seq_len/4.)+1, 4, j+1)
        ax = fig7.add_subplot(2, 2, j+1)
        #ax.imshow(ca1_noisy_input_pattern.reshape(1,len(Ec.noise_levels),num_all_pat,50,50)[0][0][j], label = str(j))
        ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]*2 + ca1_noisy_input_pattern[0,2,j]), edgecolors ='none', cmap = 'YlOrRd')
        plt.legend()
        #plt.title("Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))

    fig8 = plt.figure(8)
    for j in range(4):
        #ax = fig8.add_subplot(int(seq_len/4.)+1, 4, j+1)
        ax = fig8.add_subplot(2, 2, j+1)
        #ax.imshow(ca1_noisy_input_pattern.reshape(1,len(Ec.noise_levels),num_all_pat,50,50)[0][0][j], label = str(j))
        ax.scatter(Ca3.centers[0,:,0,0], Ca3.centers[0,:,0,1], c =(ca1_noisy_input_pattern[0,0,j]*2 + ca1_noisy_input_pattern[0,3,j]), edgecolors ='none', cmap = 'YlOrRd')
        plt.legend()
        #plt.title("Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
    '''

    
###################################################################
#Just to replicate Torsten's results (1)
def WithoutCA3recurrents():
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 250
    seq_len = 1
    num_all_pat = (num_seq * seq_len)
    in_weight = 0.185
    noise_levels= np.arange(0,1101,700)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTALinear, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    #Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = store_indizes)

    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    #centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    #Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = Network.makeWeightsOne, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers  to ca3. ... !!!!!!!!!
    
    Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = True, actFunction = AutoAssociation.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = centers)

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 0, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = Ca3.input_stored)

    #1ca1_input = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern= ca3_input)#Ec_Ca3.output_stored)    ####??????????????????????????????????
    #######
    ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored)
    ##########

########
    #Ca3_Ca1.learnOneShootAllPattern(input_pattern = Ec_Ca3.output_stored, method = OneShoot.learnActivityDependent)
##########
    Ca3_Ca1.learnAssociation(input_pattern = Ca3.input_stored , output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)
    

    #2Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)
    #Ca1_Ec.learnAssociation(input_pattern = Ca3_Ca1.output_stored , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    
    Ca3_Ca1.recall(input_pattern = Ec_Ca3.noisy_output)
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)


################################### Without CA3 recurrent collaterals ###############

    ec_input_output_correlation = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    ca3_input_output_correlation = Ec_Ca3.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    ca1_input_output_correlation = Ca3_Ca1.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(3,1,1)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ca3_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{CA3}')
    plt.title('Without CA3 recurrent collaterals')

    
    ax = fig1.add_subplot(3,1,2)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ca1_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{CA1}')
    
    ax = fig1.add_subplot(3,1,3)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ec_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{Ec}')    
    
    plt.legend()

########################################################################

def WithCA3recurrents():
#Just to replicate Torsten's results (1) No I changed the connectivity karnel

#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 250
    seq_len = 1
    num_all_pat = (num_seq * seq_len)
    in_weight = 0.185
    noise_levels= np.arange(0,1101,50)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTALinear, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    #Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = store_indizes)

    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    #centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    #Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = Network.makeWeightsOne, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers to ca3. ... !!!!!!!!!
    
    Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = True, actFunction = AutoAssociation.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionNEST, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = centers, cycles = 15, external_force = 0, internal_force = 1)# ycles = 15, external_force = 0, internal_force = 1

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 0, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = Ca3.input_stored)
    
    Ca3_Ca3.learnAssociation(input_pattern = Ca3.input_stored, output_pattern = Ca3.input_stored)
    
    #######
    ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored)
##########
    Ca3_Ca1.learnAssociation(input_pattern = Ca3.input_stored , output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    
    Ca3_Ca3.recall(input_pattern = Ec_Ca3.noisy_output, external_activity = Ec_Ca3.calcActivity(input_pattern = Ec.noisy_input_stored))    
    
    Ca3_Ca1.recall(input_pattern = Ca3_Ca3.noisy_output)
    
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)


################################### With CA3 recurrent collaterals ###############

    ec_input_output_correlation = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    #ca3_input_output_correlation = Ec_Ca3.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    ca3_recurrent_input_output_correlation = Ca3_Ca3.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)    
    ca1_input_output_correlation = Ca3_Ca1.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(3,1,1)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ca3_recurrent_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{CA3}')
    plt.title('With CA3 recurrent collaterals, LCR = 0.25')

    
    ax = fig1.add_subplot(3,1,2)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ca1_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{CA1}')
    
    ax = fig1.add_subplot(3,1,3)
    ax.plot(Ec.getOrigVsOrig(), np.mean(ec_input_output_correlation, axis=1))
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Cue quality')
    plt.ylabel('Corr_{Ec}')    
    
    plt.legend()

########################################################################

def HetteroAssociatedSequenceLerninginCa3(): # when Ec.cells == Ca3.cells >> I am testing for the same weight matrics in Feed forward and recurrent colaterals!!!!
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 15
    seq_len = 15
    num_all_pat = (num_seq * seq_len)
    in_weight = 0.185
    noise_levels= np.arange(0,1101,700)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTALinear, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    #Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = store_indizes)

    #centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.0353, store_indizes = store_indizes) 

#################################################################

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    #Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = Network.makeWeightsOne, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers to ca3. ... !!!!!!!!!
    
    #Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = True, actFunction = AutoAssociation.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = centers, cycles = 15, external_force = 0, internal_force = 1)# ycles = 15, external_force = 0, internal_force = 1
    
    Ca3_Ca3 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 0, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = Ca3.input_stored)
    
    for j in range(num_seq):
        Ca3_Ca3.learnAssociation(input_pattern = Ca3.input_stored[0, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = Ca3.input_stored[0, (j*seq_len)+1:(j*seq_len)+seq_len])# is it possible? just ask wheather weights are set to zero everytime!
        #print (j*seq_len), (j*seq_len)+seq_len-1, (j*seq_len)+1, (j*seq_len)+seq_len
    #######
    ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored)
##########
    Ca3_Ca1.learnAssociation(input_pattern = Ca3.input_stored , output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    
    ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, 2500])
    for j in range(num_seq):
        ca1_noisy_input_pattern[:,:,j*seq_len] = Ec_Ca3.noisy_output[:, :,j*seq_len]
        for i in range(1,seq_len):
            ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1])
	    #print j*seq_len, i+(j*seq_len), i+(j*seq_len)-1   
    
    Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
    
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)


################################### Seq. Learning in Ca3 ###############


    noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=Ca3.input_stored, patterns_2=ca1_noisy_input_pattern, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
########################################################################

    ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2=Ca3_Ca1.noisy_output, in_columns = False)
    corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig20 = plt.figure(20)
    for j in range(len(Ec.noise_levels)):
        ax = fig20.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca1[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca1_Ca1, C = %.3f" % (Ec.getOrigVsOrig()[j]))

#########################################################################

def Ec_Ca3_Ca1_Ec(): ## 
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 16
    seq_len = 16
    num_all_pat = (num_seq * seq_len)
    in_weight = 0.
    noise_levels= np.arange(0,1101,500)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = None)

    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    #centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = True, initMethod = Network.makeWeightsOne, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionOfflineStructured, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers  to ca3. ... !!!!!!!!!

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    
#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################

    ca3_input = np.zeros([1, num_all_pat, Ca3.cells])
    for j in range(num_seq):
        inhibition = np.copy(Ca3.input_stored[0, j]) # it works just for one sequence!!!!!!!!!!!!!!!!!!!!!!
        self_inhibition = np.ones(Ca3.input_stored[0, j].shape)


        inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
        inhibition *= in_weight
        inhibition[inhibition==0.] = 1.
        self_inhibition *= inhibition


        ca3_input[0][j*seq_len] = Ca3.input_stored[0, j]
        for i in range(1, seq_len):

            ca3_input[0, i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca3_input[0, i+(j*seq_len)-1], self_inhibition=self_inhibition)


            inhibition = np.copy(ca3_input[0, i+(j*seq_len)]) # it works just for one sequence!!!!!!!!!!!!!!!!!!!!!!


            inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
            inhibition *= in_weight
            inhibition[inhibition==0.] = 1.
            self_inhibition *= inhibition
            
    if plastic:
        Ca3_Ca3.learnOneShootAllPattern(input_pattern = ca3_input, method = OneShootmehdi.learnFiringDependent, seq_len = seq_len, num_seq= num_seq)        
            
#################################################################   

    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = ca3_input)
    #######
    ca1_input =  Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored) ##  Ca1.input_stored ## 
    
##########
    Ca3_Ca1.learnAssociation(input_pattern = ca3_input , output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    
##################################################################

    ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, Ca3.cells])

    for j in range(num_seq):
        inhibition = np.copy(Ec_Ca3.noisy_output[:, :,(j*seq_len)])#
        self_inhibition = np.ones(Ec_Ca3.noisy_output[:, :,(j*seq_len)].shape) #


        inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
        inhibition *= in_weight
        inhibition[inhibition==0.] = 1.
        self_inhibition *= inhibition


        ca1_noisy_input_pattern[:,:,j*seq_len] = Ec_Ca3.noisy_output[ :, :,(j*seq_len)]#
        for i in range(1, seq_len):
            ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1], self_inhibition=self_inhibition)

            inhibition = np.copy(ca1_noisy_input_pattern[:, :,i+(j*seq_len)]) #

            inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
            inhibition *= in_weight
            inhibition[inhibition==0.] = 1.
            self_inhibition *= inhibition
    
    Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
    
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)


################################### Seq. Learning in Ca3 ###############


    noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=ca3_input, patterns_2=ca1_noisy_input_pattern, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
########################################################################

    ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2=Ca3_Ca1.noisy_output, in_columns = False)
    corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig20 = plt.figure(20)
    for j in range(len(Ec.noise_levels)):
        ax = fig20.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca1[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca1_Ca1, C = %.3f" % (Ec.getOrigVsOrig()[j]))


def Ec_Dg_Ca3_Ca1_Ec(): ## it depends how patterns in ca1 and dg are triggered. 
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 16
    seq_len = 16
    num_all_pat = (num_seq * seq_len)

    in_weight = 0.
    noise_levels= np.arange(0,1101,500)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    DgCells = 12000
    
    Dg = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = DgCells, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.005, store_indizes = None)
    
    Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = store_indizes)

    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    #centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################
    
    Ec_Dg = HeteroAssociation(input_cells=Ec.cells, cells= DgCells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(DgCells * 0.053), e_max = 0.1, active_in_env = DgCells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    
    
    Dg_Ca3 = HeteroAssociation(input_cells=DgCells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = True, initMethod = Network.makeWeightsOne, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers  to ca3. ... !!!!!!!!!

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    
    Ec_Dg_trigered = HeteroAssociation(input_cells=Ec.cells, cells= DgCells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(DgCells * 0.053), e_max = 0.1, active_in_env = DgCells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    

#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    ca1_input = Ca1.input_stored # Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored) ## 
    dg_input = Dg.input_stored #Ec_Dg_trigered.getOutput(Ec_Dg_trigered, input_pattern = Ec.input_stored)

#################################################################

    ca3_input = np.zeros([1, num_all_pat, Ca3.cells])
    for j in range(num_seq):
        inhibition =  np.copy(Ca3.input_stored[0, j]) #
        self_inhibition = np.ones(Ca3.input_stored[0, j].shape)


        inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
        inhibition *= in_weight
        inhibition[inhibition==0.] = 1.
        self_inhibition *= inhibition


        ca3_input[0][j*seq_len] = Ca3.input_stored[0, j]
        for i in range(1, seq_len):

            ca3_input[0, i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca3_input[0, i+(j*seq_len)-1], self_inhibition=self_inhibition)


            inhibition = np.copy(ca3_input[0, i+(j*seq_len)]) # 
            

            inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
            inhibition *= in_weight
            inhibition[inhibition==0.] = 1.
            self_inhibition *= inhibition
            
    if plastic:
        Ca3_Ca3.learnOneShootAllPattern(input_pattern = ca3_input, method = OneShootmehdi.learnFiringDependent, seq_len = seq_len, num_seq= num_seq)        
            
    
##########
    Ec_Dg.learnAssociation(input_pattern = Ec.input_stored , output_pattern = dg_input)
    Dg_Ca3.learnAssociation(input_pattern = dg_input , output_pattern = ca3_input) 
    Ca3_Ca1.learnAssociation(input_pattern = ca3_input, output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Dg.recall(input_pattern = Ec.noisy_input_stored)
    Dg_Ca3.recall(input_pattern = Ec_Dg.noisy_output)
    
##################################################################

    ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, Ca3.cells])

    for j in range(num_seq):
        inhibition = np.copy(Dg_Ca3.noisy_output[:, :,(j*seq_len)])#
        self_inhibition = np.ones(Dg_Ca3.noisy_output[:, :,(j*seq_len)].shape) #


        inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
        inhibition *= in_weight
        inhibition[inhibition==0.] = 1.
        self_inhibition *= inhibition


        ca1_noisy_input_pattern[:,:,j*seq_len] = Dg_Ca3.noisy_output[ :, :,(j*seq_len)]#
        for i in range(1, seq_len):
            ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1], self_inhibition=self_inhibition)

            inhibition = np.copy(ca1_noisy_input_pattern[:, :,i+(j*seq_len)]) #

            inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
            inhibition *= in_weight
            inhibition[inhibition==0.] = 1.
            self_inhibition *= inhibition
            
    
    Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
    
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)


################################### ###############


    noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=ca3_input, patterns_2=ca1_noisy_input_pattern, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
########################################################################

    ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2=Ca3_Ca1.noisy_output, in_columns = False)
    corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig3 = plt.figure(3)
    for j in range(len(Ec.noise_levels)):
        ax = fig3.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca1[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca1_Ca1, C = %.3f" % (Ec.getOrigVsOrig()[j]))
        
########################################################################

    dg_input_output_correlation = Corelations(patterns_1=dg_input, patterns_2=Ec_Dg.noisy_output, in_columns = False)
    cordg = dg_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig4 = plt.figure(4)
    for j in range(len(Ec.noise_levels)):
        ax = fig4.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, cordg[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Dg_Dg, C = %.3f" % (Ec.getOrigVsOrig()[j]))
        
    
def Ec_Ca3_test1(): ## Ec_Ca3, Ca3 inputs are trigerred by Ec
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 16
    seq_len = 16
    num_all_pat = (num_seq * seq_len)

    in_weight = 0.
    noise_levels= np.arange(0,1101,500)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?


    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.

    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################
    
    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells= Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * 0.053), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

############
    
    Ec_Ca3_trigered = HeteroAssociation(input_cells=Ec.cells, cells= Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * 0.053), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    

#######

    Ca3_Ec = HeteroAssociation(input_cells=Ca3.cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    ca3_input = Ec_Ca3_trigered.getOutput(Ec_Ca3_trigered, input_pattern = Ec.input_stored)



##########
    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = ca3_input)

    Ca3_Ec.learnAssociation(input_pattern = ca3_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    Ca3_Ec.recall(input_pattern = Ec_Ca3.noisy_output)


################################### ###############


    noisy_cue_allcorrelations = Ca3_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=ca3_input, patterns_2=Ec_Ca3.noisy_output, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
################################### ###############

def Ec_Ca3_test2(): ## Ec-Ca3, Ca3 inputs are intrinsic
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 16
    seq_len = 16
    num_all_pat = (num_seq * seq_len)
    in_weight = 0.
    noise_levels= np.arange(0,1101,300)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?


    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.

    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################
    
    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells= Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * 0.053), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    

#######

    Ca3_Ec = HeteroAssociation(input_cells=Ca3.cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)



##########
    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = Ca3.input_stored)

    Ca3_Ec.learnAssociation(input_pattern = Ca3.input_stored , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    Ca3_Ec.recall(input_pattern = Ec_Ca3.noisy_output)


################################### ###############


    noisy_cue_allcorrelations = Ca3_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=Ca3.input_stored, patterns_2=Ec_Ca3.noisy_output, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
################################### ###############
    
def OneshotSequenceLerninginCa3():
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 15
    seq_len = 15
    num_all_pat = (num_seq * seq_len)

    in_weight = 0.185
    noise_levels= np.arange(0,1101,700)
    plastic= 0
##################################################################


    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories
    
    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None    
#################################################################

    Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTALinear, number_patterns = 400, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
    
    #Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
    
    Ca1Cells = 4200
    #Ca1 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = Ca1Cells, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = [0], sparsity = 0.097, store_indizes = store_indizes)

    #centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputWTALinear, number_patterns =num_all_pat, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = 400, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseZero, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes) 

#################################################################

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.632, learnrate = 1., subtract_input_mean = True, subtract_output_mean = True, initMethod = Network.makeWeightsZero, actFunction = Network.getOutputWTA, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = centers) # change centers to ca3. ... !!!!!!!!!
    
    #Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = True, actFunction = AutoAssociation.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = centers, cycles = 15, external_force = 0, internal_force = 1)# ycles = 15, external_force = 0, internal_force = 1
    
    #Ca3_Ca3 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097

#######
    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 0, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsOne, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
#######

    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

#################################################################   

    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = Ca3.input_stored)
    
    Ca3_Ca3.learnOneShootAllPattern(input_pattern = Ca3.input_stored, method = OneShootmehdi.learnFiringDependent, seq_len = seq_len, num_seq= num_seq)
    
    #######
    ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored)
##########
    Ca3_Ca1.learnAssociation(input_pattern = Ca3.input_stored , output_pattern = ca1_input)
    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)

##################################################################
    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
    
    ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, 2500])
    for j in range(num_seq):
        ca1_noisy_input_pattern[:,:,j*seq_len] = Ec_Ca3.noisy_output[:, :,j*seq_len]
        for i in range(1,seq_len):
            ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1])
	    #print j*seq_len, i+(j*seq_len), i+(j*seq_len)-1   
    
    
    Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
    
    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)
    


################################### Seq. Learning in Ca3 ###############


    noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)

    fig1 = plt.figure(1)
    for j in range(len(Ec.noise_levels)):
        ax = fig1.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, noisy_cue_allcorrelations[j][i])#, label = str(i))

        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ec_Ec-Noise level = %.3f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ec_Ec, C = %.3f" % (Ec.getOrigVsOrig()[j]))

########################################################################

    ca3_input_output_corr = Corelations(patterns_1=Ca3.input_stored, patterns_2=ca1_noisy_input_pattern, in_columns = False)
    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig2 = plt.figure(2)
    for j in range(len(Ec.noise_levels)):
        ax = fig2.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca3[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))
########################################################################

    ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2= Ca3_Ca1.noisy_output, in_columns = False)
    corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
    fig20 = plt.figure(20)
    for j in range(len(Ec.noise_levels)):
        ax = fig20.add_subplot(int(len(Ec.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, corca1[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        #plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
        plt.title("Ca1_Ca1, C = %.3f" % (Ec.getOrigVsOrig()[j]))


###########################  Lyapunov Exponent ########################################
def SequenceLerning():
#if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    num_seq = 9
    seq_len = 9
    no_realization = 1
    num_all_pat = (num_seq * seq_len)
    noise_levels=np.arange(0, 2501, 250)
    realization_matrix_Ca3 = np.zeros([len(noise_levels),seq_len, no_realization])
    HammingDistance_matrix_Ca3 = np.zeros([len(noise_levels), seq_len, no_realization])

    for realization in range(no_realization):

		#centers = np.random.uniform(0,1, size=(1,2500,1,2))
		#Ca3 = Input(inputMethod = Input.makeInputNormalDistributed, number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.053)




		#centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.

		centers = np.random.uniform(0,1, size=(1,2500,1,2))

		store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)

		#store_indizes = None
		Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =num_all_pat, number_to_store = num_all_pat, store_indizes = store_indizes, n_e =1, noise_levels= noise_levels, normed= 0, sparsity = 0.053, centers = centers)

		#################################################################

		Ca3_Ca3 = AutoAssociation(input_cells=2500, cells=2500, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA, number_winner= int(2500 * 0.053), active_in_env = 2500, n_e = 1, initMethod = Network.makeWeightsexponential, initConnectionMethod = Network.initConnectionMetric, weight_sparsity = None, weight_mean = 1, weight_sigma = .5, centers = centers)


		#Ca3_Ca3 = OneShootmehdi(input_cells = 2500, cells = 2500, number_winner = int(2500*0.053), connectivity = 1., learnrate = 0.05, subtract_input_mean = True, subtract_output_mean = False, initMethod = Network.makeWeightsZero, actFunction = Network.getOutputWTA, initConnectionMethod = Network.initConnectionRandom, active_in_env = 2500, n_e = 1, weight_mean = 1, weight_sigma = 0.5, centers = Ca3.centers)

		#################################################################

		for j in range(num_seq):
			Ca3_Ca3.learnAssociation(input_pattern = Ca3.input_stored[0, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = Ca3.input_stored[0, (j*seq_len)+1:(j*seq_len)+seq_len])
		#Ca3_Ca3.learnAssociation(input_pattern = Ca3.input_stored[0, :-1], output_pattern = Ca3.input_stored[0, 1:])




		#################################################################

		ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, 2500])

		for j in range(num_seq):
		    ca1_noisy_input_pattern[:,:,j*seq_len] = Ca3.noisy_input_stored[:, :,j*seq_len]
		    for i in range(1,seq_len):
		        ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1])

		#########################################################################

		ca3_input_output_correlation = Corelations(patterns_1=Ca3.input_stored, patterns_2=ca1_noisy_input_pattern, in_columns = False)
		cor = ca3_input_output_correlation.getCorOrigOrig().reshape(len(Ca3.noise_levels), num_seq, seq_len)
		HammingDistance1 = HammingDistance(pattern_1 = Ca3.input_stored, pattern_2= ca1_noisy_input_pattern, number_winner = Ca3_Ca3.number_winner, num_seq= num_seq, seq_len= seq_len, len_noise_levels= len(Ca3.noise_levels))



		#for i in range(seq_len):
		#	for j in range(len(Ca3.noise_levels)):
		#		realization_matrix_Ca3[i, j, realization] = np.mean(cor[j, : , i])
		#		HammingDistance_matrix_Ca3[i, j, realization] = np.mean(HammingDistance1[j, : , i])


                for i in range(len(noise_levels)):
			for j in range(seq_len):
				realization_matrix_Ec[i, j, realization] = np.mean(noisy_cue_allcorrelations[i, : , j])
		
		for i in range(len(noise_levels)):	
			for j in range(seq_len):
				realization_matrix_Ca3[i, j, realization] = np.mean(cor[i, : , j])
								
				
		print realization		

	#fig1 = plt.figure(1)
	#ax = fig1.add_subplot(121)
	#s = ax.imshow(np.mean(realization_matrix_Ec, axis=-1)[::-1, ::-1], extent=[0,1,1,20] ,vmin = 0, vmax = 1, aspect='auto')
	#fig1.colorbar(s)
	#plt.xlabel('Cue strength', fontsize=20)
	#plt.ylabel('patern_index', fontsize=20)
	#plt.legend()
	#plt.title("Ec_EC-mean, initialized in Ca3", fontsize=20)
	#ax = fig1.add_subplot(122)
	#p = ax.imshow(np.std(realization_matrix_Ec, axis=-1)[::-1, ::-1],extent=[0,1,1,20], aspect='auto')
	#fig1.colorbar(p)
	#plt.xlabel('Cue strength', fontsize=20)
	##plt.ylabel('patern_index', fontsize=20)	
	#plt.legend()
	#plt.title("Ec_Ec-std, initialized in Ca3", fontsize=20)
	
    
    
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(211)
    data = np.mean(realization_matrix_Ca3, axis=-1)[::-1, ::-1]
    data = np.transpose(data)
    data = data[::-1]
    data = np.transpose(data)
    s = ax.imshow(data , origin='lower', extent=[1,20,0,1] ,vmin = 0, vmax = 1, aspect='auto')
    fig2.colorbar(s)
    plt.xlabel('Position in the sequence', fontsize=20)
    plt.ylabel('Cue quality', fontsize=20)	
    plt.legend()
    plt.title("Local connectivity, Mean correlation, iteration = 30", fontsize=20)	
    ax = fig2.add_subplot(212)
    data = np.std(realization_matrix_Ca3, axis=-1)[::-1, ::-1]
    data = np.transpose(data)
    data = data[::-1]
    data = np.transpose(data)
    p = ax.imshow(data, origin='lower', extent=[1,20,0,1], aspect='auto')
    fig2.colorbar(p)		
    plt.xlabel( 'Position in the sequence', fontsize=20)
    plt.ylabel('Cue quality', fontsize=20)	
    plt.legend()
    plt.title("Standard deviation", fontsize=20)






    fig1 = plt.figure(1)
    for j in range(len(Ca3.noise_levels)):
        ax = fig1.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, j+1)
        #ax.plot(range(len(Ca3.noise_levels)), realization_matrix_Ca3[j, :, 0])
        ax.plot(np.array(range(seq_len)), np.mean(realization_matrix_Ca3, axis=-1)[:, j])
	    #print(cor[j][i])
	ax.set_ylim(-0.1, 1.1)
	#plt.legend()
	#plt.title("Ca3_Ca3-Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))
	#plt.title("Ca3_Ca3, C = %.3f" % (Ca3.getOrigVsOrig()[j]))



    colors = cm.rainbow(np.linspace(0, 1, len(Ca3.noise_levels)))
    fig4 = plt.figure(4)
    #ax = fig4.add_subplot(111)
    #y1 =  np.mean(realization_matrix_Ca3, axis=-1)
    y1 =  np.mean(HammingDistance_matrix_Ca3, axis=-1)
    x1 =  np.arange(0,120.1,1)
    for i in range(len(Ca3.noise_levels)):
        ax = fig4.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, i+1)
        x1 = y1[:-1, i]
        x2 = y1[1:, i]
        ax.plot(x1, x2, '*', color= colors[i])
        ax.plot(x1, x1)
        ax.set_xlim(0., 120)
        ax.set_ylim(0., 120)

    ##plt.xlabel('Pattern position in Seq.')
    ##plt.ylabel('Lyapunov Exponent')

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    #y1 =  np.mean(realization_matrix_Ca3, axis=-1)
    y1 =  np.mean(HammingDistance_matrix_Ca3, axis=-1)
    for i in range(len(Ca3.noise_levels)):
        #ax = fig4.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, i+1)
        x1 = y1[:-1, i]
        x2 = y1[1:, i]
        ax.plot(x1, x2, 'o',color= colors[i])#, markersize= (len(Ca3.noise_levels) - i))
        ax.set_xlim(0., 120)
        ax.set_ylim(0., 120)
    x1 =  np.arange(0,120.1,1)
    ax.plot(x1, x1)
    ##plt.xlabel('Pattern position in Seq.')
    ##plt.ylabel('Lyapunov Exponent')


    fig6 = plt.figure(6)
    for j in range(len(Ca3.noise_levels)):
        ax = fig6.add_subplot(int(len(Ca3.noise_levels)/2)+1, 2, j+1)
        for i in range(num_seq):
            ax.plot(np.array(range(seq_len))+1, cor[j][i])#, label = str(i))
            #print(cor[j][i])
        ax.set_ylim(-0.1, 1.1)
        plt.legend()
        plt.title("Ca3_Ca3-Noise level = %.1f" % (Ca3.noise_levels[j]/Ca3.cells))
        #plt.title("Ca3_Ca3, C = %.3f" % (Ec.getOrigVsOrig()[j]))

    fig7 = plt.figure(7)
    for j in range(seq_len):
        ax = fig7.add_subplot(int(seq_len/4.)+1, 4, j+1)
        #ax.imshow(ca1_noisy_input_pattern.reshape(1,len(Ec.noise_levels),num_all_pat,50,50)[0][0][j], label = str(j))
        ax.scatter(centers[0,:,0,0], centers[0,:,0,1], c =ca1_noisy_input_pattern[0,0,j])
        plt.legend()
        #plt.title("Noise level = %.1f" % (Ec.noise_levels[j]/Ec.cells))

def OfflineStructredasHetteroAssociatedSequenceLerninginCa3(num_transitions= None, ca3model= None, Ecinput_stored= None): # when Ec.cells == Ca3.cells >> I am testing for the same weight matrics in Feed forward and recurrent colaterals!!!!
#if __name__ == "__main__":
    #num_transitions= 3; ca3model= 1
    alpha_step = 1./(num_transitions-1)
    alpha_vector = np.arange(0, 1.01, alpha_step) 
    internal_force = alpha_vector[ca3model] 
    external_force = 1 - internal_force

    def save_object(obj, path, info=True, compressed=True):
	    ''' Saves an object to file.
	
	    :Parameters:
	        obj: object to be saved.
	            -type: object
	
	        path: Path and name of the file
	             -type: string
	
	        info: Prints statements if TRUE
	             -type: bool
	
	    '''
	    if info == True:
	        print '-> Saving File  ... ',
	    try:
	        if compressed:
	            fp = gzip.open(path, 'wb')
	            cPickle.dump(obj, fp)
	            fp.close()
	        else:
	            file_path = open(path, 'w')
	            cPickle.dump(obj, file_path)
	        if info == True:
	            print 'done!'
	    except:
	        print "-> File writing Error: "
	        return None

    #import ipdb; ipdb.set_trace()
    num_nummy_seq = 16
    num_seq = num_nummy_seq
    seq_len = 16
    num_all_pat = (num_seq * seq_len)
    in_weight = .000001  ### it is not working when it is exactly zero
    noise_levels_ca3= np.arange(0, 2500, 2600) 
    noise_levels_ec = np.arange(0, 1101, 1250) 
    plastic= 1
    centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.

##########################
    bo = 25 #bo: length and width of arena
    T = seq_len # length of the sequence
    numofseq = num_seq # number of trajectories

    xlimit= 2
    ylimit= bo - 2
    
    Trajec = np.zeros([numofseq,T,2])
    
    for n  in range(numofseq):
    
        lx = pl.randint(xlimit,ylimit)    # starting point
        ly = pl.randint(xlimit,ylimit)
        
        x = np.ones([bo,bo])*range(0,bo) 
        y = x.T
        
        for i in range(T):
            
            vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
            vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
        
            mdx = pl.binomial(1,0.5)   # moving direction for x
            mdy = pl.binomial(1,0.5)   # for y
            delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
            delta_y = vy*(1*mdy+(-1)*(1-mdy))
            
            # change 250 to your maximum, 50 to your minimum      
            lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
                 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
             
            ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
                 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0))) 
            Trajec[n,i,:] = lx, ly    
    
    Tr = pl.int0(Trajec.reshape(numofseq*T,2))
    
    #store_indizes = ((Tr[:,0]*T)+Tr[:,1]).reshape(1,num_all_pat)
    
    #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
    
    store_indizes = None 

    #centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
    centers = np.random.uniform(0,1, size=(1,2500,1,2))
    #store_indizes = np.arange(num_seq).reshape(1,num_seq)
    #store_indizes = None
    #Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =num_all_pat, number_to_store = num_all_pat, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels_ca3, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
    
    Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels_ca3, sparsity = 0.053, store_indizes = store_indizes) 
    
    #Ca3 = Grid(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = num_all_pat, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels_ca3, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.053)# actFunction?
################################################################  

    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.932, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
#################################################################

    Ca3_Ca3 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.9932, learnrate= 1., subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
    
    #Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.9932, learnrate = 0.05 ,subtract_input_mean = True, subtract_output_mean = True, initMethod = Network.makeWeightsNormalDistributed, actFunction = Network.getOutputWTA, initConnectionMethod = Network.initConnectionRandom, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5) 
    
##################################################################    
    ca3_triggered_patterns = Ec_Ca3.getOutput(Ec_Ca3, input_pattern = Ecinput_stored) #!!!!!!!!!!!!!!!!!!!!!!
    #initializing_ca3_storing = Ca3.makeNoiseRandomFire(pattern=ca3_triggered_patterns[0], noise_levels=noise_levels_ca3)  
    #print initializing_ca3_storing.shape  
    
    #normalize(ca3_triggered_patterns)  
    normalize(Ca3.input_stored)
      


    initializing_ca3_storing = ca3_triggered_patterns#Ca3.input_stored*internal_force  + ca3_triggered_patterns*external_force
    '''
    
############################ WTA method #######################    
    size = list(np.shape(initializing_ca3)) #dimension of input
    size[-1] = Ca3.cells # change to dimension of output

    winner = np.argsort(initializing_ca3)[...,-int(Ca3.cells * 0.053):size[-1]]

    fire_rate = np.ones(size, 'bool')
    initializing_ca3_storing = np.zeros(size, 'bool')

    indices = np.mgrid[0:size[0],0:size[1],0:int(Ca3.cells * 0.053)]
    initializing_ca3_storing[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
    '''    
####################################################################    
  
    ca3_inputs = initializing_ca3_storing.reshape([1,num_all_pat, Ca3.cells])
    for j in range(num_seq):
		Ca3_Ca3.learnAssociation(input_pattern = ca3_inputs[0, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = ca3_inputs[0, (j*seq_len)+1:(j*seq_len)+seq_len])
        #Ca3_Ca3.learnAssociation(input_pattern = Ca3.noisy_input_stored[0,1, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = Ca3.noisy_input_stored[0,1, (j*seq_len)+1:(j*seq_len)+seq_len])
        #Ca3_Ca3.learnAssociation(input_pattern = Ca3.input_stored[1, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = Ca3.input_stored[1, (j*seq_len)+1:(j*seq_len)+seq_len])# is it possible? just ask wheather weights are set to zero everytime!
        #print (j*seq_len), (j*seq_len)+seq_len-1, (j*seq_len)+1, (j*seq_len)+seq_len
        
    #ca3_input = Ca3.input_stored
    
    
    '''
    
    ca3_input = np.zeros([1, num_all_pat, Ca3.cells])
    
    for j in range(num_seq):
        inhibition =  np.copy(Ca3.input_stored[0, j]) #
        self_inhibition = np.ones(Ca3.input_stored[0, j].shape) #


        inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
        inhibition *= in_weight
        inhibition[inhibition==0.] = 1.
        self_inhibition *= inhibition
        

        ca3_input[0, j*seq_len] = Ca3.input_stored[0, j] #
         
        for i in range(1, seq_len):

            ca3_input[0, i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern=ca3_input[0, i+(j*seq_len)-1], self_inhibition=self_inhibition)


            inhibition = np.copy(ca3_input[0, i+(j*seq_len)]) # 
            

            inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
            inhibition *= in_weight
            inhibition[inhibition==0.] = 1.
            self_inhibition *= inhibition
    
    
    if plastic:
        Ca3_Ca3.learnOneShootAllPattern(input_pattern = ca3_input, method = OneShootmehdi.learnFiringDependent, seq_len = seq_len, num_seq= num_seq) 
    '''
    
    #np.savetxt('OfflineStructredWeights.txt', Ca3_Ca3.weights)
    save_object(obj = np.float32(Ca3_Ca3.weights) , path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/OfflineStructredWeights', info=True, compressed=False)

    #np.savetxt('OfflineStructredConnections.txt', np.int_(Ca3_Ca3.connection))
    save_object(obj = np.int32(Ca3_Ca3.connection) , path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/OfflineStructredConnections', info=True, compressed=False)

    
    #return ca3_input, Ca3.noisy_input_stored[:,1,np.arange(0, num_all_pat-seq_len +1, seq_len)]#, Ca3.noisy_input_stored[:,1,np.arange(0, num_all_pat-seq_len +1, seq_len)], Ca3.noisy_input_stored[:,2,np.arange(0, num_all_pat-seq_len +1, seq_len)]
    return ca3_inputs
    


#########################################################################
#def Ec_Ca3_Ca1_Ec_Closed_Loop(): ## the problem is using LCN in close loop when I have self_inhibition. Check how self inhibition is initialzed!!!! check Connectivity kernel
# having self-inhibition in pretrain or online learning model helps a lot to  retrieve a correct sequence. why?
#dummynoise=0.00
#if __name__ == "__main__":
def gg():
    #import ipdb; ipdb.set_trace()
    dummynoise=0.00
    num_nummy_seq = 16
    num_seq = num_nummy_seq
    seq_len = 16
    num_all_pat = (num_seq * seq_len)
    in_weight = .66666  ### it is not working when it is exactly zero
    noise_levels= np.arange(0, 1101, 1550)
    plastic= 0
    num_transitions = 2
    #############################################################      
    
    allcorrdata = np.zeros([2, num_transitions, 3, len(noise_levels), num_seq, seq_len])  
    
    
    for inputstatistics in range(1,2):
        if inputstatistics ==0:
            random_input_ec=1; grid_input_ec=0;
            inputtitle = "Random input, "
        if inputstatistics ==1:
            random_input_ec=0; grid_input_ec=1;
            inputtitle = "Grid input, "
        for ca3models in range(num_transitions):        
            ca3_model_random=0; ca3_model_local=0;  ca3_model_pretrained=1; ca3_model_onlinetraining=0          
            
        ##################################################################
        
        
            bo = 50 #bo: length and width of arena
            T = seq_len # length of the sequence
            numofseq = num_seq # number of trajectories
			
            xlimit= 2
            ylimit= bo - 2
			
            Trajec = np.zeros([numofseq,T,2])
			
            for n  in range(numofseq):
			
                lx = pl.randint(xlimit,ylimit)    # starting point
                ly = pl.randint(xlimit,ylimit)
				#print lx, ly
                x = np.ones([bo,bo])*range(0,bo) 
                y = x.T
				
                for i in range(T):
					
					vx = abs(pl.normal(1.5,1.5))  # mean velocity and sd
					vy = abs(pl.normal(1.5,1.5))  # mean velocity and sd
					#print vx, vy
					mdx = pl.binomial(1,0.5)   # moving direction for x
					mdy = pl.binomial(1,0.5)   # for y
					#print mdx, mdy
					delta_x = vx*(1*mdx+(-1)*(1-mdx))  # moving distance
					delta_y = vy*(1*mdy+(-1)*(1-mdy))
					#print delta_x, delta_y
					# change 250 to your maximum, 50 to your minimum      
					lx = pl.rint(lx + (1*mdx+(-1)*(1-mdx))*(min(abs(mdx*ylimit+(1-mdx)* xlimit-lx),abs(delta_x))
						 + min((abs(mdx*ylimit+(1-mdx)* xlimit-lx)-abs(delta_x)),0)))
					 
					ly = pl.rint(ly + (1*mdy+(-1)*(1-mdy))*(min(abs(mdy*ylimit+(1-mdy)* xlimit-ly),abs(delta_y))
						 + min((abs(mdy*ylimit+(1-mdy)* xlimit-ly)-abs(delta_y)),0)))
						  
					Trajec[n,i,:] = lx, ly    
			
            Tr = pl.int0(Trajec.reshape(numofseq*T,2))
            
            #store_indizes = ((Tr[:,0]*bo)+Tr[:,1]).reshape(1,num_all_pat)
            
            #store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
            
            store_indizes = None    
        #################################################################
            if grid_input_ec:
                Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = num_all_pat, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
                #np.save(outfile, Ec.noisy_input_stored)
            elif random_input_ec:
                Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    
            
            Ca1Cells = 4200
        
            centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
            #centers = np.random.uniform(0,1, size=(1,2500,1,2))
            #store_indizes = np.arange(num_seq).reshape(1,num_seq)
            store_indizes = None
   
            Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes)
            EcCa3initMethod = Network.makeWeightsZero
            ca3_input = OfflineStructredasHetteroAssociatedSequenceLerninginCa3(num_transitions= num_transitions, ca3model= ca3models, Ecinput_stored= Ec.input_stored)#, Ca3_original_noisy_1_input_stored, Ca3_original_noisy_2_input_stored 
            Ca3Ca3initMethod = Network.makeWeightsOfflineStructured
            #Ca3Ca3initMethod = Network.makeWeightsplacecellstrained
            Ca3Ca3initConnectionMethod = Network.initConnectionOfflineStructured
            #Ca3Ca3initConnectionMethod = Network.initConnectionplacecellstrained            
        #################################################################
        
            Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = EcCa3initMethod, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
        
            #Ca3_Ca3 = OneShootmehdi(input_cells = Ca3.cells, cells = Ca3.cells, number_winner = int(Ca3.cells*Ca3.sparsity), connectivity = 0.32, learnrate = 0.0, subtract_input_mean = True, subtract_output_mean = True, initMethod = Ca3Ca3initMethod, actFunction = Network.getOutputWTA_Self_Inhibition, initConnectionMethod = Ca3Ca3initConnectionMethod, active_in_env = Ca3.cells, n_e = 1, weight_mean = 1, weight_sigma = 0.5) 
            Ca3_Ca3 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= .1, subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA_Self_Inhibition, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Ca3Ca3initMethod, initConnectionMethod = Ca3Ca3initConnectionMethod, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
        
            Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097
        
        #######
            Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.097), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
            
        #######
        
            Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.32, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsZero, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
 
        #################################################################   
            
            ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.input_stored)
            
            
            Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = ca3_input)
        ##########
            Ca3_Ca1.learnAssociation(input_pattern = ca3_input , output_pattern = ca1_input)
            Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)
        
        ##################################################################
            Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
            
            initializing_ca3_retrieving = Ec_Ca3.noisy_output # it should be adjusted so for every model is usable
        ##################################################################
        
            ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, Ca3.cells])
            
            for j in range(num_seq):
        
                inhibition = np.copy(initializing_ca3_retrieving[:, :,(j*seq_len)]) # np.copy(Ca3.noisy_input_stored[:, :,j]) #     
                
                self_inhibition = np.ones(initializing_ca3_retrieving[:, :,(j*seq_len)].shape) # np.ones(Ca3.noisy_input_stored[:, :,j].shape)
                
                ca1_noisy_input_pattern[:,:,j*seq_len] =  initializing_ca3_retrieving[:, :,(j*seq_len)] # Ca3.noisy_input_stored[:, :,j]# 
        
                inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
                inhibition *= in_weight
                inhibition[inhibition==0.] = 1.
                self_inhibition *= inhibition
        
        
         
                for i in range(1, seq_len):
        
                    #Ca1Cue = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1])
                    #EcCue = Ca1_Ec.getOutput(Ca1_Ec, input_pattern=Ca1Cue)
                    #Ca3Cue = Ec_Ca3.getOutput(Ec_Ca3, input_pattern=EcCue)
                    
                    combinedCue = ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1]# + Ca3Cue
                    #combinedCue[combinedCue>1] == 1
                    ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern= combinedCue, self_inhibition=self_inhibition)
        
                    inhibition = np.copy(ca1_noisy_input_pattern[:, :,i+(j*seq_len)]) #
                    inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
                    inhibition *= in_weight
                    inhibition[inhibition==0.] = 1.
                    self_inhibition *= inhibition
            
            
            Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
            
            Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)
        
        
        ################################### Seq. Learning in Ca3 ###############
        
        
            noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
            
            ca3_input_output_corr = Corelations(patterns_1=ca3_input, patterns_2=ca1_noisy_input_pattern, in_columns = False)
            corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
        
            ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2=Ca3_Ca1.noisy_output, in_columns = False)
            corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
        ###########################################################
            allcorrdata[inputstatistics, ca3models, 2] =  noisy_cue_allcorrelations
            allcorrdata[inputstatistics, ca3models, 1] =  corca1
            allcorrdata[inputstatistics, ca3models, 0] =  corca3
            
            '''
########################## PCA on stored patterns in CA3 ###################################### 
            n_components= 0.85    
            data1 = ca3_input.reshape(num_all_pat,Ca3.cells)
            pca = PCA(n_components= n_components) 
            pca.fit(data1)
            expl_var_ratio_ca3_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_
            
            data1 = ca1_input.reshape(num_all_pat,Ca1Cells)
            pca = PCA(n_components= n_components) 
            pca.fit(data1)
            expl_var_ratio_ca1_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_
            
            data1 = Ec.input_stored.reshape(num_all_pat,Ec.cells)
            pca = PCA(n_components= n_components) 
            pca.fit(data1)
            expl_var_ratio_ec_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_
            '''
            ''' 
########################## PCA on stored patterns in CA1###################################### 
            n_components= .85  
            data1 = ca1_input.reshape(num_all_pat,Ca1Cells)
            data2 = Ca3_Ca1.noisy_output[0,0]
            #data2 = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern = Ec_Ca3.noisy_output[:,0]).reshape(num_all_pat, Ca1Cells)
            #data2 = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern = ca3_input).reshape(num_all_pat, Ca1Cells)
            pca = PCA(n_components= n_components) 
            pca.fit(data1)
            s_transformed = pca.transform(data1)
            r_transformed = pca.transform(data2)
            subcor = Corelations(patterns_1=s_transformed.reshape(1,s_transformed.shape[0], s_transformed.shape[1]), patterns_2=r_transformed.reshape(1,r_transformed.shape[0], r_transformed.shape[1]), in_columns= False)
            pairweised_corr_s_r_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]
            pairweised_corr_s_r_corresponding_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCorOrigOrig()[0]                
            
            
            #subcor = Corelations(patterns_1= ca1_input, patterns_2= Ca3_Ca1.noisy_output[:,0], in_columns= False)
            subcor = Corelations(patterns_1= ca1_input, patterns_2= data2.reshape(1,num_all_pat, Ca1Cells), in_columns= False)
            pairweised_corr_s_r_complete_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]       
            pairweised_corr_s_r_complete_corresponding_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCorOrigOrig()[0]    
            '''
    #save_object(obj =Ec.getOrigVsOrig() , path= '/home/bayatmz4/Documents/My-Project/RS/ECgetOrigvsOrig.txt', info=True, compressed=False)
    save_object(obj =allcorrdata , path= '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3/allcorrdata.txt', info=True, compressed=False)
    #save_object(obj =pairweised_corr_s_r_complete_ca1 , path= '/home/bayatmz4/Documents/My-Project/RS/pairweiseCA1.txt', info=True, compressed=False)
    #save_object(obj =pairweised_corr_s_r_complete_corresponding_ca1 , path= '/home/bayatmz4/Documents/My-Project/RS/pairweiseCA1corresponding.txt', info=True, compressed=False)
    
    #save_object(obj =expl_var_ratio_ca3_pca_stored_patts , path= '/home/bayatmz4/Documents/My-Project/RS/pcaCa3.txt', info=True, compressed=False)
    #save_object(obj =expl_var_ratio_ca1_pca_stored_patts , path= '/home/bayatmz4/Documents/My-Project/RS/pcaCa1.txt', info=True, compressed=False)
    #save_object(obj =expl_var_ratio_ec_pca_stored_patts , path= '/home/bayatmz4/Documents/My-Project/RS/pcaEc.txt', info=True, compressed=False)
    
    
    #save_object(obj =ca3inputrecord , path= '/home/bayatmz4/Documents/My-Project/RS/pairweise-correlations-in-ca3/ca3inputrecord', info=True, compressed=False)
    
    #placecellstrainedWeights


#if __name__ == "__main__":
for test in range(1):	
	for real in range(1):
	    path2 = '/home/bayatmz4/Documents/My-Project/RS/Revision-combined-inputs-to-ca3-realisation/realisationdata'
	    #import ipdb; ipdb.set_trace()
	    dummynoise=0.00
	    num_nummy_seq = 16
	    num_seq = num_nummy_seq
	    seq_len = 16
	    num_all_pat = (num_seq * seq_len)
	    in_weight = .99999999999999  ### it is not working when it is exactly zero
	    noise_levels= np.arange(0, 1101, 1220)
	    plastic= 0
	    num_transitions = 2
	    delta_force = (1./(num_transitions-1)) - 0.01
	    internal_force = np.arange(0.,1.01,delta_force)
	    external_force = 1. - internal_force 
	    
	     

	    #############################################################      
	    
	    allcorrdata = np.zeros([2, num_transitions, 3, len(noise_levels), num_seq, seq_len]) 	
	    ca3_temporal_corr = np.zeros([2, num_transitions, num_seq, seq_len-1])
	    ca3_any_corr_stored = np.zeros([2, num_transitions, (num_all_pat*(num_all_pat-1))])
	    ec_any_corr_stored = np.zeros([(num_all_pat*(num_all_pat-1))])
	    expl_var_ratio_ca3_pca_stored_patts = {}
	    expl_var_ratio_ca1_pca_stored_patts = {}
	    expl_var_ratio_ec_pca_stored_patts = {}
	    
	    pairweised_corr_s_r_complete_ca1 = {}
	    pairweised_corr_s_r_complete_corresponding_ca1 = {}    

	    pairweised_corr_s_r_complete_first_ca3 = {}
	    pairweised_corr_s_r_complete_corresponding_first_ca3 = {}  
	    
	    pairweised_corr_s_r_complete_all_ca3 = {}
	    pairweised_corr_s_r_complete_corresponding_all_ca3 = {} 	    
	    
	    for inputstatistics in range(1,2):
		if inputstatistics ==0:
		    random_input_ec=1; grid_input_ec=0;
		    inputtitle = "Random input, "
		if inputstatistics ==1:
		    random_input_ec=0; grid_input_ec=1;
		    inputtitle = "Grid input, "
	  
		##################################################################
		'''
		num_locations = 25#np.sqrt(num_all_pat) # width and length of the grid; also check the Random_walk class
		trace = np.zeros([num_seq,seq_len,2])
		trajectory = numx.int32(numx.ones([num_seq,seq_len,1])*1000)
		for i in range(num_seq):
				traj = Random_walk(width= num_locations , height= num_locations)
				trace[i,0] = traj.current_position
				for j in range(1,seq_len):
					trace[i,j] = traj.get_next_position(momentum = 0.6, reorientation = 0.8)
			
		Tr = np.int32(np.round(trace.reshape(num_all_pat,2)))
		Tr[Tr==num_locations] = num_locations - 1 
			
		store_indizes = np.int32((Tr[:,0]*num_locations)+Tr[:,1]).reshape(1,num_all_pat)
			'''
		num_locations = 42 #np.int32(np.sqrt(num_all_pat))+10#np.sqrt(num_all_pat) # width and length of the grid; also check the Random_walk class
		
		trace = np.zeros([num_seq,seq_len,2])
		trajectory = np.int32(np.ones([num_seq,seq_len,1])*10000)
		size = [num_locations,num_locations]#trace.shape
		for i in range(num_seq):
				traj = Random_walk(width= num_locations-2 , height= num_locations-2)
				trace[i,0] = traj.current_position
				position = np.int32((trace[i,0][0]*size[0])+trace[i,0][1])
				#print position
				trajectory[i,0] = position
				if (len(trajectory[trajectory==position]) !=1) or (position >= (num_locations*num_locations)):
					for h in range(50000):
						trace[i,0] = traj.current_position
						position = np.int32((trace[i,0][0]*size[0])+trace[i,0][1])
						trajectory[i,0] = position
						#print position
						if (len(trajectory[trajectory==position]) ==1) and (position < (num_locations*num_locations)):
							break
				
				for j in range(1,seq_len):
					trace[i,j] = traj.get_next_position(momentum = 0.6, reorientation = 0.99)
					position = np.int32((trace[i,j][0]*size[0])+trace[i,j][1])
					trajectory[i,j] = position
					#print position
					if (len(trajectory[trajectory==position]) !=0) or (position >= (num_locations*num_locations)):
						for k in range(50000):
							trace[i,j] = traj.get_next_position(momentum = 0.6, reorientation = 0.99)
							position = np.int32((trace[i,j][0]*size[0])+trace[i,j][1])
							trajectory[i,j] = position
							#print position
							if (len(trajectory[trajectory==position]) ==1) and (position < (num_locations*num_locations)):
								break


		store_indizes = trajectory.reshape(1,num_all_pat)
		
		#store_indizes = np.arange(num_all_pat).reshape(1,num_all_pat)# It could be adjested in a way to create more realistic trajectories
			
		#store_indizes = None    
		#################################################################
		if grid_input_ec:
				Ec = Grid(number_cells = 1100, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = num_locations*num_locations, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, rat=1, grid_mode = 'modules', cage =[1,1], sparsity = 0.35)# actFunction?
				#np.save(outfile, Ec.noisy_input_stored)
		elif random_input_ec:
				Ec = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_locations*num_locations, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)    

		Ca1Cells = 4200
		
		#centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
		
			#centers = np.random.uniform(0,1, size=(1,2500,1,2))
			#store_indizes = np.arange(num_seq).reshape(1,num_seq)
		store_indizes = None

		############################
		ec_any_cor_stored = Corelations(patterns_1=Ec.input_stored, patterns_2=Ec.input_stored, in_columns = False)	    
		ec_any_corr_stored = ec_any_cor_stored.getCorOrigOther()[0]		
		print np.float32(len(ec_any_corr_stored[ec_any_corr_stored>0.1]))/len(ec_any_corr_stored)
		############################			




		for ca3models in range(num_transitions): 

		 
		    ca3_model_random=0; ca3_model_local=0;  ca3_model_pretrained=0; ca3_model_onlinetraining=1   
		    
		    
		    if ca3_model_local:
			Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =num_seq, number_to_store = num_seq, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
			EcCa3initMethod = Network.makeWeightsZero
			Ca3Ca3initMethod = Network.makeWeightsUniformDistributed
			Ca3Ca3initConnectionMethod = Network.initConnectionNEST
		    elif ca3_model_random:
			Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_seq, number_to_store = num_seq, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes)
			EcCa3initMethod = Network.makeWeightsZero 
			Ca3Ca3initMethod = Network.makeWeightsUniformDistributed
			Ca3Ca3initConnectionMethod = Network.initConnectionRandom
		    elif ca3_model_pretrained:
			Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes)
			EcCa3initMethod = Network.makeWeightsZero
			originalSeqs, Ca3_original_noisy_0_input_stored = OfflineStructredasHetteroAssociatedSequenceLerninginCa3()#, Ca3_original_noisy_1_input_stored, Ca3_original_noisy_2_input_stored 
			Ca3Ca3initMethod = Network.makeWeightsOfflineStructured
			#Ca3Ca3initMethod = Network.makeWeightsplacecellstrained
			Ca3Ca3initConnectionMethod = Network.initConnectionOfflineStructured
			#Ca3Ca3initConnectionMethod = Network.initConnectionplacecellstrained
		    elif ca3_model_onlinetraining:
					#centers = np.ravel((np.mgrid[0:50, 0:50] + 0.0), order = 'F').reshape(1, 2500, 1, 2)/50.
			Ca3 = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 2500, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.053, store_indizes = store_indizes)
					#Ca3 = Grid(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns = num_all_pat, number_to_store =num_all_pat ,n_e =1,noise_levels = noise_levels, store_indizes = store_indizes, grid_mode = 'modules', cage =[1,1], sparsity = 0.053)# actFunction 
					#Ca3 = PlaceFields(number_cells = 2500, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputWTA, number_patterns =num_all_pat, number_to_store = num_all_pat, store_indizes = store_indizes, n_e =1, noise_levels=noise_levels, normed = 0, sparsity = 0.053, centers = centers) # numbertostore = numberofpatterns?
		   
			plastic =1  
			EcCa3initMethod = Network.makeWeightsZero
			Ca3Ca3initMethod = Network.makeWeightsZero
			Ca3Ca3initConnectionMethod = Network.initConnectionRandom          


			
		######################
		
		    Ec_Ca3 = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.932, learnrate= .95, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = EcCa3initMethod, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)

		    Ca3_Ca3 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.932, learnrate= .95, subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTA_Self_Inhibition, number_winner= int(Ca3.cells * Ca3.sparsity), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Ca3Ca3initMethod, initConnectionMethod = Ca3Ca3initConnectionMethod, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		
		    Ca3_Ca1 = HeteroAssociation(input_cells=Ca3.cells, cells=Ca1Cells, connectivity = .932, learnrate= .95, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.0497), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = .5)# initMethod and connections? 0.097
		
		#######
		    Ec_Ca1 = HeteroAssociation(input_cells=Ec.cells, cells=Ca1Cells, connectivity = 0.03, learnrate= 1, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ca1Cells * 0.0497), e_max = 0.1, active_in_env = Ca1Cells, n_e = 1, initMethod = Network.makeWeightsUniformDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		    
		#######
		
		    Ca1_Ec = HeteroAssociation(input_cells=Ca1Cells, cells=Ec.cells, connectivity = 0.932, learnrate= 0.95, subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTA, number_winner= int(Ec.cells * Ec.sparsity), e_max = 0.1, active_in_env = Ec.cells, n_e = 1, initMethod = Network.makeWeightsNormalDistributed, initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		
		#################################################################

			
		    if ca3_model_onlinetraining:
					Ec_trigered = Input(inputMethod = Input.makeInputNormalDistributed,number_cells = 1100, number_patterns = num_all_pat, number_to_store = num_all_pat, actFunction = Input.getOutputWTA, noiseMethod = Input.makeNoiseRandomFire, noise_levels = noise_levels, sparsity = 0.35, store_indizes = store_indizes)   
			  
					Ca3_Ca3_intrinsic = HeteroAssociation(input_cells=Ca3.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = True, actFunction = Network.getOutputWTALinear_Self_Inhibition, number_winner= int(Ca3.cells * 0.753), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsUniformDistributed, initConnectionMethod = Ca3Ca3initConnectionMethod, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)     
		    
					Ec_Ca3_trigered = HeteroAssociation(input_cells=Ec.cells, cells=Ca3.cells, connectivity = 0.32, learnrate= 1., subtract_input_mean = True, subtract_output_mean = False, actFunction = Network.getOutputWTALinear, number_winner= int(Ca3.cells * 0.753), e_max = 0.1, active_in_env = Ca3.cells, n_e = 1, initMethod = Network.makeWeightsUniformDistributed,initConnectionMethod = Network.initConnectionRandom, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)                       
		#################################################################			
					
					initializing_ca3_WTA = np.zeros([1, num_all_pat, Ca3.cells])
					initializing_ca3_linear = np.zeros([1, num_all_pat, Ca3.cells])
					
					ca3_triggered_patterns = Ec_Ca3_trigered.getOutput(Ec_Ca3_trigered, input_pattern = Ec.input_stored) 
					normalize(ca3_triggered_patterns)    
					
					Ca3_initialise = Ec_Ca3_trigered.getOutput(Ec_Ca3_trigered, input_pattern = Ec_trigered.input_stored) # CA3 is initialized randomly!
					normalize(Ca3_initialise)
					
					for j in range(num_seq):
						initializing_ca3_linear[0,j*seq_len] = (external_force[ca3models] * np.copy(ca3_triggered_patterns[0,j*seq_len])) +  (internal_force[ca3models] * np.copy(Ca3_initialise[0,j*seq_len])) 
						
					############################ WTA method #######################    
						size = [1,1,Ca3.cells] #dimension of input
						size[-1] = Ca3.cells # change to dimension of output
					
						winner = np.argsort(initializing_ca3_linear[0,j*seq_len])[...,-int(Ca3.cells * 0.053):size[-1]]
					
						fire_rate = np.ones(size, 'bool')
						ca3_WTA = np.zeros(size, 'bool')
					
						indices = np.mgrid[0:size[0],0:size[1],0:int(Ca3.cells * 0.053)]
						ca3_WTA[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
					    
						initializing_ca3_WTA[0,j*seq_len] = ca3_WTA
					####################################################################                    
						
						inhibition = np.copy(initializing_ca3_WTA[0,j]) 
						self_inhibition = np.ones(initializing_ca3_WTA[0,j].shape)

						inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
						inhibition *= in_weight
						inhibition[inhibition==0.] = 1.
						self_inhibition *= inhibition

					####################################################################                    					
						initializing_ca3_WTA[0,j*seq_len] = Ca3_Ca3_intrinsic.getOutput(Ca3_Ca3_intrinsic, input_pattern=initializing_ca3_WTA[0,j*seq_len], self_inhibition=self_inhibition)# Just to do one extra iteration in CA3


						 
						for i in range(0, seq_len):

							intrinsic_pattern = Ca3_Ca3_intrinsic.getOutput(Ca3_Ca3_intrinsic, input_pattern=initializing_ca3_WTA[0, i+(j*seq_len)], self_inhibition=self_inhibition)
							normalize(intrinsic_pattern)
								
							initializing_ca3_linear[0, i+(j*seq_len)] = (internal_force[ca3models] * intrinsic_pattern) + (external_force[ca3models] * np.copy(ca3_triggered_patterns[0, i+(j*seq_len)]))
							
						############################ WTA method #######################    
							size = [1,1,Ca3.cells] #dimension of input
							size[-1] = Ca3.cells # change to dimension of output
						
							winner = np.argsort(initializing_ca3_linear[0, i+(j*seq_len)])[...,-int(Ca3.cells * 0.053):size[-1]]
						
							fire_rate = np.ones(size, 'bool')
							ca3_WTA = np.zeros(size, 'bool')
						
							indices = np.mgrid[0:size[0],0:size[1],0:int(Ca3.cells * 0.053)]
							ca3_WTA[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
						    
							initializing_ca3_WTA[0, i+(j*seq_len)] = ca3_WTA
						####################################################################  						
				
							inhibition = np.copy(initializing_ca3_WTA[0, i+(j*seq_len)]) # 
							inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
							inhibition *= in_weight
							inhibition[inhibition==0.] = 1.
							self_inhibition *= inhibition
					
					  
				####################################################################                    
					ca3_input=  initializing_ca3_WTA
					
					for j in range(num_seq):
						Ca3_Ca3.learnAssociation(input_pattern = ca3_input[0, (j*seq_len):(j*seq_len)+seq_len-1] , output_pattern = ca3_input[0, (j*seq_len)+1:(j*seq_len)+seq_len])
	
		#################################################################   
		    
		    ca1_input = Ec_Ca1.getOutput(Ec_Ca1, input_pattern = Ec.noisy_input_stored[:,0])#Ec.input_stored)
		    
		    
		    Ec_Ca3.learnAssociation(input_pattern = Ec.input_stored , output_pattern = ca3_input)
		##########
		    Ca3_Ca1.learnAssociation(input_pattern = ca3_input , output_pattern = ca1_input)
		    Ca1_Ec.learnAssociation(input_pattern = ca1_input , output_pattern = Ec.input_stored)
		
		##################################################################
		    Ec_Ca3.recall(input_pattern = Ec.noisy_input_stored)
		    
		    initializing_ca3_retrieving = Ec_Ca3.noisy_output # it should be adjusted so for every model is usable
		##################################################################
		
		    ca1_noisy_input_pattern = np.zeros([Ca3.n_e, len(Ca3.noise_levels), num_all_pat, Ca3.cells])
		    
		    for j in range(num_seq):
		
			inhibition = np.copy(initializing_ca3_retrieving[:, :,(j*seq_len)]) # np.copy(Ca3.noisy_input_stored[:, :,j]) #     
			
			self_inhibition = np.ones(initializing_ca3_retrieving[:, :,(j*seq_len)].shape) # np.ones(Ca3.noisy_input_stored[:, :,j].shape)
			
			ca1_noisy_input_pattern[:,:,j*seq_len] =  initializing_ca3_retrieving[:, :,(j*seq_len)] # Ca3.noisy_input_stored[:, :,j]# 
		
			inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
			inhibition *= in_weight
			inhibition[inhibition==0.] = 1.
			self_inhibition *= inhibition
		
		
		 
			for i in range(1, seq_len):
		
			    #Ca1Cue = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern=ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1])
			    #EcCue = Ca1_Ec.getOutput(Ca1_Ec, input_pattern=Ca1Cue)
			    #Ca3Cue = Ec_Ca3.getOutput(Ec_Ca3, input_pattern=EcCue)
			    
			    combinedCue = ca1_noisy_input_pattern[:, :,i+(j*seq_len)-1]# + Ca3Cue
			    #combinedCue[combinedCue>1] == 1
			    ca1_noisy_input_pattern[:, :,i+(j*seq_len)] = Ca3_Ca3.getOutput(Ca3_Ca3, input_pattern= combinedCue, self_inhibition=self_inhibition)
		
			    inhibition = np.copy(ca1_noisy_input_pattern[:, :,i+(j*seq_len)]) #
			    inhibition[inhibition>0.] *= 1./inhibition[inhibition>0.]
			    inhibition *= in_weight
			    inhibition[inhibition==0.] = 1.
			    self_inhibition *= inhibition
		    
		    
		    Ca3_Ca1.recall(input_pattern = ca1_noisy_input_pattern)
		    
		    Ca1_Ec.recall(input_pattern = Ca3_Ca1.noisy_output)
		
		
		################################### Seq. Learning in Ca3 ###############
		
		
		    noisy_cue_allcorrelations = Ca1_Ec.Cor['StoredRecalled'].getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
		    
		    ca3_input_output_corr = Corelations(patterns_1=ca3_input, patterns_2=ca1_noisy_input_pattern, in_columns = False)
		    corca3 = ca3_input_output_corr.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
		
		    ca1_input_output_correlation = Corelations(patterns_1=ca1_input, patterns_2=Ca3_Ca1.noisy_output, in_columns = False)
		    corca1 = ca1_input_output_correlation.getCorOrigOrig().reshape(len(Ec.noise_levels), num_seq, seq_len)
		    
		    ca3_tempo_corr = Corelations(patterns_1=ca3_input.reshape(num_seq, seq_len,Ca3.cells)[:,:-1], patterns_2=ca3_input.reshape(num_seq, seq_len,Ca3.cells)[:,1:], in_columns = False)
		    ca3_tempor_corr = ca3_tempo_corr.getCorOrigOrig()[0].reshape(num_seq, seq_len-1)

		                
		###########################################################
		    allcorrdata[inputstatistics, ca3models, 2] =  noisy_cue_allcorrelations
		    allcorrdata[inputstatistics, ca3models, 1] =  corca1
		    allcorrdata[inputstatistics, ca3models, 0] =  corca3    
		    
		    ca3_temporal_corr[inputstatistics, ca3models] = ca3_tempor_corr    
		    
		    
		    ca3_any_cor_stored = Corelations(patterns_1=ca3_input, patterns_2=ca3_input, in_columns = False)
		    
		    ca3_any_corr_stored[inputstatistics, ca3models] = ca3_any_cor_stored.getCorOrigOther()[0]
		
    
		    
		'''
	########################## PCA on stored patterns in CA3 ###################################### 
		    n_components= 0.85    
		    data1 = ca3_input.reshape(num_all_pat,Ca3.cells)
		    pca = PCA(n_components= n_components) 
		    pca.fit(data1)
		    expl_var_ratio_ca3_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_
		    
		    data1 = ca1_input.reshape(num_all_pat,Ca1Cells)
		    pca = PCA(n_components= n_components) 
		    pca.fit(data1)
		    expl_var_ratio_ca1_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_
		    
		    data1 = Ec.input_stored.reshape(num_all_pat,Ec.cells)
		    pca = PCA(n_components= n_components) 
		    pca.fit(data1)
		    expl_var_ratio_ec_pca_stored_patts['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = pca.explained_variance_ratio_



	########################## PCA on stored patterns in CA1###################################### 
	      
		    n_components= .85  
		    data1 = ca1_input.reshape(num_all_pat,Ca1Cells)
		    data2 = Ca3_Ca1.noisy_output[0,0]
		    #data2 = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern = Ec_Ca3.noisy_output[:,0]).reshape(num_all_pat, Ca1Cells)
		    #data2 = Ca3_Ca1.getOutput(Ca3_Ca1, input_pattern = ca3_input).reshape(num_all_pat, Ca1Cells)
		    
		    #pca = PCA(n_components= n_components) 
		    #pca.fit(data1)
		    #s_transformed = pca.transform(data1)
		    #r_transformed = pca.transform(data2)            
		    #subcor = Corelations(patterns_1=s_transformed.reshape(1,s_transformed.shape[0], s_transformed.shape[1]), patterns_2=r_transformed.reshape(1,r_transformed.shape[0],r_transformed.shape[1]), in_columns= False)
		    #pairweised_corr_s_r_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]
		    #pairweised_corr_s_r_corresponding_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCorOrigOrig()[0]                
		    
		    
		    #subcor = Corelations(patterns_1= ca1_input, patterns_2= Ca3_Ca1.noisy_output[:,0], in_columns= False)
		    subcor = Corelations(patterns_1= ca1_input, patterns_2= data2.reshape(1,num_all_pat, Ca1Cells), in_columns= False)
		    pairweised_corr_s_r_complete_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]       
		    pairweised_corr_s_r_complete_corresponding_ca1['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCorOrigOrig()[0] 
		    '''
		'''
	########################## Pairwise corr CA3 first patterns ###################################### 
		    data1 = ca3_input[0].reshape(num_seq,seq_len,Ca3.cells)[:,1:,:].reshape(1,num_seq*(seq_len-1), Ca3.cells) #ca1_noisy_input_pattern[0,0].reshape(num_seq,seq_len,Ca3.cells)[:,1:,:].reshape(1,num_seq*(seq_len-1), Ca3.cells)
		    data2 = Ec_Ca3.noisy_output[0,0].reshape(1,num_seq,seq_len,Ca3.cells)[:,:,0] 
		    
		    subcor = Corelations(patterns_1= data1, patterns_2= data2, in_columns= False)
		    pairweised_corr_s_r_complete_first_ca3['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]       
		    pairweised_corr_s_r_complete_corresponding_first_ca3['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = allcorrdata[1,ca3models,0,:,:,0]


	########################## Pairwise corr CA3 all patterns ###################################### 
		    data1 = ca3_input
		    data2 = ca1_noisy_input_pattern[0,0].reshape(1, num_all_pat, Ca3.cells)
		    
		    subcor = Corelations(patterns_1= data1, patterns_2= data2, in_columns= False)
		    pairweised_corr_s_r_complete_all_ca3['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = subcor.getCor()[0,0]       
		    pairweised_corr_s_r_complete_corresponding_all_ca3['inputstatistics='+str(inputstatistics)+'ca3models='+str(ca3models)] = allcorrdata[1,ca3models,0]
		    '''


		'''
		fig = plt.figure(figsize = [7.5, 8])
		fig.subplots_adjust(left = .12, bottom = .08, right = .9, top = .9, wspace=0.2, hspace=0.5)
		ax_ind = [1,1]
		
		
		## grid cell examples
		grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],1], nrows_ncols = (10,10),axes_pad=0.05, aspect = 1 ,share_all= 1, cbar_mode = 'single', cbar_pad = 0.05)
		#Inpatterns_2d = Ec.patterns.reshape(int(np.sqrt(num_all_pat)),int(np.sqrt(num_all_pat)),1100)
		#Inpatterns_2d = Ec_Ca3.noisy_output[0,0].reshape(int(np.sqrt(num_all_pat)),int(np.sqrt(num_all_pat)),2500)
		#Inpatterns_2d = Ca3.input_stored.reshape(int(np.sqrt(num_all_pat)),int(np.sqrt(num_all_pat)),2500)
		Inpatterns_2d = ca1_input[0].reshape(int(np.sqrt(num_all_pat)),int(np.sqrt(num_all_pat)),4200)
		
		for i in range(100):
			grid[i].xaxis.set_visible(False)
			grid[i].yaxis.set_visible(False)
		#max_fire = np.max(np.concatenate((Ec.patterns[0,:, 0], Ec.patterns[0,:, Ec.modules[1]+1], Ec.patterns[0,:, Ec.modules[2]+1], Ec.patterns[0,:, Ec.modules[3]+1])))
		#min_fire = np.min(np.concatenate((Ec.patterns[0,:, 0], Ec.patterns[0,:, Ec.modules[1]+1], Ec.patterns[0,:, Ec.modules[2]+1], Ec.patterns[0, :,Ec.modules[3]+1])))
		s = grid[0].imshow(Inpatterns_2d[:,:,0], interpolation = 'none', origin = 'lower')#, vmin = min_fire, vmax = max_fire)
		for i in range(1,100):
		    grid[i].imshow(Inpatterns_2d[:,:,np.int32(np.random.rand()*1100)], interpolation = 'none', origin = 'lower')#, vmin = min_fire, vmax = max_fire)
		#grid[2].imshow(Inpatterns_2d[:,:,Ec.modules[2]+1], interpolation = 'none', origin = 'lower')#, vmin = min_fire, vmax = max_fire)
		#grid[3].imshow(Inpatterns_2d[:,:,Ec.modules[3]+1], interpolation = 'none', origin = 'lower')#, vmin = min_fire, vmax = max_fire)
		#fig.colorbar(s, cax = grid.cbar_axes[0], ticks = [0,1])
		#makeLabel(ax = grid[0], label = 'A', sci = 0 )	
		
		
		plt.show()
	    '''
	 
	       
	    #save_object(obj =allcorrdata , path= path2 + '/allcorrdata' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =Ec.getOrigVsOrig() , path= path2 + '/ECgetOrigvsOrig' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =ca3_temporal_corr , path= path2 + '/ca3_temporal_corr' + str(real) + '.txt', info=True, compressed=False)
	    
		#save_object(obj =ca3_any_corr_stored , path= path2 + '/ca3_any_corr_stored' + str(real) + '.txt', info=True, compressed=False)
		#save_object(obj =ec_any_corr_stored , path= path2 + '/ec_any_corr_stored' + str(real) + '.txt', info=True, compressed=False)
	    

	    #save_object(obj =pairweised_corr_s_r_complete_ca1 , path= path2 + '/pairweiseCA1' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =pairweised_corr_s_r_complete_corresponding_ca1 , path= path2 + '/pairweiseCA1corresponding' + str(real) + '.txt', info=True, compressed=False)
	    
	    #save_object(obj =pairweised_corr_s_r_complete_first_ca3 , path= path2 + '/pairweiseCA3first' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =pairweised_corr_s_r_complete_corresponding_first_ca3 , path= path2 + '/pairweiseCA3correspondingfirst' + str(real) + '.txt', info=True, compressed=False)
	    
	    #save_object(obj =pairweised_corr_s_r_complete_all_ca3 , path= path2 + '/pairweiseCA3all' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =pairweised_corr_s_r_complete_corresponding_all_ca3 , path= path2 + '/pairweiseCA3correspondingall' + str(real) + '.txt', info=True, compressed=False)
	    	    
	    
	    #save_object(obj =expl_var_ratio_ca3_pca_stored_patts , path= path2 + '/pcaCa3' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =expl_var_ratio_ca1_pca_stored_patts , path= path2 + '/pcaCa1' + str(real) + '.txt', info=True, compressed=False)
	    #save_object(obj =expl_var_ratio_ec_pca_stored_patts , path= path2 + '/pcaEc' + str(real) + '.txt', info=True, compressed=False)



	    #np.set_printoptions(threshold='nan')
