import numpy as np
import math
import copy
import warnings

####################
# HYPERPARAMETERS  #
####################
# For the Dirichlet distribution adding noise to moves probabilties for increased exploration
alpha = 0.03
eps = 0.25

# Used to balance exploration and exploitation
c_puct = 5 # from Silver et al. (2016)
# Softmax temperature for final move selection 
tau = 1    # from Silver et al. (2017) 

#Other
board_size = 5
debug = False # If true, prints an online description of the tree search
without_net = False # If false, naively assigns a uniform prior to moves and 0 evaluation
                    # to non-final board states. Can be used to compare performance with or without network.


####################
# HELPER FUNCTIONS #
####################

def r(s, pre_n=-1): #MVP:ing rotations so missing some symmetries
    '''Rotate network input before each evaluation for increased robustness'''
    in_shape = np.shape(s)
    if in_shape[1] == board_size**2:
        s = np.reshape(s, [-1,board_size,board_size]) # in order to flip legal moves, which have different input dimensions than states
    if pre_n > 0:
        n = pre_n    
    else:    
        n = np.random.choice([1, 2, 3])
    
    s = np.rot90(s, n, (1,2))
        
    if in_shape[1] == board_size**s:
        s = np.reshape(s, [-1,board_size**2]) 
        
    assert in_shape == np.shape(s)
    
    return s, n 

def anti_r(rotated_P, n):
    '''Reverse rotation'''
    rotated_P = np.reshape(rotated_P, [-1, board_size, board_size])  # fix up naming conflict with P later
    rotated_P = np.rot90(rotated_P, -n, (1,2))
    rotated_P = np.reshape(rotated_P, [-1, board_size**2])
    return rotated_P

def c(obj):
    return copy.deepcopy(obj)

def legalV(nodes):
    if type(nodes) != list:
        preV = np.array([int(x in nodes.A) for x in range(board_size**2)])
        preV = np.expand_dims(preV, 0) 
    else:
        preV = np.stack([[int(x in nodes[i].A) for x in range(board_size**2)] for i in range(len(nodes))], 0)            
    return preV 

def add_zeros_at_illegal_moves(pi_s, game):
    x = np.zeros(shape=[pow(board_size,2)])
    np.add.at(x, game.legal_moves(), pi_s)
    return x
 
def uniform_over_A(node):
    return [1/len(node.A) if a in node.A else 0 for a in range(board_size**2)]    
    
########################
# EVALUATION FUNCTIONS #
#######################

def evaluate(node, net, sess):   
    s = np.expand_dims(node.stack_s, 0) 
    S, n = r(s)
    legal, n = r(legalV(node), n)
    if without_net:
        return uniform_over_A(node), 0 # Nuisance parameters, without_net is just used to test % of compute cost due to tf
    else:
        P, v = net.P_and_v(S, legal, sess)  # randomly rotate the input to the network      
        P = anti_r(P, n)[0,:]
        
        v = v[0,0]

    if np.sum(np.isnan(P)) > 0:
        warnings.warn("Beware, the policy head outputs NaNs...\n Check the usual suspects" \
                      "to make sure gradients aren't exploding: clipping, learning rate,"   \
                      "batch norm settings")
        print(np.reshape(P, [5,5]))
        P = uniform_over_A(node)

    elif np.abs(np.sum(P)-1) > 0.01: # Ensure P is a density function, with some margin for error
        warnings.warn("Beware, the policy head does not output a proper distribution..." \
                      "renormalizing for now")
        print(np.reshape(P, [5,5]))
        print(np.sum(P))
        P = P/np.sum(P)
    
    elif np.sum(np.nonzero(np.reshape(P, [25,]))[0] != np.array(node.A)) == len(node.A):
        warnings.warn("Beware, the policy head gives some legal moves 0 probability..." \
                       "adding uniform noise to handle it for now")
        print(np.reshape(P, [5,5]))
        P = np.dot(0.95,P) + np.dot(0.05,uniform_over_A(node))  
    
    return P, v

def N(node, a):         
    return node.get(a).N 

def Q(node, a):
    return node.get(a).Q

def P(node, a):
    return node.get(a).P

def N_tot(node, tau):       
    A = node.A
    total = 0
    for b in A:
        total += N(node,b)**(1/tau)
    return total

def U(node, a):   
    '''This function scores the exploratory value of moves, using a variant of the
       Upper Confidence Bound algorithm complemented with an optimized prior on move probability'''      
    N_total = math.sqrt(N_tot(node, 1))
    U = c_puct * P(node, a) * N_total/(1+N(node, a))
    return U

def pi(a, s_0, tau):     
    return (N(s_0, a)**(1/tau)) / N_tot(s_0, tau)

################################
# CONSTRUCTING THE SEARCH TREE #
###############################

class node():
    '''Each node stores a set of summary statistics '''
    def __init__(self, game, up_edge):  
        self.s = game.s()
        self.stack_s = game.stack_s() 
        self.A = game.legal_moves() 
        self.up_edge = up_edge
        self.leaf = 1
        self.P = []  
        self.v = 0
        self.game = game.render() # a snapshot of the game when the node was _created_
        self.depth = None #assigned in the process of traversing the tree
      
    def __repr__(self):
        return "Node:\n%s\nA=%s\nv=%s\n P=%s" % (self.game, self.A, self.v, self.P)
    
    def get(self, a):
        return getattr(self, 'a'+str(a))
    
    def expand(self, game):
        if self.leaf == 1:  #don't expand if we've already done so before
            for a in self.A:
                setattr(self, 'a'+str(a), edge(game, a, self))  
            self.leaf = 0
            
            if debug:
                print(self, "is no longer a leaf\n")
                
    def add_dirichlet_noise(self):
        if debug:
            print("Adding dirichlet noise")
        alphas = [alpha if i in self.A else 0 for i in range(25)] #change to p later

        try: # sometimes low alphas cause problems 
            noise = np.random.dirichlet(alphas)
            self.P = np.dot(1-eps,self.P) + np.dot(eps,noise)
        except:
            warnings.warn("Problems with low alphas in the Dirichlet function...")
            #just some uniform noise
            noise = uniform_over_A(self) 
            self.P = np.dot(1-eps,self.P) + np.dot(eps,noise)
            pass
        assert np.shape(self.P) == (board_size**2,)
        
class edge():
    '''Each edge (state, action) stores a set of summary statistics that are updated
       by traversing the tree'''
    def __init__(self, game, a, up_node):
        self.s = game.render()
        self.a = a
        self.N = 0
        self.W = 0
        self.Q = 0 
        self.P = 1/25 # P is initialized uniformly, and updated as soon as a network evaluation is run
        self.up_node = up_node
        self.down_node = None 
    
    def __repr__(self):
        return "Edge:\ns=%s\na=%s\nN=%s, W=%s, Q=%s, P=%s" % (self.s, self.a, self.N, self.W, self.Q, self.P)
        
    def add_backup_stats(self, v):        
        self.N += 1
        self.W += v
        self.Q = self.W/self.N
    
    def create_down_node(self, game):
        '''To save memory, nodes are only initialized upon being visited'''
        if self.down_node == None: # check whether node has not bene previously initialised
            self.down_node = node(game, self) 
        try:   # this fails in case the search starts from a root without a parent node
            self.down_node.depth = self.up_node.depth+1
        except:
            pass
        
class MCTS():
    '''Main class that generates the tree data structure and traverses it, 
       guided by the deep networks for estimating a prior on moves and the 
       value of a given state. For details, see Silver et al. (2017), https://www.nature.com/articles/nature24270'''
       
    def __init__(self, game, net, sess, explore=True):
        self.game = game # The search manipulates the actual game object rather than a copy of it. 
        self.root = node(self.game, None)
        self.backup_v = 0
        self.depth = 0
        self.net = net
        self.move = None
        self.pi_s = None
        self.sess = sess
        self.explore = explore
        self.k = 0 #search depth, assigned by calling the run simulation function below
        
    def __repr__(self):
        return "Tree search currently inspecting state:\n%s\nRuns left=%s\nCurrent depth=%s" % (self.game.render(), self.k, self.depth)
        
    def prepare_next_move(self):
        # Saving memory by removing unnecessary tree branches that will now never be visited
        a = self.move
        self.clear_unnecessary_branches()
            
        # reassign the current game state as new root
        self.root.get(a).create_down_node(self.game) 
        self.root = self.root.get(a).down_node 
        if debug:
            print("assigning node ", a, self.root, " as new root")
                           
    def prepare_adversarial_move(self, a): 
        # Like prepare_next_move but for an opponent move. Completed by finish_adversarial_move 
        self.move = a      
        try:          
            self.root.expand(self.game) 
            self.clear_unnecessary_branches()    
            if debug:
                print("assigning node ", a, self.root, " as new root")
        except:  # occurs if there are no available moves
            pass
     
    def finish_adversarial_move(self, a): 
        # A bit untidy that the function has to split up like this, should fix in future version!
        self.root.get(a).create_down_node(self.game) 
        self.root = self.root.get(a).down_node 
                                 
    def clear_unnecessary_branches(self):
        A = c(self.root.A)
        A.remove(self.move)
        for b in A:
            delattr(self.root, 'a'+str(b))
        if debug:
            print("removing subtrees under moves ", A)
                
    def consider_resigning(self, v_resign, observe_games=False):
        ''' Resign game if chance of winning below v_resign'''
        '''TO-DO: continually evaluate false-positive rate alpha and set v_resign 
           such that alpha = 0.05'''
        i = self.game.outcomes[self.game.player]
        if i*self.root.v < -v_resign:
            if observe_games:
                print("Player ", self.game.player, " resigned")
            self.game.outcome = -i
            self.game.done = True
            
        return self.game.done
        
                           
    def backup(self, node, k, use_net):
        
        #Get values to be passed upwards
        if use_net:
            node.P, node.v = evaluate(node, self.net, self.sess)
            for a in node.A:
                node.get(a).P = node.P[a]
            self.backup_v = node.v
        else:
            self.backup_v = self.game.outcome 
            
        #Pass values upwards    
        while self.depth > 0:
            node.up_edge.add_backup_stats(self.backup_v) 
            self.depth -= 1
            if debug:
                print("evaluating position and passing edge statistics upwards\n"
                      "v =", self.backup_v, "\nrunning simulation from parent node\n")
            node = node.up_edge.up_node                        

        k -= 1
        self.game.reset(c(self.root.s))
        node = self.root
        return node, k
    

    
    def run_simulation(self, k):
        '''This function runs a tree search of depth k and returns the selected move, 
           the predicted search probabilities, the actual search probabilties, and
           the state evaluation. All other methods in the MCTS class are just helper methods'''
        self.k = k
        node = self.root
        node.depth = 0
        node.expand(self.game)
        
        # Get initial search probabilities and outcom expectation (used to consider resigning)
        if len(node.P) == 0:
            node.P, node.v = evaluate(node, self.net, self.sess) 

        if self.game.turn_count < 15: #Increased exploration
            node.add_dirichlet_noise()
        
        #Go down the tree k times
        while self.k >= 0:    
            assert self.game.turn_count%2 == self.game.player
            
            moves = self.game.legal_moves() 
            
            if debug:
                print(self, "\n\n")
            
            if self.game.outcome != None: # game is over 
                node, self.k = self.backup(node, self.k, use_net=0) # We don't use the net as
                if debug:                                           # an actual end result is available
                    print("Adding statistics to: ", node.up_edge, "\n\n")   
            else:    
                if self.k == 0: #simulation is over
                    pi_s = [pi(move, node, 1) for move in moves] #proportional visit counts                  
                    
                    if not(self.explore) or self.game.turn_count >= 15: # explore at the start of games, then play as strong as possible
                        move = moves[np.argmax(pi_s)]
                    else:
                        move = np.random.choice(moves, 1, p=pi_s)[0]

                    self.move = move
                    self.pi_s = add_zeros_at_illegal_moves(pi_s, self.game)
                    
                    return self.move, self.pi_s, self.root.P, self.root.v
                    
                else: # simulation is not over
                    if node.leaf: 
                        #expand
                        node.expand(self.game)                        
                        #evaluate        
                        node, self.k = self.backup(node, self.k, use_net=1) # We have to use nets to estimate
                    else:                                                   # the value of the current state, 
                        #select                                             # as an outcome cannot yet be seen
                        i = self.game.outcomes[self.game.player] # Ensures that positive Q always means winning (-1 is player 0, 1 if player 1)
                        idx = np.argmax( [i*Q(node,a) + U(node,a) for a in node.A] )
                        a = moves[idx]
                        self.game.move(a)
                        
                        node.get(a).create_down_node(self.game)
                        new_node = node.get(a).down_node
                        node = new_node
                        self.depth += 1

                        if debug:
                            print("Selected move ", a, " going down ", node.up_edge, "\n\n")
