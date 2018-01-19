import numpy as np
import math
#from profilestats import profile
import copy

c_puct = 5 # from 2016 paper
tau = 1    # from 2017 paper

board_size = 5
debug = 0
without_net = 0 # can be used to compare performance with or without network

####################
# HELPER FUNCTIONS #
####################

def r(s, pre_n=-1): #MVP:ing rotations so missing some symmetries
    '''Rotate network input before each evaluation for increased robustness'''
    in_shape = np.shape(s)
    if in_shape[1] == 25:
        s = np.reshape(s, [-1,5,5]) # in order to flip legal moves, which have different input dimensions than states
    if pre_n > 0:
        n = pre_n    
    else:    
        n = np.random.choice([1, 2, 3])
    
    s = np.rot90(s, n, (1,2))
        
    if in_shape[1] == 25:
        s = np.reshape(s, [-1,5**2]) 
        
    assert in_shape == np.shape(s)
    
    return s, n 

def anti_r(p, n):
    '''Reverse rotation'''
    p = np.reshape(p, [-1, 5, 5])  # fix up naming conflict with P later
    p = np.rot90(p, -n, (1,2))
    p = np.reshape(p, [-1, 5**2])
    return p   

def c(obj):
    return copy.deepcopy(obj)

def legalV(nodes):
    if type(nodes) != list:
        preV = np.array([int(x in nodes.A) for x in range(board_size**2)])
        preV = np.expand_dims(preV, 0) 
    else:
        preV = np.stack([[int(x in nodes[i].A) for x in range(board_size**2)] for i in range(len(nodes))], 0)            
    return preV 

def softmax(x):  # gotten from here: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def add_zeros_at_illegal_moves(pi_s, game):
    x = np.zeros(shape=[pow(board_size,2)])
    np.add.at(x, game.legal_moves(), pi_s)
    return x
     
    
########################
# EVALUATION FUNCTIONS #
#######################

def evaluate(nodes, net, sess):   
    if type(nodes) != list:
        nodes = [nodes]
    s = np.stack([nodes[i].stack_s for i in range(len(nodes))], 0)
    S, n = r(s)
    legal, n = r(legalV(nodes), n)
    if without_net:
        P, v = legalV(nodes[0]), -0.2 # 
    else:
        P, v = net.P_and_v(S, legal, sess)  # randomly rotate the input to the network      
        P = anti_r(P, n)

        if len(nodes) == 1:
            v = v[0,0]
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
    N_total = math.sqrt(N_tot(node, 1))
    U = c_puct * P(node, a) * N_total/(1+N(node, a))
    return U

def pi(a, s_0, tau):     
    return (N(s_0, a)**(1/tau)) / N_tot(s_0, tau)

##  BUILDING A TREE

class node():
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
        if self.leaf == 1:  #don't expand if you've already done so before
            for a in self.A:
                setattr(self, 'a'+str(a), edge(game, a, 0.3, self))  #this 0.3 is irrelevant and shuold be removed!
            self.leaf = 0
            
            if debug:
                print(self, "is no longer a leaf\n")
                
    def add_dirichlet_noise(self):
        eps = 0.25
        alphas = [0.03 if i > 0 else 0 for i in self.P] #change to p later
        try: # sometimes low alphas cause problems 
            noise = np.random.dirichlet(alphas)
            self.P = (1-eps)*self.P + eps*noise
        except:
            pass
        assert np.shape(self.P) == (board_size**2,)
        
class edge():
    def __init__(self, game, a, P, up_node):
        self.s = game.render()
        self.a = a
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = P
        self.up_node = up_node
        self.down_node = None 
    
    def __repr__(self):
        return "Edge:\ns=%s\na=%s\nN=%s, W=%s, Q=%s, P=%s" % (self.s, self.a, self.N, self.W, self.Q, self.P)
        
    def add_backup_stats(self, v):        
        self.N += 1
        self.W += v
        self.Q = self.W/self.N
    
    def create_down_node(self, game):
        if self.down_node == None:
            self.down_node = node(game, self) 
        try:   # root node has no up_node
            self.down_node.depth = self.up_node.depth+1
        except:
            pass
        
class MCTS():
    def __init__(self, game, net, sess, explore=True):
        self.game = game # currently inspected game
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
        # removing unnecessary tree branches that will now never be visited
        a = self.move
        self.clear_unnecessary_branches()
            
        # reassign the current game state as new root
        self.root.get(a).create_down_node(self.game) 
        self.root = self.root.get(a).down_node 
        if debug:
            print("assigning node ", a, self.root, " as new root")
                           
    def prepare_adversarial_move(self, a): 
        # removing unnecessary tree branches that will now never be visited  
        self.move = a      
        try:          
            self.root.expand(self.game) 
            self.clear_unnecessary_branches()    
            if debug:
                print("assigning node ", a, " as new root")
        except:  # occurs if there are no available moves
            pass
     
    def finish_adversarial_move(self, a):   
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
            P, v = evaluate(node, self.net, self.sess)
            node.P = P[0,:]
            if not(without_net):
                assert np.shape(node.P) == (board_size**2,)
                assert np.abs(np.sum(node.P)-1) < 0.01 # Ensure P is a density function, with some margin for error
            for a in node.A:
                node.get(a).P = node.P[a]
            node.v = v
            self.backup_v = v
        else:
            self.backup_v = self.game.outcome 
            
        #Pass values upwards    
        while self.depth > 0:
            node.up_edge.add_backup_stats(self.backup_v) # add backup stats (should make it clearer that's what's happening)
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
        self.k = k
        node = self.root
        node.depth = 0
        node.expand(self.game)
        
        # Get initial search probabilities
        if len(node.P) == 0:
            if without_net:
                v, temp_P = 0, legalV(node) 
                node.v = v 
            else:
                temp_P, v = self.net.P_and_v(np.expand_dims(node.stack_s,0), legalV(node), self.sess) 
                node.v = v[0,0]

            node.P = temp_P[0]
            if not(without_net):
                assert np.shape(node.P) == (board_size**2,)
                assert np.abs(np.sum(node.P)-1) < 0.01 # Ensure P is a density function, with some margin for error

        assert np.sum(np.nonzero(node.P)[0] == node.A) == len(node.A)

        if self.game.turn_count < 15: #Increased exploration
            node.add_dirichlet_noise()
        
        #Go down the tree k times
        while self.k >= 0:    
            assert self.game.turn_count%2 == self.game.player
            
            moves = self.game.legal_moves() 
            
            if debug:
                print(self, "\n\n")
            
            if self.game.outcome != None: # game is over 
                node, self.k = self.backup(node, self.k, use_net=0)
                if debug:
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
                    
                    return self.move, self.pi_s, self.root.P
                    
                else: # simulation is not over
                        
                    if node.leaf: # or len(node.P) == 0:
                        #expand
                        node.expand(self.game)
                        node.leaf = 0
                        
                        #evaluate        
                        node, self.k = self.backup(node, self.k, use_net=1)
                        
                        if debug:
                            print("This is a leaf, so expanding. Node after evaluation:\n", node)
                            
                    else:
                        #select
                        i = self.game.outcomes[self.game.player] # Ensures that positive Q always means winning (-1 is player 0, 1 if player 1)
                        idx = np.argmax( [i*Q(node,a) + U(node,a) for a in node.A] )
                        a = moves[idx]
                        self.game.move(a)
                        
                        node.get(a).create_down_node(self.game)
                        new_node = node.get(a).down_node
                                           
                        self.depth += 1
                        
                        node = new_node

                        if debug:
                            print("Selected move ", a, " going down ", node.up_edge, "\n\n")
