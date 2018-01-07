import numpy as np
import copy

def _coord_to_action(x, y):
    return int(x+5*y)

def _action_to_coord(a):
    x = int(a%5)
    y = int((a-x)/5)
    return x, y

def _create_space(xy):
    x, y = _action_to_coord(xy)
    return space(x, y)

def c(obj):
    return copy.deepcopy(obj)

class Game():
    def __init__(self):
        ''' Create space objects that hold buildings and workers'''
        for a in range(25):
            self.set_a(a)  
            
        ''' Player to move next '''
        self.player = 0
        
        ''' Workers '''
        self.workers = []
        self.selected_worker = None  
        
        '''
        Monitors the current stage of each players turn
        0: Place worker
        1: Select worker
        2: Select space to move to
        3: Select space to build in
        '''
        self.stage = 0 
        
        ''' Whether game is completed, and who won '''
        self.done = False
        self.outcome = None
        self.outcomes = [-1, 1]
        self.turn_count = 0
        
    def s(self):
        ''' A deep copy of the entire game object, 
            in order to be able to reset the game to a previous turn'''
        return c(self)
    
    def set_a(self, a):
        return setattr(self, 'a'+str(a), _create_space(a))
    
    def get(self, a):
        return getattr(self, 'a'+str(a))  
      
    def legal_moves(self, binaryV=False):
        
        def clear(area, condition, is_set=False):
            if is_set:
                cleared_area = [xy for xy in area if not( len(set(xy).intersection(condition))>0 )]
            else:
                cleared_area = [xy for xy in area if not( xy in condition )]
            return cleared_area # in order to handle both items and sublists
#                
        number_board = [self.get(a).height if self.get(a).inhabited == False else 5 for a in range(25)]
        
        if self.stage == 0:
            A = [a for a in range(25) if number_board[a] < 5] 
            
        elif self.stage == 1:
            workers = [self.workers[0+self.player], self.workers[2+self.player]]
            A = [w.a() for w in workers]
            
        else: 
            # Create a list of nearby spaces in all directions... 
            offsets = [-1, 0, 1]
            x, y = self.selected_worker.x, self.selected_worker.y
            surrounding = [[x+i, y+j] for i in offsets for j in offsets]
            
            # ...not containing illegal moves
            # remove starting coords
            surrounding.remove([x, y])
            
            # remove coords too far out
            surrounding = clear(surrounding, [-1, 5], True) 
            surrounding = [_coord_to_action(xy[0], xy[1]) for xy in surrounding]
            
            # remove workers and domes
            workers_and_domes = [a for a in range(25) if number_board[a] >= 4]
            A = clear(surrounding, workers_and_domes) 

            if self.stage == 2:
                # remove buildings too high to climb onto
                height = self.selected_worker.space.height
                too_high = [a for a in range(25) if number_board[a] >= height+2]
                
                A = clear(A, too_high) 
                
        A = sorted(A)
        
        if binaryV:
            A = [1 if a in A else 0 for a in range(25)]
            
        return A
            
    def move(self, a):             
        if self.stage == 0:     #placing the workers before the game begins
            n = self.num_workers_left_to_place()-1
            names = ['A', 'B']
            self.selected_worker = worker(self.player, self.get(a), names[self.player]+str(n))
            self.workers.append(self.selected_worker)
            self.get(a).place(self.selected_worker)
            
            if n == 0:
                self.stage = 1 #proceed to next stage
            
            self.player = (self.player+1)%2
            self.turn_count += 1
            
        else:
            if self.stage == 1: #select worker
                self.selected_worker = self.get(a).inhabitant
                self.stage += 1  
                
                #check whether game is over
                if len(self.legal_moves()) == 0:
                    self.done = True                                
                    if self.player == 0:
                        self.outcome = 1
                    else:
                        self.outcome = -1
                    
           
            elif self.stage == 2: #move
                self.get(self.selected_worker.a()).remove_worker() # remove from past space
                self.get(a).place(self.selected_worker)          # move into new space
                
                self.stage += 1
        
                #Check if game is completed
                if self.get(a).height == 3: # maybe a bit unfortunate to have to check legal moves here?
                    self.done = True                                # or is this still too far from being a major time-constraint?
                    if self.player == 0:
                        self.outcome = -1
                    else:
                        self.outcome = 1

            elif self.stage == 3: #build
                self.get(a).build()
                
                self.stage = 1
                self.player = (self.player+1)%2
                self.turn_count += 1
                self.selected_worker = None
        
        return self, self.outcome, self.done
                        
    def render(self):
        sprites = {0:'     ', 1:'.....', 2:':::::', 3:'|||||', 4:'||D||'}
        out = np.array(['    ' for x in range(25)])
        
        for a in range(25):
            out[a] = sprites[self.get(a).height] 
            if self.get(a).inhabited == True:
                out[a] = out[a][:3] + self.get(a).inhabitant.name
                   
        if self.selected_worker != None:
            a = _coord_to_action(self.selected_worker.space.x, self.selected_worker.space.y)
            out[a] = 'X'+ out[a][1:]
        
        return np.reshape(out, [5,5])
    
    def num_workers_left_to_place(self):
        num_placed = len( [1 for a in range(25) if self.get(a).inhabited == True] )
        return 4-num_placed
    
    def reset(self, root):

        for a in range(25): 
            setattr(self, 'a'+str(a), root.get(a))
        self.player = root.player
        self.workers = root.workers
        self.selected_worker = root.selected_worker
        self.stage = root.stage
        self.done = root.done
        self.outcome = root.outcome
        self.turn_count = root.turn_count
    
    def stack_s(self):   
        '''Gets the current state in a multi-channel format to be input to the deep conv net'''
        number_board = [[self.get(a).height, self.get(a).inhabited, self.get(a).inhabitant] for a in range(25)] 
        
    #    Binary planes with positions of bulidings of height 1, 2, 3 or dome
        X = [np.reshape([1 if number_board[i][0] == height else 0 for i in range(25)], [5,5]) 
                                                                  for height in range(1,5)]     
    #    Binary planes with positions of player 1 and 2 workers
        Y = [np.reshape([1 if number_board[i][1] and number_board[i][2].player==player else 0 for i in range(25)], [5,5]) 
                                                                                              for player in range(2)]
    #   Player indication plane
        if self.player == 1:
            P = np.ones_like(X[0]) #X_1)
        else:
            P = np.zeros_like(X[0])  #X_1)
            
    #   Binary plane with position of selected worker
        if self.selected_worker == None: #n == 0:
            W = np.reshape([0 for i in range(25)], [5,5])
        else:
            a = self.selected_worker.a()
            W = np.reshape([1 if i == a else 0 for i in range(25)], [5,5])
            
    #   Four binary planes indicating the stage of the turn
        stages = [np.ones_like(X[0]) if self.stage == i else np.zeros_like(X[0]) for i in range(4)]
                
        return np.stack([X[0], X[1], X[2], X[3], Y[0], Y[1], P, W, stages[0], stages[1], stages[2], stages[3]], -1) #X_1, X_2, X_3, X_d, Y_0, Y_1, P, W, s_0, s_1, s_2, s_3],-1)


class space():
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.height = 0
        self.dome = False
        self.inhabited = False
        self.inhabitant = None
        
    def build(self):
        assert self.height < 4 # Illegal to build on top of domes
        self.height += 1
        if self.height == 4:
            self.dome = True
    
    def place(self, worker):
        assert self.inhabited == False # Illegal to place worker where there is already another worker
        self.inhabited = True
        self.inhabitant = worker
        worker.space = self
        worker.x, worker.y = self.x, self.y
    
    def remove_worker(self):
        self.inhabited = False
        self.inhabitant = None

class worker():
    def __init__(self, player, space, name):
        self.player = player
        self.space = space
        self.x, self.y = space.x, space.y
        self.name = name
    
    def a(self):
        return _coord_to_action(self.x, self.y)
