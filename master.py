import tensorflow as tf
import numpy as np
import random
import time
import santorini 
import nets as net
import M
#from toy_problems import toy_problem
#from toy_problems import playback_game

import multiprocessing
import threading
from time import sleep

####################
# OVERALL SETTINGS #
####################
board_size = 5
search_depth = int(input("Search depth?\n"))
v_resign = 0.75

observe_games = int(input("Observe games?"))
explore = True
load_model = bool(int(input("Load model? 0/1\n")))
parallell = int(input("Asynchronous? 0/1\n"))
save_dir = input("Directory for saving weights:\n") 

####################
# HELPER FUNCTIONS #
####################

def update_target_graph(from_scope,to_scope):
    '''Copied from Arthur Juliani's amazing tutorials: https://medium.com/@awjuliani'''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def r(s, z, pi_s, legal_moves):
    '''Randomly rotating net input for more robust training due to less spurious correlations'''
    flip = np.random.choice([0, 1, 2])
    n = np.random.choice([1, 2, 3])
    
    inputs = [s, np.reshape(pi_s, [-1,board_size,board_size]), np.reshape(legal_moves, [-1,board_size,board_size])]
    inputs = [np.rot90(x, n, (1,2)) for x in inputs]
     
    if flip:
        inputs = [np.flip(x, flip) for x in inputs]
  
    s = inputs[0]
    pi_s = np.reshape(inputs[1], [-1, board_size**2])
    legal_moves = np.reshape(inputs[2], [-1, board_size**2])   
                
    return s, z, pi_s, legal_moves

class train_set(): 
    '''Storing self-play games for training'''
    def __init__(self, size = 5000):
        self.history = []
        self.size = size
        
    def add(self, s, pi, z, legal_moves):
        if len(self.history) + 1 >= self.size:
            self.history[0:(1+len(self.history))-self.size] = []
        self.history.extend([[s, pi, z, legal_moves]])
            
    def sample(self,batch_size):
        idxes = random.sample(range(len(self.history)), batch_size)
        s = np.stack([self.history[:][idx][0] for idx in idxes], 0)
        pi_s = np.stack([self.history[:][idx][1] for idx in idxes], 0)
        z = np.expand_dims(np.stack([self.history[:][idx][2] for idx in idxes], 0), 1)
        legal_moves = np.stack([self.history[:][idx][3] for idx in idxes], 0)
        return s, pi_s, z, legal_moves
    
coord = tf.train.Coordinator()


class coordinator():
    def __init__(self, condition):
        # We only want to use the coordinator is the model is run in parallell
        self.condition = condition
        
    def __call__(self, f):
        if self.condition:
            def wrapper(*args, evaluation=False, first_step=False):
                i = 0
                while not coord.should_stop():
                # Keeps training/self-playing indefinitely until the evaluation is completed
                    z = f(*args)
                    if evaluation:
                        return z
                    else:
                        print("Just finished self-play game: %s" % i)
                        i += 1
                        if first_step and i == 10:
                            coord.request_stop()
                print("Stop requested!")
            return wrapper
        return f
P1 = None
P2 = None

@coordinator( parallell )
def self_play(storage, player1, player2=None, explore=True, num_games=1):
    for n in range(num_games):
        if not parallell:
            print("Self-play game: %s" % n)
        game = santorini.Game() 
    
        p1 = M.MCTS(game, player1, sess, explore) 
        
        if player2 != None:
            evaluation = True
            p2 = M.MCTS(game, player2, sess, explore) 
            players = [p1, p2]
            global P1
            P1 = p1
            global P2
            P2 = p2
        else:
            evaluation = False
                   
        temp_history = []
    
        done = False
    
        while done == False:
            if evaluation:
                player = game.turn_count%2    
                tree = players[player]
                other_tree = players[(player+1)%2]
            else:
                tree = p1
                
            t0 = time.time()
            done = tree.consider_resigning(v_resign, observe_games)  
            a, pi_s, P = tree.run_simulation(search_depth)  #here's a difference
            temp_history.extend([[game.stack_s(), pi_s, game.legal_moves(binaryV=True)]])
            if evaluation:
                other_tree.prepare_adversarial_move(a)
            game.move(a)
            done = game.done
            if evaluation:
                other_tree.finish_adversarial_move(a)
            tree.prepare_next_move()
    
            if observe_games:
                for i in range(10):
                    print("\n")
                print("P (predicted tree search probs):\n%s\n\n" % np.reshape(P, [5,5]),
                      "pi (actual tree search probs):\n%s\n\n" % np.reshape(pi_s, [5,5]),
                      "Chosen move: %s\n" % a,
                      "Overall game state:\n%s\n\n" % game.render())
                print("time: ", time.time()-t0)
        z = game.outcome
        
        #store data
        for entry in temp_history:
            storage.add(entry[0], entry[1], z, entry[2])
        
    return z
  
@coordinator( parallell )
def train(history, network, epochs=1):
    print("Training...")
    for epoch in range(epochs):    
        try:
            s0, pi_s0, z0, legal_moves0 = history.sample(500)
        except:
            try:
                s0, pi_s0, z0, legal_moves0 = history.sample(100)
            except:
                print("no training, history is too empty")
                if parallell:
                    sleep(60)
                return None
    
        losses = []
        for t in range(8): # MVP of exploiting eight possible symmetries
            s, z, pi_s, legal_moves = r(s0, z0, pi_s0, legal_moves0)
            loss = network.train(s, z, pi_s, legal_moves, sess)
            losses.append(loss)
            
        print("loss: ", np.mean(losses))  



def evaluator(storage, challenger, champion, coord, sess, num_games=10):
    current_streak = 0
    players = [challenger, champion]
    outcomes = [-1, 1]
    for i in range(num_games):
        #Randomize who goes first
        coinflip = np.random.choice(2)
        player1 = players[coinflip]
        player2 = players[(coinflip+1)%2]
        if parallell:
            z = self_play(storage, player1, player2, False, evaluation=True)
        else:
            z = self_play(storage, player1, player2, False)
            
        if z == outcomes[coinflip]:
            current_streak += 1         
        print("Challenger's current streak %s" % current_streak)
        if current_streak >= 0.6*num_games:
            print("We have a new champion! Updating weights...")
            # Tell self-play and optimization to stop in order to be restarted 
            # with the new champion
            if parallell:
                coord.request_stop()
                # Crown the new champion by copying its weights to the past champion
                sess.run(update_target_graph("Test_challenger", "Champion"))
                # Make room for the new contender
                sess.run(update_target_graph("Challenger", "Test_challenger"))
            else:
                sess.run(update_target_graph("Challenger", "Champion"))
            break
        if num_games-1-i+current_streak < 0.6*num_games:
            break
        
    #reset the challenger in case the new weights didn't beat the champion
    if current_streak < 0.6*num_games:
        print("Challenger couldn't meet the challenge... resetting weights to previous champion...")
        if parallell:
            #Request that the other threads stop and wait for them to do so
            coord.request_stop()
            #Copy the champion to train to a new challenger
            sess.run(update_target_graph("Champion", "Challenger"))
            #Make room for the waiting challenger
            sess.run(update_target_graph("Challenger", "Test_challenger"))
        else:
            sess.run(update_target_graph("Champion", "Challenger"))
            
            
 
history = train_set()

game_history = []  #can do lots with this in order to store and generate games easily!

annealing_schedule = {0:0.01, 1:0.01, 2:0.01, 3:0.001, 4:0.0001}

print("Setting up network...")

tf.reset_default_graph()

champion = net.deep_net(False, "Champion", 0.01)
#Using both test and normal challenger to prevent issues that arise when training
#a network that's simultaneously in use
challenger = net.deep_net(False, "Challenger", 0.01)
test_challenger = net.deep_net(False, "Test_challenger", 0.01)
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Champion'))
   
stage = 0 #math.ceil(step/1000)
challenger.lr = annealing_schedule[stage]

steps = 20

with tf.Session() as sess:
    if load_model == True:
        print("Restoring model...")
        saver.restore(sess, save_dir+"champion.ckpt") 
        update_target_graph("Champion", "Challenger")
        update_target_graph("Champion", "Test_challenger")
          
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        print("Step: ", step)

        if not(parallell):
           
           
           print("Starting self-play...")
           self_play(history, challenger, num_games=10)
           
           train(history, challenger, epochs=20)
           
           print("Evaluating new challenger...")
           evaluator(history, challenger, champion, coord, sess, num_games=5)

           
           print("Saving...")
           saver.save(sess, save_dir+"champion.ckpt")
            
        else:
            num_workers = multiprocessing.cpu_count()
            print("Utilizing %s CPUs" % num_workers)
            
            def create_thread(worker):
                t = threading.Thread(target=(worker))
                t.start()
                global worker_threads
                worker_threads.append(t)
                sleep(0.1)            
                    
                
            # Currently an issue here, as it gets stuck training and never proceeds beyond step 0
            # maybe train just has to receive coord as input?
    
                worker_threads = [] 
                if step == 0: first_step = True
                else: first_step = False
                print(first_step)
                for i in range(num_workers-int(not(first_step))):
                    create_thread( lambda: self_play(history, champion, first_step=first_step) )
                create_thread( lambda: train(history, challenger))
                if step > 0:
                    create_thread( lambda: evaluator(history, test_challenger, champion, coord, sess, worker_threads) )
                coord.join(worker_threads) #maybe put this inside evaluator?
                print("Saving current champion...")
                saver.save(sess, save_dir+"champion.ckpt")
                coord.clear_stop()
    
