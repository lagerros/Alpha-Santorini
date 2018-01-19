import tensorflow as tf
import numpy as np
import random
import tensorflow_rename_variables as trv
import time
import math
import santorini 
import nets as net
import M
#from toy_problems import toy_problem
#from toy_problems import playback_game
import os
import shutil
import time

####################
# OVERALL SETTINGS #
####################
board_size = 5
search_depth = 1600
testing = 1
training = 1
v_resign = 0.75

observe_games = 1
explore = True
load_model = False

parent_dir = "/home/jakey/Alpha/"
archive_dir = parent_dir + "Archive/"
current_dir = parent_dir + "Current/" 



####################
# HELPER FUNCTIONS #
####################

def copy_rename(src_dir, dst_dir, old_file_name, new_file_name):
#        src_dir= os.curdir
#        dst_dir= os.path.join(os.curdir , "subfolder")
        src_file = os.path.join(src_dir, old_file_name)
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        os.rename(dst_file, new_dst_file_name)

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
    def __init__(self, size = 25000):
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
        
history = train_set()


#############
# SELF-PLAY #
############

num_games = 2
steps = 5
game_history = []  #can do lots with this in order to store and generate games easily!

annealing_schedule = {0:0.01, 1:0.01, 2:0.01, 3:0.001, 4:0.0001}

for step in range(steps):
    print("Step: ", step)
    print("Setting up network...")
    
    tf.reset_default_graph()
    
    challenger = net.deep_net(False, "Challenger", 0.01)
       
    stage = math.ceil(step/1000)
    challenger.lr = annealing_schedule[stage]
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if load_model == True and step > 0:
            print("Restoring model...")
            copy_rename(current_dir, parent_dir, "champion.ckpt", "challenger.ckpt")
            trv.rename(parent_dir+"challenger.ckpt", "Champion", "Challenger")
            saver.restore(sess, parent_dir+"champion.ckpt") 
        
            
        print("Starting self-play...")
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("/tmp/test", graph=tf.get_default_graph())
        
        for i in range(num_games):
            print("game ", i)
            game = santorini.Game() 
            
            m = M.MCTS(game, challenger, sess, explore)            
            temp_history = []

            done = False

            while done == False:
                t0 = time.time()
                t0_cpu = time.clock()
                done = m.consider_resigning(v_resign, observe_games)  
                a, pi_s, P = m.run_simulation(search_depth)
                temp_history.extend([[game.stack_s(), pi_s, game.legal_moves(binaryV=True)]])
                game.move(a)
                done = game.done
                m.prepare_next_move()

                if observe_games:
                    for i in range(10):
                        print("\n")
                    print("P (predicted tree search probs):\n", np.reshape(P, [5,5]),
                          "pi (actual tree search probs): %s", pi_s,
                          "Chosen move: ", a,
                          "Overall game state:\n",
                          game.render())
                print("time: %s", time.time()-t0, time.clock()-t0_cpu)

            z = game.outcome

            #store data
            for x in temp_history:
                history.add(x[0], x[1], z, x[2])   
        
        #############
        # TRAINING #
        ############
        
        if training:
            print("Training...")
            for x in range(10):
                try:
                    s0, pi_s0, z0, legal_moves0 = history.sample(500)
                except:
                    s0, pi_s0, z0, legal_moves0 = history.sample(2)
                for t in range(8): # MVP of exploiting eight possible symmetries
                    s, z, pi_s, legal_moves = r(s0, z0, pi_s0, legal_moves0)
                    loss = challenger.train(s, z, pi_s, legal_moves, sess)
                    print("loss: ", loss[0])  
        
#            save_path = saver.save(sess, current_dir+"challenger.ckpt")
#            if step == 0:
#                saver.save(sess, current_dir+"champion.ckpt")  
#                trv.rename(current_dir+"champion.ckpt", "Challenger", "Champion", add_prefix=False, dry_run=False)
            
#            print("Model saved in file: %s" % save_path)
#            #trv.rename("/home/jakey/Alpha/current.ckpt", "Best", "Current", add_prefix=False, dry_run=False)
                
    #############
    # TESTING  #
    ############
    if testing:
        num_test_games = 2
        current_streak = 0
        explore = False
        
        tf.reset_default_graph()
        champion = net.deep_net(False, "Champion", 0)
        
        
        with tf.Session() as sess:
        #    saver.restore(sess, current_dir+"champion.ckpt") 
            print("Starting testing vs previous champion...")
            time.sleep(3)
            sess.run(tf.global_variables_initializer())                 
            
            for i in range(num_test_games):
                print(i)
                game = santorini.Game()
                done = False
                coinflip = np.random.choice([-1, 0, 1, 2, 4, 5])
                game = toy_problem(game, coinflip)
                m_current = M.MCTS(game, challenger, sess, explore)  #alternate who goes first
                m_best = M.MCTS(game, champion, sess, explore)
                players = [m_current, m_best]
                
                while done == False:
                    player = game.turn_count%2    
            
                    tree = players[player]
                    other_tree = players[(player+1)%2]
                    
                    done = tree.consider_resigning(v_resign, observe_games)
                    tree.run_simulation(search_depth)
                    a = tree.move
                    other_tree.prepare_adversarial_move(a)
                    game.move(a)
                    done = game.done
                    other_tree.finish_adversarial_move(a)
                    tree.prepare_next_move()
                    
                    if observe_games:
                        print("\n\n\n\n")
                        print("Overall game state:")
                        print(game.render())
                if game.outcome == -1:
                    current_streak += 1            
                             
            if current_streak/num_test_games > 0.55:
                print("We have a new champion! Saving and renaming...")
                key = str(np.random.choice(1000))
                champion_saver = tf.train.Saver(var_list=
                                                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Champion'))
                champion_saver.save(sess, archive_dir+key+".ckpt")

                challenger_saver = tf.train.Saver(var_list=
                                                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Challenger'))
                challenger_saver.save(sess, current_dir+"champion.ckpt")

           
        tf.reset_default_graph()
        
        if current_streak/num_test_games > 0.55:
            trv.rename(current_dir+"champion.ckpt", "Challenger", "Champion", add_prefix=False, dry_run=False)
  
