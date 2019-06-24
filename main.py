## Author: Amine Dassouli
## PyRat authors : Carlos Lassance, Nicolas Farrugia, Aymane Abdali
## The goal is to train an agent to play PyRat againt a greedy using Deep Reinforcement learning

""" Importations """
import tensorflow as tf
import numpy as np
from random import random, randint
from tqdm import tqdm
from AIs import manh
import matplotlib.pyplot as plt

## The game.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent
import game

## The rl.py file describes the reinforcement learning procedure, including Q-learning, Experience replay, and Stochastic Gradient Descent (SGD). SGD is used to approximate the Q-function
import rl

## The monte_carlo.py file describes the Monte Carlo technique to get a path, if it exists,  for the agent in order for him to win 
from monte_carlo import monte_carlo, distance


""" Definitions :
 - An epoch is an iteration of training, it corresponds to a full play of a PyRat game. 
 - A batch is a set of experiences we use for training during one epoch
 - An experience is a set of vectors < s, a, r, s’ > describing the consequence of being in state s, doing action a, receiving reward r, and ending up in state s' 
"""


""" Variables """
number_epochs = 200
number_of_batches = 12
batch_size = 32
max_memory = 2000

opponent = manh # AI used for the opponent
width = 21 # Size of the playing field
height = 15 # Size of playing field

model = rl.NLinearModels(2 * 1189, 4, batch_size) ## The neural network used for training the agent

## If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters
load = 0
save = 1

global winrate_pereps, drawrate_pereps
winrate_pereps, drawrate_pereps = [], []

## Initiate a game environment
env = game.PyRat()


""" Functions """
# This function is used to play games 
def play(sess, model, number_epochs, train = True):

    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0
    
    for epoch in tqdm(range(number_epochs)):
        # We generate a new environment 
        env.reset()
        game_over = False
        input_t = env.observe()
        
        if not train :
            exp_replay.decay = 1
        
        # We try to find an optimized path using Monte Carlo
        mc_path = monte_carlo(env.width, env.height, env.player, env.enemy , env.piecesOfCheese)
        if mc_path != [] :
            # If a path is found, we are going to follow it in order to choose the actions
            mc_path.pop(0)
            mc_path = [ (target[0], target[1]) for target in mc_path]
            while mc_path != [] :
                input_tm1 = input_t
                current_target = mc_path[0]
                if distance(current_target, env.player) <= 1 : 
                    mc_path.pop(0)
                if current_target[0] > env.player[0] :
                    action = 1
                elif current_target[0] < env.player[0] :
                    action = 0
                elif current_target[1] > env.player[1]:
                    action = 2
                elif current_target[1] < env.player[1]: 
                    action = 3
                
                # We apply the action on the state and get the reward and the new state
                input_t, reward, game_over = env.act(action)
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)
                
        else :  
            # If no path is found, we are just going to apply the classical deep reinforcement learning
            while not game_over :
                input_tm1 = input_t
                if random() < exp_replay.eps :
                    action = randint(0, model._num_actions - 1)
                else:           
                    q = model.predict_one(sess, input_tm1)
                    action = np.argmax(q[0])
                exp_replay.eps = exp_replay.min_eps + (exp_replay.max_eps - exp_replay.min_eps) \
                                      * np.exp(-exp_replay.decay * epoch)
                                      
                 # We apply the action on the state and get the reward and the new state
                input_t, reward, game_over = env.act(action)
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)
                
        # Statistics
        steps += env.round
        if env.score > env.enemy_score: 
            win_cnt += 1 
        elif env.score == env.enemy_score: 
            draw_cnt += 1 
        else:
            lose_cnt += 1 
        cheese = env.score 
            
        win_hist.append(win_cnt) 
        cheeses.append(cheese) 
        
        # Training the agent
        if train :
            for _ in range(number_of_batches):                
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
                model.train_batch(sess, inputs.reshape((32,-1)), targets.reshape((32,-1)))
                
        # Show statistics every 100 epochs
        if (epoch + 1) % 100 == 0 : 
            cheese_np = np.array(cheeses)
            string = "Epoch {:03d}/{:03d} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}".format(
                        epoch, number_epochs, cheese_np.sum(), 
                        cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt, 
                        win_cnt-last_W, draw_cnt-last_D, lose_cnt-last_L, steps/100)
            print(string)
            winrate_pereps.append((win_cnt-last_W)/100)
            drawrate_pereps.append((draw_cnt-last_D)/100)
            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt                

# This is the main function called in order to train the agent
if __name__ == '__main__':
    saver = tf.train.Saver()
    with tf.Session() as sess:
        exp_replay = rl.ExperienceReplay(sess, model, max_memory=max_memory)
        if not load:
            init = tf.initialize_all_variables()
            sess.run(init)
        else :
            saver.restore(sess, "save_rl/model.ckpt")
        print("Training")
        play(sess, model, number_epochs, True)
        
        # Plotting the statistics for the winrate
        plt.figure()
        plt.plot(range(100, number_epochs + 100, 100), winrate_pereps)
        plt.savefig('winrate.png')
        plt.xlabel('epochs number')
        plt.ylabel('winrate')
        plt.savefig('drawrate1.png')
        
        if save :
           saver.save(sess, "save_rl/model.ckpt")
           print("done")
        print("Training done")
        print("Testing")
        play(sess, model, 100, False)
        print("Testing done")