""" Team name to be displayed in the game """
TEAM_NAME = "Twitch_AI"


""" When the player is performing a move, it actually sends a character to the main program
The four possibilities are defined below """
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'


""" Imports """
import tensorflow as tf
import rl
import numpy as np
from math import inf

""" Global variables """
global model, exp_replay, input_tm1, action, score, last_positions


""" Functions """
# This function creates a numpy array representation of the maze
def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 2)
    canvas = np.zeros(im_size)
    (x,y) = player
    (xx,yy) = opponent
    center_x, center_y = mazeWidth - 1, mazeHeight - 1
    for (x_cheese, y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1
    canvas[yy + center_y - y, xx + center_x - x, 1] = 1   
    canvas = np.expand_dims(canvas, axis=0)
    return canvas

# This function defines the best target for the player 
def best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    
    # First we should check how many pieces of cheese each player has to see if the match is over
    totalPieces = len(piecesOfCheese) + playerScore + opponentScore
    if playerScore > totalPieces / 2 or opponentScore > totalPieces / 2 or len(piecesOfCheese) == 0:
        return (-1,-1), playerScore

    # If the match is not over, then we try to find the best piece of cheese in term of score
    best_target_so_far, best_score_so_far = (-1, -1), -1
    for target in piecesOfCheese:
        end_state = simulate_game_until_target(
            target,playerLocation,opponentLocation,
            playerScore,opponentScore,piecesOfCheese.copy())
        _, score = best_target(*end_state)
        if score > best_score_so_far:
            best_score_so_far = score
            best_target_so_far = target

    return best_target_so_far, best_score_so_far

# This function is used to obtain the new location after a move
def move(location, move):
    if move == MOVE_UP :
        return (location[0], location[1] + 1)
    if move == MOVE_DOWN :
        return (location[0], location[1] - 1)
    if move == MOVE_LEFT :
        return (location[0] - 1, location[1])
    if move == MOVE_RIGHT :
        return (location[0] + 1, location[1])

# This function calculates the distance between two locations
def distance(la, lb):
    ax,ay = la
    bx,by = lb
    return abs(bx - ax) + abs(by - ay)

# This function returns the movement corresponding to the action
def action_to_movement(action) :
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

# This function is used to update the location after a turn depending on the chosen target
def updatePlayerLocation(target, playerLocation) :
    if playerLocation[1] != target[1]:
        if target[1] < playerLocation[1]:
            playerLocation = move(playerLocation, MOVE_DOWN)
        else:
            playerLocation = move(playerLocation, MOVE_UP)
    elif target[0] < playerLocation[0]:
        playerLocation = move(playerLocation, MOVE_LEFT)
    else:
        playerLocation = move(playerLocation, MOVE_RIGHT)
    return playerLocation

# This function simulates the next movement of the opponent considering that it is a greedy
def turn_of_opponent(opponentLocation, piecesOfCheese):    
    closest_poc, best_distance = (-1,-1), inf
    for poc in piecesOfCheese :
        if distance(poc, opponentLocation) < best_distance :
            best_distance = distance(poc, opponentLocation)
            closest_poc = poc
    ax, ay = opponentLocation
    bx, by = closest_poc
    if bx > ax:
        return MOVE_RIGHT
    elif bx < ax:
        return MOVE_LEFT
    elif by > ay:
        return MOVE_UP
    elif by < ay:
        return MOVE_DOWN
    pass

# This function checks if a player reached a piece of cheese and updates the scores
def checkEatCheese(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):
    if playerLocation in piecesOfCheese and playerLocation == opponentLocation :
        playerScore += 0.5
        opponentScore += 0.5
        piecesOfCheese.remove(playerLocation)
    else :
        if playerLocation in piecesOfCheese :
            playerScore += 1
            piecesOfCheese.remove(playerLocation)
        if opponentLocation in piecesOfCheese:
            opponentScore += 1
            piecesOfCheese.remove(opponentLocation)
    return playerScore, opponentScore

# This function simulates the game until the target cheese has been eaten by either player 
def simulate_game_until_target(target,playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese):    
    while target in piecesOfCheese:
        playerLocation = updatePlayerLocation(target, playerLocation)
        opponentLocation = move(opponentLocation, turn_of_opponent(opponentLocation, piecesOfCheese))
        playerScore, opponentScore = checkEatCheese(
            playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
    return playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese
    


# This function is called once at the start of a game
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    """ Arguments are:
     - mazeMap : dict(pair(int, int), dict(pair(int, int), int))
     - mazeWidth : int
     - mazeHeight : int
     - playerLocation : pair(int, int)
     - opponentLocation : pair(int,int)
     - piecesOfCheese : list(pair(int, int))
     - timeAllowed : float
    """
    global model, exp_replay, input_tm1, action, score, last_positions
    
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    model = rl.NLinearModels(2*1189,4,32)
    last_positions = [(0,0)] 
    

# This function is called at each turn and it is expected to return a move
current_target =(-1,-1)
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):  
    """ Arguments are:
     - mazeMap : dict(pair(int, int), dict(pair(int, int), int))
     - mazeWidth : int
     - mazeHeight : int
     - playerLocation : pair(int, int)
     - opponentLocation : pair(int,int)
     - playerScore : float
     - opponentScore : float
     - piecesOfCheese : list(pair(int, int))
     - timeAllowed : float
    """
    global model,input_tm1, action, score, current_target, last_positions

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "save_rl/model.ckpt")
        input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)
        q = model.predict_one(sess, input_tm1)[0]
        action = np.argmax(q)
        q[action] = -1000
        new_spot = move(playerLocation, action_to_movement(action))

        # We check if the new state is illegal (a wall) and if it is the case we choose another action
        while new_spot[0] in [-1,mazeWidth] or new_spot[1] in [-1,mazeHeight] :
            action = np.argmax(q)
            q[action] = -1000
            new_spot = move(playerLocation, action_to_movement(action))
            
        # We check if the new state is the same as the previous one which leaded to the current state, so that we avoid endless loops
        if len(last_positions) == 1 :
            last_positions.append((0,0))    
        elif new_spot == last_positions[0] :
            action = np.argmax(q)
            new_spot = move(playerLocation, action_to_movement(action))
        last_positions.pop(0)
        last_positions.append(new_spot)
        
        score = playerScore
        return action_to_movement(action)
