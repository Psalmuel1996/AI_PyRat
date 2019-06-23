# In this example, we can obtain scores in the order of: "win_python": 0.07 "win_rat": 0.93
import random
from time import time

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

# Useful utility functions to obtain new location after a move
def move(location, move):
    if move == MOVE_UP:
        return (location[0], location[1] + 1)
    if move == MOVE_DOWN:
        return (location[0], location[1] - 1)
    if move == MOVE_LEFT:
        return (location[0] - 1, location[1])
    if move == MOVE_RIGHT:
        return (location[0] + 1, location[1])
    

# The first things we do is we program the AI of the opponent, so that we know exactly what will be its decision in a given situation
def distance(la, lb):
    ax,ay = la
    bx,by = lb
    return abs(bx - ax) + abs(by - ay)

# With this template, we are building an AI that will apply combinatorial game theory tools against a greedy opponent
TEAM_NAME = "TWITCH"


def updatePlayerLocation(target,playerLocation):

    if playerLocation[0] < target[0] :
        playerLocation = move(playerLocation, MOVE_RIGHT)
    elif playerLocation[0] > target[0] :
        playerLocation = move(playerLocation, MOVE_LEFT)
    elif playerLocation[1] < target[1] :
        playerLocation = move(playerLocation, MOVE_UP)
    else :
        playerLocation = move(playerLocation, MOVE_DOWN)
    return playerLocation

def checkEatCheese():
    global player_score
    global opponent_score
    
    if player_loc in remain_pieces and player_loc == opponent_loc:
        player_score = player_score + 0.5
        opponent_score = opponent_score + 0.5
        remain_pieces.remove(player_loc)
        scenario_path.append(player_loc)
        
    else:
        if player_loc in remain_pieces:
            player_score +=  1
            remain_pieces.remove(player_loc)
            scenario_path.append(player_loc)
            
        if opponent_loc in remain_pieces:
            opponent_score += 1
            remain_pieces.remove(opponent_loc)
    

def trier_cheese(pieces, reference) :
    pieces_in_order = pieces[:]
    distances = [distance(cheese, reference) for cheese in pieces_in_order]
    pieces_to_return = []
    distances_to_return = []
    
    i = 0
    while i < len(distances)  :
        j = 0
        while distances[j] < distances[i] and j < i:
            j += 1
        distances_to_return.insert(j, distances[i])
        pieces_to_return.insert(j, pieces_in_order[i])
        i += 1
    return pieces_to_return
    
def gen_maze(piecesOfCheese, playerLocation) :
#    print('cheese :', piecesOfCheese)
#    print('playerLoc : ', playerLocation)
    maze = {}    
    spots = [playerLocation] + piecesOfCheese[:]
    for spot in spots :
        maze[spot] = {}
    
    while len(spots) > 1 :
        reference = spots.pop(0)
        pieces_of_cheese = trier_cheese(spots,reference)
        
        # creating bounds
        for piece in pieces_of_cheese :
#            print(reference, ' | ', piece)
            maze[reference][piece] = 3 / distance(reference, piece)
            if reference != playerLocation :
                maze[piece][reference] = 3 / distance(reference, piece)
    return maze

def best_target_play(remain_pieces, player_loc, metagraph, opponent_loc):
    global player_targ
    
    if distance(player_targ, player_loc) != 0 : 
        pass
    
    else :
        maze = metagraph[player_loc].copy()
        val = 0
        pieces_to_consider = []
        for item in maze.items() :
            if item[0] in remain_pieces :
                pieces_to_consider.append(item)
                val += item[1]
        u = random.random() * val
        j = 0
        v = 0
        while v < u :
            item = pieces_to_consider[j]
            v += item[1]
            j += 1
        player_targ = pieces_to_consider[j-1][0]
    
def best_target_adv(piecesOfCheese, opponentLocation, mazeWidth, mazeHeight) :

    closest_poc = (-1,-1)
    best_distance = mazeWidth + mazeHeight
    for poc in remain_pieces:
        if distance(poc, opponentLocation) < best_distance:
            best_distance = distance(poc, opponentLocation)
            closest_poc = poc
    return closest_poc

def gen_turn(metagraph, mazeWidth, mazeHeight):
    global opponent_loc
    global player_loc 
    opponent_target = best_target_adv(remain_pieces, opponent_loc, mazeWidth, mazeHeight)
    opponent_loc = updatePlayerLocation(opponent_target, opponent_loc)
    best_target_play(remain_pieces, player_loc, metagraph, opponent_loc)
    player_loc = updatePlayerLocation(player_targ, player_loc)
    
    checkEatCheese()
    
    
def gen_scenario(metagraph, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese):
    global player_loc
    global opponent_loc
    global player_score
    global opponent_score
    global remain_pieces 
    global player_targ
    global scenario_path
    
    scenario_path = [playerLocation]
    player_targ = playerLocation
    player_loc = playerLocation            
    opponent_loc = opponentLocation
    player_score = opponent_score = 0
    remain_pieces = piecesOfCheese[:]
    scenario_path = [playerLocation]
    while player_score < 20 and opponent_score < 20 and opponent_score - player_score < 5:
        gen_turn(metagraph, mazeWidth, mazeHeight)
    scores = [player_score, opponent_score]    
    return scores
    
def monte_carlo(mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese):
    global path
    global win_path
    global opponent_final_score
    global win_scenario
    global s
    path = [playerLocation]
    metagraph = gen_maze(piecesOfCheese, playerLocation)
    win_scenario = False
    t = 0
    s = 0
    opponent_final_score = 21
    win_path = []
    while t != 1000 :
        scores = gen_scenario(metagraph, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese)
        if scores[0] >= 20:
            if not win_scenario : 
                win_scenario = True
                win_path = scenario_path
                opponent_final_score = scores[1]

            else : 
                
                if scores[1] < opponent_final_score :
                    opponent_final_score = scores[1]
                    win_path = scenario_path
#                    print("win")
                if opponent_final_score < 18 :
                    return win_path
                    break

                        
        t += 1
        if t==1000 and s < 2 :
            t = 0
            s+=1
    return win_path

    
    


   
