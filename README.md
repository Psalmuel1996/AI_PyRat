## PyRat
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyRat is a maze game with two opponents (a rat and a python), some pieces of cheese, and potentially obstacles and mud. The  goal is to collect as much pieces as possible.

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Here is a picture below. For more information about the game and for the game’s source code visit this link : https://github.com/vgripon/PyRat
 <p align="center"><img src="https://github.com/aminedassouli/AI_PyRat/blob/master/PyRat_game.png"></p>

## The AI
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The AI was conceived using deep Q-learning. The tensorflow model for neural network designed to predict the Q function is written in the rl.py file.

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Here is the curve of the evolution of the winrate through training : 
 <p align="center"><img src="https://github.com/aminedassouli/AI_PyRat/blob/master/winrate.png"></p>

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The winrate becomes about 60 % in the end of the training. However, since I was more ambitious, I tried to supervise the Q-learning by choosing the actions to learn from with Monte Carlo instead of learning from completely random actions. 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Therefore, the winrate reaches 80 % !
