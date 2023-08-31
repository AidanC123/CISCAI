import math
import random
import copy
import numpy as np
#all fairly conventional python libraries

class State:
    def __init__(self, board, player):
        self.board = board
        self.player = player

#whos moving
def toMove(state):
    return state.player

#what the person whos moving can do
def actions(state):
    action_list = []
    player = toMove(state)
    for i in range(3):
        for j in range(3):
            if(state.board[i][j] == -1 and player == -1):
                if((state.board[i+1][j] == 0) and (i < 2)):
                    action_list.append(("advance", i+1, j))
                if((j > 0) and (i < 2)):
                    if(state.board[i+1][j-1] == 1):
                        action_list.append(("capture left", i+1, j-1))
                if((j < 2) and (i < 2)):
                    if(state.board[i+1][j+1] == 1):
                        action_list.append(("capture right",i+1,j+1))
            if(state.board[i][j] == 1 and player == 1):
                if((state.board[i-1][j] == 0) and (i > 0)):
                    action_list.append(("advance",i-1,j))
                if((j > 0) and (i > 0)):
                    if(state.board[i-1][j-1] == -1):
                        action_list.append(("capture left", i-1, j-1))
                if((j < 2) and (i > 0)):
                    if(state.board[i-1][j+1] == -1):
                        action_list.append(("capture right",i-1,j+1))
    return action_list

#have the person whos moving actually move
def result(state, action):
    player = toMove(state)
    new_board = copy.deepcopy(state.board)
    new_state = None
    if player == -1:
        new_state = State(new_board,1)
    if player == 1:
        new_state = State(new_board,-1)
    
    new_state.board[action[1]][action[2]] = player
    if(action[0] == "advance"):
        if(player == 1):
            new_state.board[action[1]+1][action[2]] = 0
        else:
            new_state.board[action[1]-1][action[2]] = 0

    if(action[0] == "capture left"):
        if(player == 1):
            new_state.board[action[1]+1][action[2]+1] = 0
        else:
            new_state.board[action[1]-1][action[2]+1] = 0

    if(action[0] == "capture right"):
        if(player == 1):
            new_state.board[action[1]+1][action[2]-1] = 0
        else:
            new_state.board[action[1]-1][action[2]-1] = 0

    return new_state
    
#is the game over?
def isTerminal(state):
    if bool(utility(state)) == True:
        return True
    else:
        return False
        

#helper with isTerminal, basically checks if game is over
def utility(state):
    if(actions(state)):
        if -1 in state.board[2][:]:
            return -1
        if 1 in state.board[0][:]:
            return 1
        else:
            return 0
    else:
        return -1 * state.player

#All possible non-term states 
def maker(state):
    space = []
    queue = [state]
    while queue:
        current = queue.pop()
        if isTerminal(current):
            current = None
            exit
        else:
            for a in actions(current):
                appendme = result(current, a)
                queue.append(appendme)
                space.append(appendme)
            
    return space

#aima adapted, normal minmax
#https://github.com/aimacode/aima-python/blob/master/games4e.ipynb
def buildMinimax(state):
    player = toMove(state)  
    def max_value(state):
        move = None
        if isTerminal(state):
            return utility(state), move
        v = -np.inf
        for a in actions(state):
            v2, etc = min_value(result(state, a))
            if v2 > v:
                v = v2
                move = a
        return v, move

    def min_value(state):
        move = None
        if isTerminal(state):
            return utility(state), move
        v = np.inf
        for a in actions(state):
            v2, etc = max_value(result(state, a))
            if v2 < v:
                v = v2
                move = a
        return v, move
    value, move = max_value(state)
    return move

#build policy table from minmax
def buildPolicyTable(root):
    policytable = []
    for state in maker(root):
        movelist = []
        nextmove = buildMinimax(state)
        s = state
        while isTerminal(s) == False:
            movelist.append(nextmove)
            nextmove = buildMinimax(s)
            print(movelist)
            s = result(s, nextmove)
        u = utility(s)
        policytable.append((state, state.player, (u,movelist)))
    return policytable

#parts adapted from https://github.com/scottfones/HexapaFFNN/blob/main/graph.py
#I believe I understand this well enough but he created it much better than I did
#Obviously its up to the point where I think I understand, Im trying to do classify
class neuron_structure:
    def __init__(self, hidden_layers, in_units, out_units, hidden_units, network_layers):
        self.hidden_layers = hidden_layers
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self.network_layers = network_layers

def build_network(structure):
    structure.network_layers = []
    for i in range(structure.hidden_layers):
        if i > 0:
            if type(structure.network_layers[i - 1]) is list:
                hiddena = structure.network_layers[i - 1][0].shape[1]
            else:
                hiddena = structure.network_layers[i - 1].shape[1]
        else:
            hiddena = structure.in_units
        hiddenb = structure.hidden_units
        tempw = (np.random.randn(hiddena, hiddenb))
        tempn = (np.random.randn(hiddenb, 1))
        structure.network_layers.append([tempw, tempn])

    tempw = np.random.randn(structure.hidden_units, structure.out_units)
    tempn = np.zeros((structure.out_units, 1))
    structure.network_layers.append([tempw, tempn])
    return structure.network_layers

def classify(structure, data = [1,2,3,4]):
    data = np.array(data)
    x = structure.network_layers[0][1]
    y = structure.network_layers[0][0]
    print(x.shape)
    print(data.shape)
    dotproduct = y.T.dot(data) + x.T
    return dotproduct
    
#adapted from https://github.com/aimacode/aima-python/blob/master/deep_learning4e.py
#AIMA
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_derv(x):
    return value * (1 - value)

def ReLU(x):
    return max(0, x)

def ReLU_derv(value):
    return 1 if value > 0 else 0

    
  
#Initial Board
test_board = [[-1,-1,-1],[0,0,0],[1,1,1]]

test_state = State(test_board, 1)



#buildPolicyTable(test_table)
radarada = buildPolicyTable(test_state)
inputs = radarada
#print(inputs[0])
neuron = neuron_structure(1, 2, 2, 2, 0)
chunk = (neuron_structure(1, 2, 2, 2, build_network(neuron)))
print(chunk.network_layers)
print("__________________")
#print(classify(neuron, radarada))
#Not my finest. Im sorry. Have a nice rest of your semester!
