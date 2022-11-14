import numpy as np
import random as rm
 # Define the 6 existng states 
states = ["I","II","III","IV","V","VI"]

#Since we have probablities associated with each transitions define the transitions 
transitionName = [["I-I","I-II","I-III","I-IV","I-V","I-VI"],["III-III","III-I","III-II","III-IV","III-V","III-VI"],
                  ["V-V","V-I","V-II","V-III","V-IV","V-VI"]]

# Probabilities matrix (transition matrix)
#for a start assign random values of probablity adding to a value of 1
transitionMatrix = [[0.4,0.1,0.25,0.05,0.15,0.05],[0.4,0.1,0.25,0.05,0.15,0.05],[0.4,0.1,0.25,0.05,0.15,0.05]]
#Define R for each states /use this after an attempt of just markov chain 
#reward=[0,-1,1,-1,1,-1]

#checking if the sum of probablities is equal to 3, since we have 3 states where transistions can start. note no transitions
#happens when fail state is reached 
if sum(transitionMatrix[0])+sum(transitionMatrix[1])+sum(transitionMatrix[2]) != 3:
    print("Somewhere, something went wrong. Transition matrix, perhaps?")
else: print("All is gonna be okay, you should move on!! ;)")

# A function that implements the Markov model to forecast the state
def state_transitions(iterations):
    # Choose the starting state
    initial_state = "I"
    print("Start state: " +  initial_state)
    # Shall store the sequence of states taken. So, this only has the starting state for now.
    state_List = [initial_state]
    i = 0
   
    prob = 1
    """use while loop to got through sequence of transitions based on the number of iteration number given
    The code is not efficient way to go about it for high number of transitions since a separate conditions
    are given for each states"""
    while i != iterations:
        if initial_state == "I": # when next state is state I
            TT=transitionName[0]
            TP=transitionMatrix[0]
            change = np.random.choice(TT,replace=True,p=TP)
            if change == TT[0]:
                prob = prob * TP[0]
                state_List.append("I")
                pass
            elif change == TT[1]:
                prob = prob * TP[1]
                initial_state = "II"
                state_List.append("II")
            elif change == TT[2]:
                prob = prob * TP[2]
                initial_state = "III"
                state_List.append("III")
            elif change == TT[3]:
                prob = prob * TP[3]
                initial_state = "IV"
                state_List.append("IV")
            elif change == TT[4]:
                prob = prob * TP[4]
                initial_state = "V"
                state_List.append("V")
            else:
                prob = prob * TP[5]
                initial_state = "VI"
                state_List.append("VI")
        elif initial_state == "III": # when next state is state III
            TT=transitionName[1]
            TP=transitionMatrix[1]
            change = np.random.choice(TT,replace=True,p=TP)
            if change == TT[0]:
                prob = prob * TP[0]
                state_List.append("III")
                pass
            elif change == TT[1]:
                prob = prob * TP[1]
                initial_state = "I"
                state_List.append("I")
            elif change == TT[2]:
                prob = prob * TP[2]
                initial_state = "II"
                state_List.append("II")
            elif change == TT[3]:
                prob = prob * TP[3]
                initial_state = "IV"
                state_List.append("IV")
            elif change == TT[4]:
                prob = prob * TP[4]
                initial_state = "V"
                state_List.append("V")
            else:
                prob = prob * TP[5]
                initial_state = "VI"
                state_List.append("VI")
        elif initial_state == "V": # when next state is state v
            TT=transitionName[2]
            TP=transitionMatrix[2]
            change = np.random.choice(TT,replace=True,p=TP)
            if change == TT[0]:
                prob = prob * TP[0]
                state_List.append("V")
                pass
            elif change == TT[1]:
                prob = prob * TP[1]
                initial_state = "I"
                state_List.append("I")
            elif change == TT[2]:
                prob = prob * TP[2]
                initial_state = "II"
                state_List.append("II")
            elif change == TT[3]:
                prob = prob * TP[3]
                initial_state = "III"
                state_List.append("III")
            elif change == TT[4]:
                prob = prob * TP[4]
                initial_state = "IV"
                state_List.append("IV")
            else:
                prob = prob * TP[5]
                initial_state = "VI"
                state_List.append("VI")
        else:
            prob = prob
            break
            
        i += 1  
    print("State transitions in sequence: " + str(state_List))
    print("End state after "+ str(iterations) + " iterations: " + initial_state)
    print("Probability of the possible sequence of states: " + str(prob))

# Function that forecasts the possible state for the next num days
num=10
state_transitions(num)
