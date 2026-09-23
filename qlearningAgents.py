# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.Q = {}

    "*** YOUR CODE HERE ***"


  def setQValue(self,state,action,value):
    """
      Set Q(state,action) to the value
    """
    "*** YOUR CODE HERE ***"
    self.Q[(state, action)] = value
    

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    val = self.Q.get((state, action))
    if(val is None) :
      return 0.0
    return val
    "*** YOUR CODE HERE ***"

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    legalActions = self.getLegalActions(state)
    qNext = [] 
    for action in legalActions :
      val = self.getQValue(state, action)
      if val is None :
        qNext.append(0.0)
      qNext.append(val)
      
    return max(qNext, default = 0.0)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.
      Note that the policy does not return here all the best actions, but break ties randomly
      to return one of the best actions.
      Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    
    legalActions = self.getLegalActions(state)
    print(legalActions)
    print(len(legalActions))
    if( len(legalActions) == 0) :
      return None
    else : 
      best_action = []
      max_value = -99999999
      for action in legalActions :
        value = self.getQValue(state, action)
        if(value >= max_value) :
          best_action.append(action)
          max_value = value
      return random.choice(best_action)
        


  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)

    """
    "*** YOUR CODE HERE ***"
    legalActions = self.getLegalActions(state)
    if(util.flipCoin(self.epsilon)):
      return random.choice(legalActions)
    else :
      return self.getPolicy(state)
    



  def update(self, state, action, nextState, reward):
    """
      The parent class calls this whan a transition has been observed
      state , action => nextState , reward .
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    
    qValue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.epsilon * self.getValue(nextState))
    self.setQValue(state, action, qValue)
    "*** YOUR CODE HERE ***"
    print(f"state : {nextState} action : {action} -> {self.getQValue(state, action)}")

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action





class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is (assuming that your methods
     in QLearningAgent call getQValue instead of accessing Q-values directly)
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
