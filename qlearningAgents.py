# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This assignment is completed by Kevin Cheng for the course CMPT 310 (Intro Artificial Intelligence) at Simon Fraser University
# The repo for this assignment can be found at https://github.com/kzcheng/CMPT310-Asn3


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math
import logging

# Debug tags
VERBOSE = False
# VERBOSE = True

logging.basicConfig(level=logging.INFO, format='%(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# Question 6: Q-Learning

# Note that your value iteration agent does not actually learn from experience. Rather, it ponders its MDP model to arrive at a complete policy before ever interacting with a real environment. When it does interact with the environment, it simply follows the precomputed policy (e.g. it becomes a reflex agent). This distinction may be subtle in a simulated environment like a Gridword, but it’s very important in the real world, where the real MDP is not available.

# You will now write a Q-learning agent, which does very little on construction, but instead learns by trial and error from interactions with the environment through its update(state, action, nextState, reward) method. A stub of a Q-learner is specified in QLearningAgent in qlearningAgents.py, and you can select it with the option -a q. For this question, you must implement the update, computeValueFromQValues, getQValue, and computeActionFromQValues methods.

# Note: For computeActionFromQValues, you should break ties randomly for better behavior. The random.choice() function will help. In a particular state, actions that your agent hasn’t seen before still have a Q-value, specifically a Q-value of zero, and if all of the actions that your agent has seen before have a negative Q-value, an unseen action may be optimal.

# Important: Make sure that in your computeValueFromQValues and computeActionFromQValues functions, you only access Q-values by calling getQValue. This abstraction will be useful for question 10 when you override getQValue to use features of state-action pairs rather than state-action pairs directly.

# With the Q-learning update in place, you can watch your Q-learner learn under manual control, using the keyboard:
#   python gridworld.py -a q -k 5 -m

# Recall that −k will control the number of episodes your agent gets to learn. Watch how the agent learns about the state it was just in, not the one it moves to, and ”leaves learning in its wake.” Hint: to help with debugging, you can turn off noise by using the `−−noise 0.0` parameter (though this obviously makes Q-learning less interesting). If you manually steer Pacman north and then east along the optimal path for four episodes, you should see the following Q-values.

# Grading: We will run your Q-learning agent and check that it learns the same Q-values and policy as our reference implementation when each is presented with the same set of examples. To grade your implementation, run the autograder:
#   python autograder.py -q q6


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # This is the qValues that we learned, which gets calculated on the fly
        # The key is a (state, action) tuple
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # max_action Q(state,action)
        # This means max over action ( Q(state,action) )
        # As in, the max value of Q(state,action)
        # So we should only return a single value here

        "*** YOUR CODE HERE ***"
        # Figured out the issue. The problem is, getQValue is called at an unfortunate time. In ApproximateQAgent, getQValue is upgraded into something else, which means we are no longer allowed to call it when the state is terminal.
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        return self.getQValue(state, self.getPolicy(state))

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # The best action is the one with the highest Q-value

        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        maxQValue = float('-inf')
        maxActions = []
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                maxActions = [action]
            elif qValue == maxQValue:
                maxActions.append(action)
        maxAction = random.choice(maxActions)
        return maxAction

    # Question 7: Epsilon Greedy

    # Complete your Q-learning agent by implementing epsilon-greedy action selection in getAction, meaning it chooses random actions an epsilon fraction of the time, and follows its current best Q-values otherwise. Note that choosing a random action may result in choosing the best action - that is, you should not choose a random sub-optimal action, but rather any random legal action.

    # You can choose an element from a list uniformly at random by calling the random.choice function. You can simulate a binary variable with probability p of success by using util.flipCoin(p), which returns True with probability p and False with probability 1 − p.

    # After implementing the getAction method, observe the following behavior of the agent in gridworld (with epsilon = 0.3).
    #   python gridworld.py -a q -k 100

    # Your final Q-values should resemble those of your value iteration agent, especially along well-traveled paths. However, your average returns will be lower than the Q-values predict because of the random actions and the initial learning phase.

    # You can also observe the following simulations for different epsilon values. Does that behavior of the agent match what you expect?
    #   python gridworld.py -a q -k 100 --noise 0.0 -e 0.1
    #   python gridworld.py -a q -k 100 --noise 0.0 -e 0.9

    # To test your implementation, run the autograder:
    #   python autograder.py -q q7

    # With no additional code, you should now be able to run a Q-learning crawler robot:
    #   python crawler.py

    # If this doesn’t work, you’ve probably written some code too specific to the GridWorld problem and you should make it more general to all MDPs.

    # This will invoke the crawling robot from class using your Q-learner. Play around with the various learning parameters to see how they affect the agent’s policies and actions. Note that the step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of your learning algorithm, and the discount factor is a property of the environment.

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

          This method modifies self.qValues
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (R(s, a, s') + gamma * value(s'))
        self.qValues[(state, action)] = \
            (1 - self.alpha) * self.getQValue(state, action) + \
            self.alpha * (reward + self.discount * self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


# Question 9: Q-Learning and Pacman

# Time to play some Pacman! Pacman will play games in two phases. In the first phase, training, Pacman will begin to learn about the values of positions and actions. Because it takes a very long time to learn accurate Q-values even for tiny grids, Pacman’s training games run in quiet mode by default, with no GUI (or console) display. Once Pacman’s training is complete, he will enter testing mode. When testing, Pacman’s self.epsilon and self.alpha will be set to 0.0, effectively stopping Q-learning and disabling exploration, in order to allow Pacman to exploit his learned policy. Test games are shown in the GUI by default. Without any code changes you should be able to run Q-learning Pacman for very tiny grids as follows:
#   python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Note that PacmanQAgent is already defined for you in terms of the QLearningAgent you’ve already written. PacmanQAgent is only different in that it has default learning parameters that are more effective for the Pacman problem (epsilon=0.05, alpha=0.2, gamma=0.8).

# You will receive full credit for this question if the command above works without exceptions and your agent wins at least 80% of the time. The autograder will run 100 test games after the 2000 training games.

# Hint: If your QLearningAgent works for gridworld.py and crawler.py but does not seem to be learning a good policy for Pacman on smallGrid, it may be because your getAction and/or computeActionFromQValues methods do not in some cases properly consider unseen actions. In particular, because unseen actions have by definition a Q-value of zero, if all of the actions that have been seen have negative Q-values, an unseen action may be optimal. Beware of the argmax function from util.Counter!

# Note: To grade your answer, run:
#   python autograder.py -q q9

# Note: If you want to experiment with learning parameters, you can use the option -a, for example -a epsilon=0.1,alpha=0.3,gamma=0.7. These values will then be accessible as self.epsilon, self.gamma and self.alpha inside the agent.

# Note: While a total of 2010 games will be played, the first 2000 games will not be displayed because of the option -x 2000, which designates the first 2000 games for training (no output). Thus, you will only see Pacman play the last 10 of these games.

# The number of training games is also passed to your agent as the option numTraining.

# Note: If you want to watch 10 training games to see what’s going on, use the command: python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

# During training, you will see output every 100 games with statistics about how Pacman is faring. Epsilon is positive during training, so Pacman will play poorly even after having learned a good policy: this is because he occasionally makes a random exploratory move into a ghost. As a benchmark, it should take between 1,000 and 1400 games before Pacman’s rewards for a 100 episode segment becomes positive, reflecting that he’s started winning more than losing. By the end of training, it should remain positive and be fairly high (between 100 and 350).

# Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.

# However, you will find that training the same agent on the seemingly simple mediumGrid does not work well. In our implementation, Pacman’s average training rewards remain negative throughout training. At test time, he plays badly, probably losing all of his test games. Training will also take a long time, despite its ineffectiveness.

# Pacman fails to win on larger layouts because each board configuration is a separate state with separate Q-values. He has no way to generalize that running into a ghost is bad for all positions. Obviously, this approach will not scale.

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

# Question 10: Approximate Q-Learning

# Implement an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. Write your implementation in ApproximateQAgent class in qlearningAgents.py, which is a subclass of PacmanQAgent.

# Note: Approximate Q-learning assumes the existence of a feature function f(s, a) over state and action pairs, which yields a vector [f1(s, a), ..., fi(s, a), ..., fn(s, a) of feature values.
# We provide feature functions for you in featureExtractors.py. Feature vectors are util.Counter (like a dictionary) objects containing the non-zero pairs of features and values; all omitted features have value zero.
# So, instead of an vector where the index in the vector defines which feature is which, we have the keys in the dictionary define the identity of the feature.

# The approximate Q-function takes the following form:
#   Q(s, a) = Sum from i=1 to n ( fi(s, a) * wi )
# where each weight wi is associated with a particular feature fi(s, a).
# In your code, you should implement the weight vector as a dictionary mapping features (which the feature extractors will return) to weight values.

# You will update your weight vectors similarly to how you updated Q-values:
#   wi <- wi + alpha * difference * fi(s, a)
#   difference = (reward + gamma * max_a'(Q(s', a'))) - Q(s, a)

# Note that the difference term is the same as in normal Q-learning, and reward is the experienced reward.

# By default, ApproximateQAgent uses the IdentityExtractor, which assigns a single feature to every (state,action) pair.
# With this feature extractor, your approximate Q-learning agent should work identically to PacmanQAgent. You can test this with the following command:
#   python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

# Important: ApproximateQAgent is a subclass of QLearningAgent, and it therefore shares several methods like getAction. Make sure that your methods in QLearningAgent call getQValue instead of accessing Q-values directly, so that when you override getQValue in your approximate agent, the new approximate q-values are used to compute actions.

# Once you’re confident that your approximate learner works correctly with the identity features, run your approximate Q-learning agent with our custom feature extractor, which can learn to win with ease:
#   python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# Even much larger layouts should be no problem for your ApproximateQAgent. (warning: this may take a few minutes to train)
#   python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

# If you have no errors, your approximate Q-learning agent should win almost every time with these simple features, even with only 50 training games.

# Grading: We will run your approximate Q-learning agent and check that it learns the same Q-values and feature weights as our reference implementation when each is presented with the same set of examples. To grade your implementation, run the autograder:
#   python autograder.py -q q10


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        logging.debug(f"state = {state}")
        featureVector = self.featExtractor.getFeatures(state, action)

        # Q(s, a) = Sum from i=1 to n ( fi(s, a) * wi )
        logging.debug(f"self.weights * featureVector = {self.weights * featureVector}")
        return self.weights * featureVector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        featureVector = self.featExtractor.getFeatures(state, action)
        logging.debug(f"featureVector = {featureVector}")

        # difference = (reward + gamma * max_a'(Q(s', a'))) - Q(s, a)
        difference = (reward + self.discount * self.getValue(nextState)) \
            - self.getQValue(state, action)

        for feature in featureVector:
            # wi <- wi + alpha * difference * fi(s, a)
            self.weights[feature] += self.alpha * difference * featureVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
