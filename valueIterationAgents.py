# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import collections
from learningAgents import ValueEstimationAgent
import util
import mdp
import logging

# Debug tags
VERBOSE = False
# VERBOSE = True

logging.basicConfig(level=logging.INFO, format='%(message)s')
calledCounters = collections.Counter()


def called(functionName):
    calledCounters[functionName] += 1
    logging.debug(f"\n\n[ Called {functionName} {calledCounters[functionName]} times ]")


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              self.mdp.getStates()
              self.mdp.getPossibleActions(state)
              self.mdp.getTransitionStatesAndProbs(state, action)
              self.mdp.getReward(state, action, nextState)
              self.mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        stateList = self.mdp.getStates()
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in stateList:
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    continue
                bestAction = None
                bestValue = float('-inf')
                for action in actions:
                    qValue = self.getQValue(state, action)
                    if qValue > bestValue:
                        bestAction = action
                        bestValue = qValue
                newValues[state] = bestValue
            self.values = newValues.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        if VERBOSE:
            logging.getLogger().setLevel(logging.DEBUG)

        # Main Logic of Method
        # logging.debug("\n\n")
        # called("computeQValueFromValues")
        # logging.debug(f"state = {state}")
        # logging.debug(f"action = {action}")

        # Given the values of the states, compute the Q-value of the state-action pair
        # Formula:
        # Q(s,a) = Sum of all s' (  T(s,a,s') * (R(s,a,s') + discount * V(s'))  )

        potentialNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        # logging.debug(f"potentialNextStates = self.mdp.getTransitionStatesAndProbs(state, action) = {self.mdp.getTransitionStatesAndProbs(state, action)}")

        # self.mdp.getTransitionStatesAndProbs(state, action) = [((0, 2), 1.0), ((0, 1), 0.0)]
        # ((0, 2), 1.0) is a returned tuple
        # (0, 2) is the position (which is also the "state")
        # 1.0 is the probability

        totalQValue = 0

        for potentialNextState in potentialNextStates:
            nextState = potentialNextState[0]
            # logging.debug(f"nextState = {nextState}")
            probability = potentialNextState[1]
            reward = self.mdp.getReward(state, action, nextState)

            # if probability == 0:    # meaningless optimization
            #     continue
            totalQValue += probability * (reward + self.discount * self.getValue(nextState))
            # logging.debug(f"self.getValue(nextState) = {self.getValue(nextState)}")

        # logging.debug(f"totalQValue = {totalQValue}")

        logging.getLogger().setLevel(logging.INFO)
        return totalQValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if VERBOSE:
            logging.getLogger().setLevel(logging.DEBUG)

        # state is just a string lmao
        # logging.debug(f"state = {state}")
        # state = TERMINAL_STATE
        # type(state) = <class 'str'>

        # This is all the states in the given MDP
        # logging.debug(f"self.mdp.getStates() = {self.mdp.getStates()}")

        # logging.debug(f"self.mdp.isTerminal(state) = {self.mdp.isTerminal(state)}")

        # Main Logic of Method
        # logging.debug("\n\n")
        # called("computeActionFromValues")
        # logging.debug(f"state (current) = {state}")

        actions = self.mdp.getPossibleActions(state)
        # logging.debug(f"actions = self.mdp.getPossibleActions(state) = {self.mdp.getPossibleActions(state)}")
        if not actions:
            return None

        bestAction = None
        bestValue = float('-inf')

        for action in actions:
            # logging.debug(f"action = {action}")

            valueOfAction = self.getQValue(state, action)

            # logging.debug(f"valueOfAction = {valueOfAction}")

            if valueOfAction > bestValue:
                bestAction = action
                bestValue = valueOfAction

        # logging.debug(f"bestAction = {bestAction}")

        logging.getLogger().setLevel(logging.INFO)
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        stateList = self.mdp.getStates()
        if not stateList:
            return

        stateID = -1
        for i in range(self.iterations):
            stateID += 1

            if stateID >= len(stateList):
                stateID = 0

            state = stateList[stateID]
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue

            bestAction = None
            bestValue = float('-inf')
            for action in actions:
                qValue = self.getQValue(state, action)
                if qValue > bestValue:
                    bestAction = action
                    bestValue = qValue

            self.values[state] = bestValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # You will now implement PrioritizedSweepingValueIterationAgent, which has been partially specified for you in valueIterationAgents.py. Note that this class derives from AsynchronousValueIterationAgent, so the only method that needs to change is runValueIteration, which actually runs the value iteration

        # Prioritized sweeping attempts to focus updates of state values in ways that are likely to change the policy.

        # For this project, you will implement a simplified version of the standard prioritized sweeping algorithm, which is described in this paper. We’ve adapted this algorithm for our setting. First, we define the predecessors of a state s as all states that have a nonzero probability of reaching s by taking some action 𝑎. Also, theta, which is passed in as a parameter, will represent our tolerance for error when deciding whether to update the value of a state.
        "*** YOUR CODE HERE ***"

        # Here’s the algorithm you should follow in your implementation.
        # • Compute predecessors of all states.
        def computePredecessors():
            """
            Compute the predecessors of all states and store them in a dictionary.
            Each state will map to a set of its predecessors.
            """
            predecessors = {}
            for state in self.mdp.getStates():
                predecessors[state] = set()

            for state in self.mdp.getStates():
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[nextState].add(state)

            return predecessors

        predecessors = computePredecessors()

        # • Initialize an empty priority queue.
        priorityQueue = util.PriorityQueue()

        # • For each non-terminal state s, do:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # – Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
                currentValue = self.values[state]
                actions = self.mdp.getPossibleActions(state)
                pQValues = [self.getQValue(state, action) for action in actions]
                diff = abs(currentValue - max(pQValues))
                # – Push s into the priority queue with priority -diff (note that this is negative).
                # We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                # This way, items with a bigger diff will be prioritized first.
                priorityQueue.push(state, -diff)

        # • For iteration in 0,1,2,… ,self.iterations − 1, do:
        for i in range(self.iterations):

            # – If the priority queue is empty, then terminate.
            if priorityQueue.isEmpty():
                break

            # – Pop a state s off the priority queue.
            state = priorityQueue.pop()

            # – Update s’s value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                pQValues = [self.getQValue(state, action) for action in actions]
                self.values[state] = max(pQValues)

            # – For each predecessor p of s, do:
            for predecessor in predecessors[state]:
                # * Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p
                # (this represents what the value should be).
                # Call this number diff.
                # Do NOT update self.values[p] in this step.
                pCurrentValue = self.values[predecessor]
                pActions = self.mdp.getPossibleActions(predecessor)
                pQValues = [self.getQValue(predecessor, action) for action in pActions]
                diff = abs(pCurrentValue - max(pQValues))

                # * If diff > theta, push p into the priority queue with priority −diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority.
                # As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
