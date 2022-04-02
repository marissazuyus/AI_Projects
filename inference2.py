# inference.py
# ------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.

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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """

    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"

        total = self.total() # get total of all key values

        if total != 0:
            for k in self.keys():
                newVal = self[k]
                newVal = newVal / total # we divide each key by total sum to get proportional distribution which will lead to the total sum of values being 1
                self[k] = newVal # once normalized, we update the value of the keys
        return self
        raiseNotDefined()

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"

        total = self.total()
        randomNum = random.random() * total
        size = len(self)
        arr = [0] * (size + 1)
        arr2 = [0] * size

        arr[0] = 0
        index = 0
        cum = 0
        for k in self.keys():
            arr[index + 1] = self[k] + cum
            cum = arr[index + 1]
            arr2[index] = k
            index += 1

        for i in range(size):
            if arr[i] <= randomNum < arr[i + 1]:
                return arr2[i]

        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                                                               gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"

        if noisyDistance is None and ghostPosition == jailPosition: #if observation is none and the ghost is in jail
            return 1
        if noisyDistance is None and ghostPosition != jailPosition: #if observation is none and ghost is not in jail
            return 0
        if noisyDistance is not None and ghostPosition == jailPosition: #if observation is not none and the ghost is in jail
            return 0
        else:
            dist = manhattanDistance(pacmanPosition, ghostPosition) # we get the true distance between pacman and ghost using manhattanDistance()
            ans = busters.getObservationProbability(noisyDistance, dist)
            return ans
        raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.
        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # raiseNotDefined()

        self.beliefs.normalize() # we normalize all the values in the current list of beliefs in order to get each of the values in proportion to the total sum

        for ghost in self.allPositions:
            self.beliefs[ghost] = self.beliefs[ghost] * self.getObservationProb(observation, gameState.getPacmanPosition(), ghost, self.getJailPosition())
             #above we multiply each value in the beliefs list by the probability of each observation for the position of the ghost(which we get using getObservationProb())
        self.beliefs.normalize() # we normalize the updated beliefs again

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.
        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        temp_beliefs = self.beliefs.copy() #we create a copy list of beliefs


        finalDistrs = {}
        for initialGhostPos in self.allPositions: # take each initial ghost position in the list of all positions
            finalPosDistr = self.getPositionDistribution(gameState, initialGhostPos) # we find the new position distribution for ghost given its old position
            finalDistrs[initialGhostPos] = finalPosDistr # we then update the list of final position distributions with the value found above

        for finalGhostPos in self.allPositions:
            prob = 0 # probability for the new/final position  distribution of ghost is initialized to 0
            for initialGhostPos in self.allPositions:
                prob = prob + self.beliefs[initialGhostPos] * finalDistrs[initialGhostPos][finalGhostPos] # we update prob by adding the 1.) observation for the current position and 2.) the new position distribution given the current ghost position

            temp_beliefs[finalGhostPos] = prob # we update the copy of the beliefs list for each ghost position with the probabilities that we now calculated including new distribution for every one time elapse

        self.beliefs = temp_beliefs # put the values of the updated copied list into the original beliefs list

        # raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        numPos = len(self.legalPositions)
        partsPerPos = self.numParticles // numPos
        for position in self.legalPositions:
            for i in range(partsPerPos):
                self.particles.append(position)
        if self.numParticles % numPos != 0:
            for i in range(self.numParticles % numPos):
                self.particles.append(self.legalPositions[i])

        # raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"

        belief_dist = self.getBeliefDistribution() #get belief distribution
        discrete_dist = DiscreteDistribution() #get discrete distribution
        for particle in self.particles:
            observProb = self.getObservationProb(observation, gameState.getPacmanPosition(), particle,
                                                 self.getJailPosition()) #get probability of observed position for ghost

            if particle in discrete_dist: #if particle is in discrete distribution
                disWeight = discrete_dist[particle]
                disWeight = disWeight + observProb # change weight by adding probability of observation given pacman's position
                discrete_dist[particle] = disWeight #updte the weight of the particle
            else:
                discrete_dist[particle] = observProb #if particle is not part of discrete distribution then assign a weight to it(here, weight is probability of the observation given pacman position)

        discrete_dist.normalize() # normalize the list of particles

        if discrete_dist.total() > 0: #if particles have weight
            self.particles = [discrete_dist.sample() for _ in range(self.numParticles)] # resample from weighted list
        else: # if particles receive zero weight
            self.initializeUniformly(gameState) #re-initialize the list of particles
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"

        new_particles = [] #create a new list for particles
        for particle in self.particles:
            finalParticleDistr = self.getPositionDistribution(gameState, particle) # gives distribution over new positions given current particle
            sample = finalParticleDistr.sample() #take sample of new position distribution for particles advancing a time step
            new_particles.append(sample)  # add this sample to the new list of particles
        self.particles = new_particles #upadte the particles list with the new particle values
        # raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        dist = DiscreteDistribution()
        for particle in self.particles:
            if particle in dist:
                totalCount = dist[particle]
                totalCount = totalCount + 1
                dist[particle] = totalCount
            else:
                dist[particle] = 1
        dist.normalize()
        return dist
        # raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        particle_pos = itertools.product(self.legalPositions, repeat=self.numGhosts)# gives the cartesian product of legal position with itself for n (n= number of ghosts) times
        particle_pos = list(particle_pos)
        random.shuffle(particle_pos) # we shuffle the permutations yielded by the cartesian product
        self.particles = []
        particle_len=len(self.particles)
        while particle_len < self.numParticles: #while length of particles list is less than number of particles
            self.particles += particle_pos


        # raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacman_pos = gameState.getPacmanPosition() # get current pacman position
        particles = self.particles
        allGhostPositions = self.legalPositions # list of all legal positions for a ghost

        newDistr = DiscreteDistribution() #list of new distributions
        for particle in self.particles:
            prob = 1.0
            for i in range(self.numGhosts):
                ghost_dist = observation[i] # we take distance to ghost as the given observation
                jail_pos = self.getJailPosition(i) #get the jail position of current ghost as there are multiple
                if ghost_dist is not None: #if ghost distance has a value
                    prob *= self.getObservationProb(ghost_dist, pacman_pos, particle[i], jail_pos) #gives probability of an observation regarding ghost distance
                else: # else we just initialize probaility of observation for a particle as 0
                    particle = list(particle)
                    particle[i] = jail_pos
                    particle = tuple(particle)
                    prob = 0
            newDistr[particle] += prob # add the probability of observation calculated to new list of distributions
        newDistr.normalize() # assigns weights to particles in proportion such that total sum =1

        if newDistr.total() == 0: #if all particles receive zero weight
            self.initializeUniformly(gameState)
            newDistr = self.getBeliefDistribution()
            particles = self.particles
            #we recreate self.particles from prior belief distribution by calling initializeUniformly

        self.particles = []
        # we add the resample particles to self.particles
        for i in range(self.numParticles):
            self.particles.append(newDistr.sample())
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            for i in range(self.numGhosts):
                newDistr = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i]) #get distributions over new position for a ghost given all it's previous positions for oldParticle
                new_pos = newDistr.sample() # we draw a new position based on the previous positions of ghosts at the previous time step
                newParticle[i] = new_pos # then we update the list of ghost positions with the new position for the given particle
            # raiseNotDefined()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle)) #We append the updated particle with new position value to the new list for particles
        self.particles = newParticles # we now assign the new list to the original self.particles list


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
