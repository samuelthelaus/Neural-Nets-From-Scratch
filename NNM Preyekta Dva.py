# Worked together with B081705, B081973

# Import random and numpy
import random
import numpy

class link(object):
    # Create an object 'link' that takes arguments node1, node2, weight
    # This object holds information on links between nodes in the network

    def __init__(self, node1, node2, weight):

        # link has 2 links and a weight between them
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

    def learn_fun(self):

        # Learning function takes activation of nodes in link and changes weight accordingly
        self.weight += (2 * self.node1.act - 1)*(2 * self.node2.act - 1)


class node(object):
    # Make object 'node' with defaults for activation, input, connection,
    # reset input, excite, and inhibit

    def __init__(self):
        self.act = 0.0  # No activation when initialized
        self.input = 0.0  # No input when initialized
        self.connection = []  # Connection holds information about links connected to node

    def act_fun(self):

        # Node is binary threshold unit
        # Threshold can be set to any value X, default is 1 here
        if self.input > 1:
            self.act = 1
        else:
            self.act = 0

    def input_fun(self):

        self.input = 0.0

        for links in self.connection:
            # Loop looking at each link in self.connection
            # Will find which node is not own node and add other node's activation to input

            #     if statement tests if node1 is self
            #     if true, add node2's activation * weight to own input,
            #     else add node1's activation * weight
            if links.node1 is self:
                self.input += links.weight * links.node2.act
            else:
                self.input += links.weight * links.node1.act


# Create list Network and put 16 nodes into it
Network = []

for x in range(16):
    Network.append(node())


# Each node looks at all other nodes in the network and create links between them
for node1 in Network:

    for node2 in Network:
        if node2 is not node1:
            node1.connection.append(link(node1, node2, 1.0))


# Different training sets depending on problem, uncomment relevant patterns

# Problem 1
#trainingSet = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]

# Problem 2

#trainingSet = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

# Problem 3

#trainingSet = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
#[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]]

# Problem 4

#base_pattern = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

#trainingSet = []

# Loop creating 10 mutations of base pattern (question 4)
#for i in range(10):

    #s = []

    #for j in base_pattern:

        # Pick a random number between 1 and 8, if the number is 1 (1/8 chance), then change unit in pattern
        #chance = random.randint(1, 8)

        #if chance == 1 and j == 1:
            #s.append(0)
        #elif chance == 1 and j == 0:
            #s.append(1)
        #else:
            #s.append(j)

    #trainingSet.append(s)


def run_hopfield():

    # Loop training network
    for pattern in trainingSet:

        # Impose pattern on network
        for p, nodes in enumerate(Network):

            nodes.act = pattern[p]

        # For all nodes, take each nodes' links and run learn function
        for nodes in Network:

            for links in nodes.connection:
                links.learn_fun()

    # Loop testing the network
    for p, pattern in enumerate(trainingSet):

        correctCounter = 0  # Correct counter set to 0

        mutated_pattern = []  # Empty list with testing pattern

        # Creating mutated patterns
        for j in pattern:

            # Pick a random number, increasing second number makes patterns mutate with lower probability
            chance = random.randint(1, 8)

            # If random number is 1, change the unit, j, and append to mutated pattern
            if chance == 1 and j == 1:
                mutated_pattern.append(0)
            elif chance == 1 and j == 0:
                mutated_pattern.append(1)
            else:
                mutated_pattern.append(j)

        # Impose mutated pattern on network
        for z, node in enumerate(Network):

            node.act = mutated_pattern[z]

        # Run network for a set number of iterations, x
        for x in range(200):

            # Randomly select a node by picking a random value to index the Network list with
            rand = random.randint(0, 15)

            # Update input and activation for that node
            Network[rand].input_fun()
            Network[rand].act_fun()

            act_list = []  # Empty list to store activations

            # Add activations of nodes in network
            for nodes in Network:
                act_list.append(nodes.act)

            # Calculate hamming distance by comparing activations to desired pattern
            ham = sum([abs(d-a) for d, a in zip(pattern, act_list)])

            # If hamming distance is 0, correctCounter increases by one, else it is reset to 0
            if ham == 0:
                correctCounter += 1
            else:
                correctCounter = 0

            # If correctCounter reaches a certain number, the network has had hamming distance 0 for long enough
            # Network is deemed to have settled, for loop is broken out of and results are printed
            if correctCounter == 20:
                print(str(p + 1)+'. Iterating stopped at iteration '+str(x + 1)+' for pattern '+str(act_list))
                print('            Mutated pattern used in this run was ' + str(mutated_pattern))
                print('           Training pattern used in this run was ' + str(pattern)+'\n')

                break

            # If the network reaches its final iteration it has not settled
            # Results are printed
            if x == 199:
                print(str(p + 1)+'. Ran full iterations')
                print('Final activations were '+str(act_list))
                print(' Mutated pattern used: '+str(mutated_pattern))
                print('Training pattern used: '+str(pattern))
                print('Hamming Distance is: '+str(ham)+'\n')


run_hopfield()

weights = []  # Create empty list of weights

# Loop taking all weights for all nodes
for node in Network:

    weight = []

    for link in node.connection:

        weight.append(link.weight)

    weights.append(weight)


# Change weight list to array
w_array = numpy.array(weights)

# Change trainingSet to array
train_array = numpy.array(trainingSet)

# Print weights and training patterns
print('These patterns were used for training: \n', train_array)
print('\nWeights for nodes 1 through 16 below: \n',  w_array)
