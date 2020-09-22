import numpy as np
import math

# This ID3 algorithm is built to work on any data set with a boolean target attribute

BINS = np.array([0,1,2])

rawData = np.genfromtxt("data/synthetic-1.csv", delimiter=",", dtype=float)
print(rawData.shape)
print(np.amin(rawData, axis=0))
print(np.amax(rawData, axis=0))

bins = np.linspace(-3.0, 13.0, 3) #Creates 3 bins equal size from -3 to 13 (min and max values I found)

binIndexOne = np.digitize(rawData[:, 0], bins) # Descretize attribute 1 to bins
binIndexTwo = np.digitize(rawData[:, 1], bins) # Descretize attribute 2 to bins

cleanData = np.array([binIndexOne, binIndexTwo, rawData[:, 2]]).transpose() #Creates a 2D array with the attributes seperated by bins

print(cleanData)


class Node(object): # Set parent to None if Root.
    def __init__(self, data, target, parent):
        self.data = data
        self.target = target
        self.parent = parent

    def distance_from_root(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.distance_from_root()

    def add_child(self, child):
        np.append(self.children, child)

def GetEntropy(examples, target):
    total_count = len(examples[:, target])
    negative_count = 0
    positive_count = 0
    if total_count == 0: return 0

    for i in examples[:, target]:
        if i == 0:
            negative_count += 1
        else:
            positive_count += 1

    if negative_count == 0 or positive_count == 0: return 1

    a = ((negative_count * -1) / total_count)
    b = ((positive_count * -1) / total_count)
    return  a * math.log(a, 2) - b * math.log(b, 2)

def GetBestInfoGain(examples, target, attributes):
    total = len(examples[:, target])
    entropy = GetEntropy(examples, target, attributes)
    attribute_entropys = []
    for attribute in attributes:
        binOne = np.array([])
        binTwo = np.array([])
        binThree = np.array([])
        for i in len(examples[:,attribute]):
            if i == 1:
                np.append(binOne,examples[i,target])
            if i == 2:
                np.append(binTwo,examples[i,target])
            if i == 3:
                np.append(binThree,examples[i,target])
        a = GetEntropy(binOne, 1)
        b = GetEntropy(binOne, 1)
        c = GetEntropy(binOne, 1)
        attribute_entropys[attribute] = entropy - ((a * len(binOne)/total) + (b * len(binTwo)/total) + (c * len(binThree)/total))
    return np.max(attribute_entropys)





def Id3(examples, targetAttribute, attributes, parent):

    node = Node(examples,