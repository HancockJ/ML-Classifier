import numpy as np
import math


# This ID3 algorithm is built to work on any data set with a boolean target attribute

# BINS = np.array([0,1,2])
#
# rawData = np.genfromtxt("data/synthetic-1.csv", delimiter=",", dtype=float)
# print(rawData.shape)
# print(np.amin(rawData, axis=0))
# print(np.amax(rawData, axis=0))
#
# bins = np.linspace(-3.0, 13.0, 3) #Creates 3 bins equal size from -3 to 13 (min and max values I found)
#
# binIndexOne = np.digitize(rawData[:, 0], bins) # Descretize attribute 1 to bins
# binIndexTwo = np.digitize(rawData[:, 1], bins) # Descretize attribute 2 to bins
#
# cleanData = np.array([binIndexOne, binIndexTwo, rawData[:, 2]]).transpose() #Creates a 2D array with the attributes seperated by bins
#
# print(cleanData)
# rawData = np.genfromtxt("Example.csv", delimiter=",", dtype=float)


class Node(object):  # Set parent to None if Root.
    def __init__(self, data, label, distance):
        self.data = data
        self.label = label
        self.distance = distance
        self.children = None

    def add_child(self, child):
        np.append(self.children, child)


# feature is a dict key:pair feature:value
class dataNode(object):
    def __init__(self, features, currentFeature, data):
        self.features = features
        self.currentFeature = currentFeature
        self.data = data

    def getFeatures(self):
        return self.features

    def setFeature(self, feature):
        self.feature = feature

    def getExamplesWithValue(self, attribute, value):
        list = []
        for i in self.data[:, attribute]:
            if i == value:
                list.append(self.data[i, :])
        return list


def GetEntropy(examples, class_label):
    total_count = len(examples[:, class_label])
    negative_count = 0
    positive_count = 0
    if total_count == 0:
        return 0

    for i in examples[:, class_label]:
        if i == 0:
            negative_count += 1
        if i == 1:
            positive_count += 1

    if negative_count == 0 or positive_count == 0:
        return 0

    a = (negative_count / total_count)
    b = (positive_count / total_count)

    return -1 * a * math.log2(a) - b * math.log2(b)


def GetBestInfoGain(examples, target, attributes):
    print(examples)
    attribute_entropys = {}
    total = len(examples[:, target])
    entropy = GetEntropy(examples, target)
    for attribute in attributes:
        binOne = np.empty((0, 3), float)
        binTwo = np.empty((0, 3), float)
        binThree = np.empty((0, 3), float)
        for i in range(examples.shape[0]):
            if examples[i, attribute] == 0:
                binOne = np.append(binOne, np.array([examples[i, :]]), 0)
            if examples[i, attribute] == 1:
                binTwo = np.append(binTwo, np.array([examples[i, :]]), 0)
            if examples[i, attribute] == 2:
                binThree = np.append(binThree, np.array([examples[i, :]]), 0)
        a = GetEntropy(binOne, 2)
        b = GetEntropy(binTwo, 2)
        c = GetEntropy(binThree, 2)
        #attribute_entropys = np.append(attribute_entropys, np.array([[attribute, entropy - ((a * len(binOne)/total) + (b * len(binTwo)/total) + (c * len(binThree)/total))]]), 0)
        attribute_entropys[attribute] = entropy - ((a * len(binOne)/total) + (b * len(binTwo)/total) + (c * len(binThree)/total))
    print(attribute_entropys)
    maximum = max(attribute_entropys, key=attribute_entropys.get)
    return np.max(attribute_entropys)


def AllNegative(examples, label):
    for i in examples[:, 2]:
        if i == 1:
            return False
    return True


def AllPositive(examples, label):
    for i in examples[:, 2]:
        if i == 0:
            return False
    return True


def MostCommonValue(examples, label):
    total = 0
    for i in examples[:, 2]:
        total += i
    if total >= .5:
        return 1
    return 0


def Id3(data, label, attributes):
    distance = 0
    if AllNegative():
        return Node(data, 0, None)
    if AllPositive():
        return Node(data, 1, None)
    if len(attributes) == 0:
        return Node(data, MostCommonValue(), None)
    A = None  # MAKE THIS THE BEST ATTRIBUTE
    # set current feature to A
    newData = data
    newData.setFeature(A)
    root = Node(newData, label, distance)
    for x in range(3):
        childData = data.getExamplesWithValue(A, x)
        if len(childData) == 0:
            return

    # For every choice in the attribute
    # If choice == empty then add a child leaf with label = most common target value
    # Else, Recursion with data having the current feature and choice added to dataNode and attribute(A) removed


with open('Example.csv', 'r', encoding='utf-8-sig') as f:
    rawData = np.genfromtxt(f, dtype=float, delimiter=',')
#print(rawData.shape)
#print(rawData)
target_ = 2
entropy_ = GetEntropy(rawData, target_)
#print(entropy_)
attributes_ = [0, 1]
IG = GetBestInfoGain(rawData, target_, attributes_)
#print(IG)

