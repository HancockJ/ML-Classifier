import numpy as np
import math


class Node(object):
    def __init__(self, feature, target, label, children, data):
        self.feature = feature
        self.label = label
        self.target = target
        self.children = children
        self.data = data

    def setFeature(self, feature, value):
        self.feature = (feature, value)

    def addChildren(self, children):
        self.children = children

    def findSuccessRate(self):
        correct = 0
        total = 0
        if len(self.data[:, self.label]) == 0 or self.target is not None:
            for i in self.data[:, self.label]:
                total += 1
                if i == self.target:
                    correct += 1
        if self.children is not None:
            for child in self.children:
                tempC, tempT = child.findSuccessRate()
                correct += tempC
                total += tempT
        return correct, total

    def print_tree(self, distance):
        if distance == 0:
            print("---------------------")
            print("ROOT OF TREE")
            correct, total = self.findSuccessRate()
            print("TOTAL SUCCESS RATE: " + str(correct) + "/" + str(total) + " = " + str(correct/total))
        else:
            correct, total = self.findSuccessRate()
            if total == 0:
                print("0 Nodes in sub tree")
            else:
                print("Success rate of sub tree: " + str(correct) + "/" + str(total) + " = " + str(correct/total))
        print("---------------------")
        print("Parent, distance: " + str(distance))
        if self.label is not None:
            print("Feature: " + str(self.feature))
            print("Label: " + str(self.label))
            print("Children:")
        if self.children is not None:
            index = 0
            for child in self.children:
                print("  - Child " + str(index)
                      + ", Feature: " + str(child.feature) + ", Label: " + str(child.label))
            for child in self.children:
                print(child.print_tree(distance + 1))
                


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
    attribute_entropys = {}
    total = len(examples[:, target])
    entropy = GetEntropy(examples, target)
    for attribute in attributes:
        binOne = np.empty((0, 3), float)
        binTwo = np.empty((0, 3), float)
        binThree = np.empty((0, 3), float)
        for i in range(examples.shape[0]):
            if examples[i, attribute] == 0:
                binOne = np.append(binOne, np.array([examples[i, :3]]), 0)
            if examples[i, attribute] == 1:
                binTwo = np.append(binTwo, np.array([examples[i, :3]]), 0)
            if examples[i, attribute] == 2:
                binThree = np.append(binThree, np.array([examples[i, :3]]), 0)
        a = GetEntropy(binOne, 2)
        b = GetEntropy(binTwo, 2)
        c = GetEntropy(binThree, 2)
        attribute_entropys[attribute] = entropy - (
                (a * len(binOne) / total) + (b * len(binTwo) / total) + (c * len(binThree) / total))
    return max(attribute_entropys, key=attribute_entropys.get)


def AllNegative(examples, label):
    for i in examples[:, label]:
        if i == 1:
            return False
    return True


def AllPositive(examples, label):
    for i in examples[:, label]:
        if i == 0:
            return False
    return True


def MostCommonValue(examples, label):
    total = 0
    count = 0
    for i in examples[:, label]:
        total += i
        count += 1
    average = total / count
    if average >= .5:
        return 1
    return 0


def splitByValue(data, feature, value):
    group = np.zeros((0, 3), float)
    index = 0
    for i in data[:, feature]:
        if i == value:
            group = np.append(group, np.array([data[index, :]]), 0)
        index += 1
    return group


def ID3(data, label, attributes, distance):
    if distance > 3:
        return Node(None, MostCommonValue(data, label), label, None, data)
    if AllNegative(data, label):
        return Node(None, 0, label, None, data)
    if AllPositive(data, label):
        return Node(None, 1, label, None, data)
    if len(attributes) == 0:
        return Node(None, MostCommonValue(data, label), label, None, data)
    A = GetBestInfoGain(data, label, attributes)
    children = []
    newAttributes = attributes
    newAttributes.remove(A)
    for x in range(3):
        split = splitByValue(data, A, x)
        if len(split) == 0:
            children = np.append(children, [Node(None, MostCommonValue(data, label), label, None, split)])
            children[-1].setFeature(A, x)
        else:
            children = np.append(children, [ID3(split, label, newAttributes, distance + 1)])
            children[-1].setFeature(A, x)
    return Node(None, None, label, children, data)


with open('data/synthetic-4.csv', 'r', encoding='utf-8-sig') as f:
    rawData = np.genfromtxt(f, dtype=float, delimiter=',')

# MAIN
# This ID3 algorithm is built to work on any data set with a boolean target attribute

BINS = np.array([0, 1, 2])

if np.amin(rawData, axis=0)[0] < np.amin(rawData, axis=0)[1]:
    bottom = np.amin(rawData, axis=0)[0] - .01
else:
    bottom = np.amin(rawData, axis=0)[1] - .01

if np.amax(rawData, axis=0)[0] > np.amax(rawData, axis=0)[1]:
    top = np.amax(rawData, axis=0)[0] + .01
else:
    top = np.amax(rawData, axis=0)[1] + .01

bins = np.linspace(bottom, top, 3)

binIndexOne = np.digitize(rawData[:, 0], bins)  # Descretize attribute 1 to bins
binIndexTwo = np.digitize(rawData[:, 1], bins)  # Descretize attribute 2 to bins

cleanData = np.array(
    [binIndexOne, binIndexTwo, rawData[:, 2]]).transpose()  # Creates a 2D array with the attributes seperated by bins


tree = ID3(cleanData, 2, [0, 1], 0)
print(tree.print_tree(0))
