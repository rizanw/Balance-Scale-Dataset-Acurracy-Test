
import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(1, 5):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(1, 5):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def manhattanDistance(instance1, instance2):
    distance = 0
    for x in range(1, 5):
        distance += abs(instance1[x] - instance2[x])
    return distance

def minkoswkiDistance(instance1, instance2):
    distance = 0
    for x in range(1, 5):
        distance += pow(abs(instance1[x] - instance2[x]), 3)
    return distance**(1/float(3))

def getNeighborsEuclidean(trainingSet, testInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getNeighborsManhattan(trainingSet, testInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist2 = manhattanDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist2))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getNeighborsMinkowski(trainingSet, testInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist3 = minkoswkiDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist3))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0

    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('balance-scale.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    
    k = 0
    i = 0
    for y in range(7):
        k += 1
        predictionsEuclidean = []
        predictionsManhattan = []
        predictionsMinkowski = []
        for x in range(len(testSet)):
            i += 1
            neighborsEuclidean = getNeighborsEuclidean(trainingSet, testSet[x], k)
            resultEuclidean = getResponse(neighborsEuclidean)
            predictionsEuclidean.append(resultEuclidean)
            neighborsManhattan = getNeighborsManhattan(trainingSet, testSet[x], k)
            resultManhattan = getResponse(neighborsManhattan)
            predictionsManhattan.append(resultManhattan)
            neighborsMinkowski = getNeighborsMinkowski(trainingSet, testSet[x], k)
            resultMinkowski = getResponse(neighborsMinkowski)
            predictionsMinkowski.append(resultMinkowski)
            
            #print('t-' + str(i) +'> predicted=' + repr(resultEuclidean) + ', actual=' + repr(testSet[x][0]))
            #print('t-' + str(i) +'> predicted=' + repr(resultManhattan) + ', actual=' + repr(testSet[x][0]))
            #print('t-' + str(i) +'> predicted=' + repr(resultManhattan) + ', actual=' + repr(testSet[x][0]))
        accuracyEuclidean = getAccuracy(testSet, predictionsEuclidean)
        print('k = ' + str(k))
        print('Accuracy for Euclidean Distance: ' + repr(accuracyEuclidean) + '%')
        accuracyManhattan = getAccuracy(testSet, predictionsManhattan)
        print('Accuracy for Manhattan Distance: ' + repr(accuracyManhattan) + '%')
        accuracyMinkowski = getAccuracy(testSet, predictionsMinkowski)
        print('Accuracy for Minkowski Distance: ' + repr(accuracyMinkowski) + '%')


main()