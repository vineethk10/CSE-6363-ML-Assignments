"""
CSE 6363-007: Machine Learning Assignment 5
Name: Ananthula, Vineeth Kumar. UTA ID: 1001953922

Tree plotter Implementation
"""

import matplotlib.pyplot as plt

nodeDecision = dict(boxstyle="sawtooth", fc="0.8")
nodeLeaf = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getLeafCount(myTree):
    LeafCount = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            LeafCount += getLeafCount(secondDict[key])
        else:
            LeafCount += 1
    return LeafCount


def getTreeDepth(myTree):
    depthMax = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > depthMax:
            depthMax = thisDepth
    return depthMax


def plotNode(nodeText, centerPoint, parentPoint, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPoint, xycoords='axes fraction',
                            xytext=centerPoint, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPoint, parentPoint, txtString):
    xMid = (parentPoint[0] - cntrPoint[0]) / 2.0 + cntrPoint[0]
    yMid = (parentPoint[1] - cntrPoint[1]) / 2.0 + cntrPoint[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPoint, nodeText):  # if the first key tells you what feat was split on
    LeafCount = getLeafCount(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPoint = (plotTree.xOff + (1.0 + float(LeafCount)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPoint, parentPoint, nodeText)
    plotNode(firstStr, cntrPoint, parentPoint, nodeDecision)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPoint, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPoint, nodeLeaf)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPoint, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getLeafCount(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]
