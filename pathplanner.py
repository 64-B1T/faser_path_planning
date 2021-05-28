from faser_math import FASER as fsr
from faser_math.tm import *
import numpy as np
from rtree import index
import random
from faser_math import FASER as fsr
from faser_utils.disp.disp import *


class R6Tree:

    def __init__(self, dimension = 6):
        p = index.Property()
        self.dimension = dimension
        p.dimension = self.dimension
        self.idx = index.Index(properties=p)
        self.count = 0

    def place(self, node):
        b = node.getPosition()
        #self.idx.insert(100, (b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5]), node)
        if self.dimension == 6:
            self.idx.insert(100, (b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5]), node)
        else:
            self.idx.insert(100, (b[0], b[1], b[2], b[0], b[1], b[2]), node)
        self.count+=1

    def nearestNeighbors(self, node, n):
        b = node.getPosition()
        #return list(self.idx.nearest((b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5]), n))
        if self.dimension == 6:
            return list(self.idx.nearest((b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5]), n, objects = True))
        else:
            return list(self.idx.nearest((b[0], b[1], b[2], b[0], b[1], b[2]), n, objects = True))

    def intersection(self, node):
        b = node.getPosition()
        #return list(self.idx.intersection((b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5])))
        if self.dimension == 6:
            return list(self.idx.intersection((b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5])))
        else:
            return list(self.idx.intersection((b[0], b[1], b[2], b[0], b[1], b[2])))

    def containsBounded(self, L, R):
        intersects = []
        if self.dimension == 6:
            intersects =  list(self.idx.intersection((L[0], L[1], L[2], L[3], L[4], L[5], R[0], R[1], R[2], R[3], R[4], R[5])))
        else:
            intersects =  list(self.idx.intersection((L[0], L[1], L[2], R[0], R[1], R[2])))
        return not (len(intersects) == 0)

    def getAll(self):
        return self.nearestNeighbors(PathNode(tm()), self.count)

    def getNeighborChain(self, node, num):
        chain = []
        for i in range(num):
            if(node.getParent() != None):
                chain.append(node.getParent())
                node = node.getParent()
            else:
                break
        return chain

    def getCount(self):
        return self.count

class Tree6Node:

    def __init__(self, node, parent = None):
        self.data = node
        self.parent = parent
        self.size = 1
        self.level = 1
        self.children = [None] * 64

    def place(self, node):
        ind = self.findChildInd(node)
        if self.children[ind] == None:
            self.children[ind] = Tree6Node(node, self)
            self.children[ind].level = self.level + 1
            self.size += 1
            return
        else:
            self.children[ind].place(node)
            self.size += 1
            return

    def getSize(self):
        return self.size

    def find(self, node):
        ind = self.findChildInd(node)
        if (self.children[ind] == None):
            return -1
        if (self.children[ind].getSize() == 1):
            if (self.children[ind].data == node):
                return self.children[ind].data
            print("Didn't Match")
            print(node.getPosition())
            print(self.children[ind].data.getPosition())
            return -1
        return self.children[ind].find(node)


    def delete(self, node):
        ind = self.findChildInd(self, node)
        if (self.children[ind] == None):
            return False
        if (self.children[ind].getSize() == 1):
            self.children[ind] == None
            self.size -= 1
            return True
        if self.children[ind].delete(node):
            self.size -= 1
            return True
        return False

    def findChildInd(self, node):
        ind = 0
        if(node.getPosition()[0] < self.data.getPosition()[0]):
            ind += 32
        if(node.getPosition()[1] < self.data.getPosition()[1]):
            ind += 16
        if(node.getPosition()[2] < self.data.getPosition()[2]):
            ind += 8
        if(node.getPosition()[3] < self.data.getPosition()[3]):
            ind += 4
        if(node.getPosition()[4] < self.data.getPosition()[4]):
            ind += 2
        if(node.getPosition()[5] < self.data.getPosition()[5]):
            ind += 1
        return ind

    def print(self):
        print(self.printhelper(""))


    def printhelper(self, strfx):
        strx = strfx + self.data.getPosition().__str__() + "\n"
        for i in range(64):
            if (self.children[i] == None):
                #strx += strfx + "Empty" + "\n"
                a = 0
            else:
                strx += self.children[i].printhelper(strfx + "  ")
        return strx




class PathNode:

    def __init__(self, position = None, parent = None, mode = 3):
        self.position = position
        self.parent = parent
        self.mode = mode
        self.children = []
        self.cost = 0
        self.type = 0

    def setChild(self, child):
        self.children.append(child)

    def setParent(self, parent):
        parent.setChild(self)
        self.parent = parent

    def removeChild(self, child):
        if child in self.children:
            self.children.remove(child)

    def setCost(self):
        self.cost = self.parent.getCost() +self.getDistance()

    def getDistance(self, other = None):
        if other != None:
            if(self.mode == 3):
                return fsr.Distance(self.getPosition(), other.getPosition())
            else:
                return fsr.ArcDistance(self.getPosition(), other.getPosition())
        if(self.mode == 3):
            self.cost = self.parent.getCost() + fsr.Distance(self.position, self.parent.getPosition())


    def getCost(self):
        return self.cost

    def getPosition(self):
        return self.position

    def getParent(self):
        return self.parent

    def __eq__(self, a):
        try:
            eq = sum(abs(self.getPosition().gTAA()-a.getPosition().gTAA())) < .0001
            return eq
        except:
            return False

class Graph:

    def __init__(self, init = None):
        self.nodeList = []

        if init != None:
            self.nodeList.append(init)

    def findClosest(self, node):
        closeind = 0
        maxdist = 9999999999
        for i in range(len(self.nodeList)):
            dist = self.nodeList[i].getDistance(node)
            if dist < maxdist:
                maxdist = dist
                closeind = i
        return closeind

    def getNode(self, ind):
        return self.nodeList[ind]

class DualStar:

    def __init__(self, origin, goal):
        #Purpose built only to find path from origin to goal, hopefully better and faster than RRTStar
        print("Does Nothing Yets")
        self.dimension = 0


class RRTStar:
    #Adapted from here https://arxiv.org/pdf/1105.1186.pdf
    def __init__(self, origin = None):

        self.dimension = 6
        self.dmode = 0
        self.G = R6Tree(self.dimension)
        self.obstructions = []
        self.bounds = [[-10, 10], [-10, 10], [-10, 10], [-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]
        self.pathRejectionDistance = 5
        self.nearestNeighborsLim = 15
        self.maxDist = 100
        self.minDist = 0.1
        self.iterations = 1500
        if origin != None:
            self.G.place(PathNode(origin))
        else:
            self.G.place(PathNode(tm()))

    def Obstructed(self, node, node2):
        for obstruction in self.obstructions:
            pos = node.getPosition()
            if (pos[0] >= obstruction[0][0] and pos[0] <= obstruction[1][0]
             and pos[1] >= obstruction[0][1] and pos[1] <= obstruction[1][1]
             and pos[2] >= obstruction[0][2] and pos[2] <= obstruction[1][2]
             and pos[3] >= obstruction[0][3] and pos[3] <= obstruction[1][3]
             and pos[4] >= obstruction[0][4] and pos[4] <= obstruction[1][4]
             and pos[5] >= obstruction[0][5] and pos[5] <= obstruction[1][5]):
                return True
        return False

    def Obstruction(self, n1, n2):
        for obstruction in self.obstructions:
            p1 = n1.getPosition()
            p2 = n2.getPosition()
            b1 = obstruction[0]
            b2 = obstruction[1]

            mid = np.array([(b2[0] + b1[0])/2, (b2[1] + b1[1])/2, (b2[2] + b1[2])/2])
            a = np.array([p1[0] - mid[0], p1[1] - mid[1], p1[2] - mid[2]])
            b = np.array([p2[0] - mid[0], p2[1] - mid[1], p2[2] - mid[2]])

            extends = np.abs(b2[0:3].reshape((3))-mid)

            LMid = (a + b) / 2
            L = (a - LMid)
            LExt = np.abs(L)

            if abs(LMid[0]) > extends[0] + LExt[0]: continue
            if abs(LMid[1]) > extends[1] + LExt[1]: continue
            if abs(LMid[2]) > extends[2] + LExt[2]: continue

            if abs(LMid[1] * L[2] - LMid[2] * L[1]) > (extends[1] * LExt[2] + extends[2] * LExt[1]):
                continue
            if abs(LMid[0] * L[2] - LMid[2] * L[0]) > (extends[0] * LExt[2] + extends[2] * LExt[0]):
                continue
            if abs(LMid[0] * L[1] - LMid[1] * L[0]) > (extends[0] * LExt[1] + extends[1] * LExt[0]):
                continue
            return True
        return False

    def ArmObstruction(self, arm, x, y):
        if(self.Obstruction(x, y)):
            return True
        startind = 0

        while (sum(arm.S[3:6,startind]) == 1):
            startind = startind + 1

        poses = arm.getJointTransforms()
        for i in range(startind, len(poses[startind:])):
            if poses[i] == None:
                continue
        Dims = np.copy(arm._Dims).T
        dofs = arm.S.shape[1]
        for i in range(startind, dofs):
            zed = poses[i]
            try:
                Tp = fsr.TMMidPoint(poses[i], poses[i+1])
                T = fsr.TMMidRotAdjust(Tp ,poses[i], poses[i+1], mode = 1)
                dims = Dims[i+1,0:3]
                dx = dims[0]
                dy = dims[1]
                dz = dims[2]
                corners = .5 * np.array([[-dx,-dy,-dz],[dx, -dy, -dz],[-dx, dy, -dz],[dx, dy, -dz],[-dx, -dy, dz],[dx, -dy, dz],[-dx, dy, dz],[dx, dy, dz]]).T
                Tc = np.zeros((3,8))
                for i in range(0,8):
                    h = T.gTM() @ np.array([[corners[0,i]],[corners[1,i]],[corners[2,i]],[1]])
                    Tc[0:3,i] = np.squeeze(h[0:3])
                segs = np.array([[1, 2],[1, 3],[2, 4],[3, 4],[1, 5],[2, 6],[3, 7],[4, 8],[5, 6],[5, 7],[6, 8],[7,8]])-1
                #disp(Tc)
                for i in range(12):
                    a = segs[i,0]
                    b = segs[i,1]
                    if self.Obstruction(a, b):
                        return True
            except:
                pass
        return False

    def addObstruction(self, L, R):
        self.obstructions.append([tm([L[0], L[1], L[2], -2*np.pi, -2*np.pi,-2*np.pi]), tm([R[0], R[1], R[2], 2*np.pi, 2*np.pi,2*np.pi])])

    def generateTerrain(self, xd, yd, xc, yc, zvar, xs = 0, ys = 0):
        cx = int(xd/xc)
        cy = int(yd/yc)
        for i in range(cx):
            for j in range(cy):
                h = random.uniform(0.1, zvar + .1)
                self.addObstruction([xc * i + xs, yc * j + ys, 0.1], [xc * (i+1) + xs, yc*(j+1) + ys, h])


    def RandomPos(self):
        pos = [None] * 6
        for i in range(6):
            pos[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
        postf = tm(pos)
        pnode = PathNode(postf)
        return pnode

    def Distance(self, pos1, pos2):
        if self.dmode == 1:
            return fsr.ArcDistance(pos1, pos2)
        else:
            return fsr.Distance(pos1, pos2)

    def generalGenerateTree(self, randomGenerator, distanceFunction, collisionDetector):
        for i in range(self.iterations):
            progressBar(i, self.iterations)
            newNode = randomGenerator()
            nearest = self.G.nearestNeighbors(newNode, 1)
            dist = distanceFunction(newNode.getPosition(), nearest[0].object.getPosition())
            while dist > self.maxDist or dist < self.minDist or collisionDetector(newNode, nearest[0].object):
                newNode = randomGenerator()
                nearest = self.G.nearestNeighbors(newNode, 1)
                dist = distanceFunction(newNode.getPosition(), nearest[0].object.getPosition())
            if len(nearest) == 0:
                self.G.place(newNode)
                continue
            costNew = distanceFunction(newNode.getPosition(), nearest[0].object.getPosition()) + nearest[0].object.getCost()
            newNode.cost = costNew
            newNode.setParent(nearest[0].object)
            nearest = self.G.nearestNeighbors(newNode, self.nearestNeighborsLim)
            #nearest = self.G.getNeighborChain(newNode, 15)
            for j in range(0, len(nearest)):
                #if newNode.cost + self.Distance(newNode.getPosition(), nearest[i].object.getPosition()) < nearest[i].object.getCost():
                #    nearest[i].object.cost = newNode.cost + self.Distance(newNode.getPosition(), nearest[i].object.getPosition())
                #    nearest[i].object.setParent(newNode)
                if distanceFunction(newNode.getPosition(), nearest[j].object.getPosition()) + nearest[j].object.getCost() < newNode.cost and not collisionDetector(newNode, nearest[j].object):
                    newNode.cost = distanceFunction(newNode.getPosition(), nearest[j].object.getPosition()) + nearest[j].object.getCost()
                    newNode.setParent(nearest[j].object)
            #for i in range(0, len(nearest)):
            #    if newNode.cost + self.Distance(newNode.getPosition(), nearest[i].getPosition()) < nearest[i].getCost():
            #        nearest[i].cost = newNode.cost + self.Distance(newNode.getPosition(), nearest[i].getPosition())
            #        nearest[i].setParent(newNode)
            self.G.place(newNode)

    def generateTree(self):
        self.generalGenerateTree(lambda : self.RandomPos(), lambda x, y : self.Distance(x, y), lambda x, y : self.Obstruction(x,y))

    def generateTreeDual(self):
        for i in range(self.iterations):
            progressBar(i, self.iterations)
            newNode = self.RandomPos()
            nearest = self.G.nearestNeighbors(newNode, 1)
            while not self.Obstructed(newNode) or self.Distance(newNode.getPosition(), nearest[0].object.getPosition()) > self.maxDist:
                newNode = self.RandomPos()
                nearest = self.G.nearestNeighbors(newNode, 1)
            if len(nearest) == 0:
                self.G.place(newNode)
                continue
            costNew = self.Distance(newNode.getPosition(), nearest[0].object.getPosition()) + nearest[0].object.getCost()
            newNode.cost = costNew
            newNode.setParent(nearest[0].object)
            newNode.type = nearest[0].object.type

            nearest = self.G.nearestNeighbors(newNode, self.nearestNeighborsLim)
            #nearest = self.G.getNeighborChain(newNode, 15)
            for j in range(0, len(nearest)):
                #if newNode.cost + self.Distance(newNode.getPosition(), nearest[i].object.getPosition()) < nearest[i].object.getCost():
                #    nearest[i].object.cost = newNode.cost + self.Distance(newNode.getPosition(), nearest[i].object.getPosition())
                #    nearest[i].object.setParent(newNode)
                if self.Distance(newNode.getPosition(), nearest[j].object.getPosition()) + nearest[j].object.getCost() < newNode.cost:
                    if(newNode.type == 0 and nearest[j].object.type == 1):
                        temp = nearest[j].object.getParent()
                        tempold = nearest[j].object
                        prev = newNode
                        while(temp != None):
                            tempold.setParent(prev)
                            prev = tempold
                            tempold.type = 0
                            tempold = temp
                            print(temp.getCost())
                            temp = temp.getParent()

                        break
                    newNode.cost = self.Distance(newNode.getPosition(), nearest[j].object.getPosition()) + nearest[j].object.getCost()
                    newNode.setParent(nearest[j].object)


                    newNode.type = nearest[j].object.type
            #for i in range(0, len(nearest)):
            #    if newNode.cost + self.Distance(newNode.getPosition(), nearest[i].getPosition()) < nearest[i].getCost():
            #        nearest[i].cost = newNode.cost + self.Distance(newNode.getPosition(), nearest[i].getPosition())
            #        nearest[i].setParent(newNode)
            self.G.place(newNode)

    def findPath(self, goal):
        self.generateTree()
        closest = self.G.nearestNeighbors(PathNode(goal), 1)[0].object
        posList = []
        while closest != None:
            posList.insert(0, closest.getPosition())
            closest = closest.getParent()
        posList.append(goal)
        return posList

    def findPathGeneral(self, treeMethod, goal):
        treeMethod()
        closest = self.G.nearestNeighbors(PathNode(goal), 1)[0].object
        posList = []
        while closest != None:
            posList.insert(0, closest.getPosition())
            closest = closest.getParent()
        posList.append(goal)
        return posList

    def findPathDual(self, goal):
        goalpath = PathNode(goal)
        goalpath.type = 1
        self.G.place(goalpath)
        self.generateTreeDual()
        closest = self.G.nearestNeighbors(PathNode(goal), 1)[0].object
        posList = []
        while closest != None:
            posList.insert(0, closest.getPosition())
            closest = closest.getParent()
        posList.append(goal)
        return posList
