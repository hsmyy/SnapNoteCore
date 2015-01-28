import cv2
import os
from matplotlib import pyplot as plt

def labHist(resultList):
    for res in resultList:
        img = cv2.imread(res[0])
        im = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        color = ['r','g','b']
        lab = ['l','a','b']
        score = []
        for i, col in enumerate(lab):
            histr = cv2.calcHist([im],[i], None, [256],[0,256])
            # print histr
            # avg = sum(histr) / len(histr)
            avg = sum([j * data[0] for j,data in enumerate(histr)]) / sum([ data2[0] for data2 in histr])
            # print res[0] + "[" + col + "]:"
            # print avg
            score.append(str(avg))
            plt.plot(histr, color=color[i])
        print ','.join(score)

        plt.savefig(res[1])
        plt.clf()

def bgrHist(resultList):
    for res in resultList:
        img = cv2.imread(res[0])
        color = ['r','g','b']
        for i, col in enumerate(color):
            histr = cv2.calcHist([img],[i], None, [256],[0,256])
            # print histr
            # avg = sum(histr) / len(histr)
            avg = sum([j * data[0] for j,data in enumerate(histr)]) / sum([ data2[0] for data2 in histr])
            print res[0] + "[" + col + "]:" + str(avg)
            plt.plot(histr, color=col)
        plt.savefig(res[1])
        plt.clf()

input = "input/"
hist = "hist/"
files = os.listdir(input)
resultList = [ (input + f, hist + f) for f in files]
labHist(resultList)
# bgrHist(resultList)