# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:34:17 2023

@author: Didier
"""

import random
import matplotlib.pyplot as plt
import sys

def randlist(N):
    return [random.random() for i in range(N)]

def graph(N):
    fig, ax = plt.subplots(3,1)
    y = randlist(N)
    x = [y[i+1] for i in range(len(y)-1)]
    
    ax[0].plot(y)
    ax[1].scatter(x,y[:-1])
    ax[2].hist(y,bins = 10)
    plt.show()

if __name__ == "__main__" and len(sys.argv)==2:
    graph(int(sys.argv[1]))
