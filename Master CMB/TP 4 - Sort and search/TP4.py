####################################################################
#1 Sorting algorithm
####################################################################

def insertion_sort(arr) :
    i= 1
    n = len(arr)
    while i < n :
        x =  arr[i]
        j =  i
        
        while j > 0 and arr[j-1] > x :
            arr[j] = arr[j - 1]
            j =  j - 1
        
        arr[j] =  x
        i =  i + 1
    
    return arr

import time

def bubble_sort(arr) :
    i = 1
    n = len(arr)
    while i <= n-1 :
        j =  i
        
        while j>=0 and arr[j-1] > arr[j] :
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j=j-1
    
        i = i + 1
    
    return arr

array = [12,5,19,99,51,3,65,1]

#t0 = time.perf_counter()
#print(insertion_sort(array), time.perf_counter()-t0)

#t0 = time.perf_counter()
#print(bubble_sort(array), time.perf_counter()-t0)

def SiftDown(arr, start, end) : 
    n = len(arr)
    root = start
    while 2 * root + 1 <= end :
        child = 2 * root + 1
        sw = root
        if arr[sw] < arr[child] :
            sw = child
        
        if child + 1 <= end and arr[sw] < arr[child + 1] :
            sw = child + 1
        
        if sw == root :
            return arr
        else :
            arr[root], arr[sw]= arr[sw], arr[root]
        root = sw
    return arr

def Heapify(arr, end):
    n = len(arr)
    start =  (end-1)//2
    while start >= 0 :
        arr = SiftDown(arr, start, n-1)
        start = start - 1
    return arr

def heap_sort(arr):
    n = len(arr)
    arr = Heapify(arr, n - 1)
    end = n - 1
    while end > 0:
        arr[end], arr[0] = arr[0], arr[end]
        end = end - 1
        arr = SiftDown(arr, 0, end)
    return arr

####################################################################
#2 Complexity in practice
####################################################################

import numpy as np

#t0 = time.perf_counter()
#print(heap_sort(array), time.perf_counter()-t0)



x=[]
t_insert = []
for l in range(2,14):
    x.append(l)
    array = np.random.randint(0,2**l,2**l)
    t0 = time.perf_counter()
    insertion_sort(array)
    t_insert.append(time.perf_counter()-t0)
    
t_bubble = []
for l in range(2,14):
    array = np.random.randint(0,2**l,2**l)
    t0 = time.perf_counter()
    bubble_sort(array)
    t_bubble.append(time.perf_counter()-t0)

t_heap = []
for l in range(2,14):
    array = np.random.randint(0,2**l,2**l)
    t0 = time.perf_counter()
    heap_sort(array)
    t_heap.append(time.perf_counter()-t0)

import matplotlib.pyplot as plt
    
fig,ax = plt.subplots(2,1)
ax[0].plot(x,t_insert,label='insertion')
ax[0].plot(x,t_bubble,label='bubble')
ax[0].plot(x,t_heap,label='heap')
ax[0].set_title('temps pour trier selon 3 méthodes')
ax[0].legend()
ax[0].set_xlabel("taille du vecteur trié (en puissance de 2)")
ax[0].set_ylabel("temps (s)")
#plt.show()

####################################################################
#3 Search in sorted array
####################################################################

def seq_search(arr, val):
    """sequentialy search val in arr and return True or False"""
    for elem in arr:
        if elem == val:
            print(val, 'found')
            return True
    print(val, 'NOT found')
    return False
    
arr = []
for i in range(100):
    arr.append(i)
arr.pop(30)

#seq_search(arr,5) 
#seq_search(arr,30)

def dic_search(arr,val):
    """search val by dichotomy in arr"""
    n = len(arr)
    while n>1:
        n = n//2
        if val == arr[n-1]:
            print (val, 'found')
            return True
        elif val > arr[n-1]:
            arr = arr[n:]
            #print(arr)
        else:
            arr = arr[:n-1]
            #print(arr)
    #print(arr)
    if len(arr)==0 or arr[0]!=val:
        print(val, 'NOT found')
        return False
    else:
        print(val, 'found')
        return True

#dic_search(arr,1)
#dic_search(arr,66)
#dic_search(arr,30)



x = []
t_seq = []
t_dic = []

import random

for l in range(2,25):
    x.append(l)
    array = np.random.randint(0,2**l,2**l)
    array = np.sort(array)
    
    val = random.randint(0,2**l)
    
    t0 = time.perf_counter()
    seq_search(array,val)
    t_seq.append(time.perf_counter()-t0)
    
    t0 = time.perf_counter()
    dic_search(array,val)
    t_dic.append(time.perf_counter()-t0)


    

ax[1].plot(x,t_seq,label='sequential')
ax[1].plot(x,t_dic,label='dichotomous')
ax[1].legend()
ax[1].set_title('temps pour chercher une valeur selon 2 méthodes')
ax[1].set_xlabel("taille du dictionnaire (en puissance de 2)")
ax[1].set_ylabel("temps (s)")
plt.show()
    
