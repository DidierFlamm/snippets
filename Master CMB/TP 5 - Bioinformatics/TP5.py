# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:36:04 2023

@author: Didier
"""
###############################################################################
#TASK 1 : HAMMING DISTANCE
###############################################################################

DNA_samples = ["ACCATACCTTCGATTGTCGTGGCCACCCTCGGATTACACGGCAGAGGTGC",
 "GTTGTGTTCCGATAGGCCAGCATATTATCCTAAGGCGTTACCCCAATCGA",
 "TTTTCCGTCGGATTTGCTATAGCCCCTGAACGCTACATGCACGAAACCAC",
 "AGTTATGTATGCACGTCATCAATAGGACATAGCCTTGTAGTTAACAG",
 "TGTAGCCCGGCCGTACAGTAGAGCCTTCACCGGCATTCTGTTTG",
 "ATTAAGTTATTTCTATTACAGCAAAACGATCATATGCAGATCCGCAGTGCGCT",
 "GGTAGAGACACGTCCACCTAAAAAAGTGA",
 "ATGATTATCATGAGTGCCCGGCTGCTCTGTAATAGGGACCCGTTATGGTCGTGTTCGATCAGAGCGCTCTA",
 "TACGAGCAGTCGTATGCTTTCTCGAATTCCGTGCGGTTAAGCGTGACAGA",
 "TCCCAGTGCACAAAACGTGATGGCAGTCCATGCGATCATACGCAAT",
 "GGTCTCCAGACACCGGCGCACCAGTTTTCACGCCGAAAGCATC",
 "AGAAGGATAACGAGGAGCACAAATGAGAGTGTTTGAACTGGACCTGTAGTTTCTCTG",
 "ACGAAGAAACCCACCTTGAGCTGTTGCGTTGTTGCGCTGCCTAGATGCAGTGG",
 "TAACTGCGCCAAAACGTCTTCCAATCCCCTTATCCAATTTAACTCACCGC",
 "AATTCTTACAATTTAGACCCTAATATCACATCATTAGACACTAATTGCCT",
 "TCTGCCAAAATTCTGTCCACAAGCGTTTTAGTTCGCCCCAGTAAAGTTGT",
 "TCAATAACGACCACCAAATCCGCATGTTACGGGACTTCTTATTAATTCTA",
 "TTTTTCGTGGGGAGCAGCGGATCTTAATGGATGGCGCCAGGTGGTATGGA"]

def Hamming(v,w):
    n = 0
    for i in range(len(v)):
        if v[i]!=w[i]:
            n+=1
    return n

print("Hamming distance between DNA_1 and DNA_2 =", Hamming(DNA_samples[0], DNA_samples[1]))

###############################################################################
#TASK 2 : LEVENSHTEIN DISTANCE
###############################################################################

import numpy as np

def Levenshtein(s1,s2):
    m = len(s1)
    n = len(s2)
    d= np.zeros((m+1,n+1))
    
    for i in range(1,m+1):
        d[i, 0] = i
    
    for j in range(1,n+1):
        d[0, j] = j
    
    for j in range(1, n+1):
        
        for i in range(1, m+1):
            
            insertCost =  d[i - 1, j] + 1
            deleteCost =  d[i, j - 1] + 1
            if s1[i-1] == s2[j-1]:
                subCost =  d[i - 1, j - 1]
            else:
                subCost =  d[i - 1, j - 1] + 1
            
            d[i, j] =  min(insertCost, deleteCost, subCost)
    #print(d)
    return d[m,n]

print('Levenshtein distance between kryptonite and python =', Levenshtein('kryptonite','python'))

d = len(DNA_samples)
M = np.zeros((d,d))

for i in range(d):
    for j in range(d):
        M[i,j]=Levenshtein(DNA_samples[i],DNA_samples[j])

print(M[0])

np.savetxt('Lmatrix.txt',M, fmt = '%i')

###############################################################################
#TASKS 3,4,5 : SMITH-WATERMAN ALGORITHM
###############################################################################

#pip install swalign

import swalign

dna_string = "python"
reference_string = "kryptonite"
match_score = 2
mismatch_score = -1
matrix = swalign.NucleotideScoringMatrix(match_score, mismatch_score)
lalignment_object = swalign.LocalAlignment(matrix)
alignment_object = lalignment_object.align(dna_string, reference_string)
alignment_object.dump()

dna_string = DNA_samples[0]
reference_string = DNA_samples[1]
alignment_object = lalignment_object.align(dna_string, reference_string)
alignment_object.dump()

