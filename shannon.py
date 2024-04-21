import copy

import numpy as np

#def Calculer_Entropie_Croisee(P,Q):
#    return -np.sum(P * np.log2(Q))

#Inspiré du code de Lucie Quarta
def Calculer_Entropie_Croisee(P1, P2): 
    X = P1.flatten()
    Y = P2.flatten()
    
    r = 0
    epsilon = 0.0000000000000001 #Sécurité pour éviter un artefact avec log
    for i in range(len(X)): #pour chaque index de PX
        if X[i] == 0:
            r = r - (Y[i] * (np.log2(epsilon)))
        else:
            r = r - (Y[i] * (np.log2(X[i])))
    Entropie_croisee = r
    return Entropie_croisee


def divergence_kl_pixel(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))



