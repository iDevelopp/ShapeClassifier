import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import kit
import randmat
import shannon as sh

"""
ETAPE 1
Je génère 2000 données pour chaque classe.
"""
#kit.generer_image_classes(6,100,"data")

"""
ETAPE 2
On créer un dictionnaire sur 1000 images pour calculer l'entropie croisée 
entre chaque paire de PDF. On affiche la matrice.
"""
images = kit.creer_dic_image(1,1000)
#print(images)
pdf = kit.calcul_pdf(images)  # Calcul des densités de probabilité
#print(pdf)
mat_CE = kit.CE_pdf(pdf)  # Calcul de l'entropie croisée entre chaque paire de PDF
#print(mat_CE)
kit.afficher_matrice(mat_CE, "Entropies croisées entre classes d'images")

"""
ETAPE 3
Maintenant, on test notre fonction bruitage pour une image seulement dont on choisie
la classe, et on test sa prédiction.
"""

classe = 2
kit.afficher_image(kit.load_from_csv("img", classe, classe))
img_choisi = randmat.bruitage(kit.load_from_csv("img", classe, classe),0.5)
kit.afficher_image(img_choisi)
#print(img_choisi)
kit.predire(img_choisi, images, classe)

"""
ETAPE 4
On trace le taux d'accuracy pour chaque classe 
"""
def predireCE(image_bruite, pdfs, classe_reelle):
    entropies = []
    pdf_img_br = kit.calcul_pdf_single(image_bruite)
    for pdf_classe in pdfs:
        entropie = sh.Calculer_Entropie_Croisee(pdf_img_br, pdf_classe)
        entropies.append(entropie)
    prediction = np.argmin(entropies) + 1
    if prediction == classe_reelle:
        return 1
    else:
        return 0

def plot_accuracy_dB(images, n, pdf):
    # On génère une liste de décibels 
    dB_values = list(range(0, n))
    accuracy = [[],[],[],[],[],[]]
    for dB in dB_values:
        images_bruitees = {}  # Créer un dictionnaire pour stocker les images bruitées pour chaque classe
        for classe in images.keys():
            images_bruitees[classe] = []
            # Bruitage de chaque image
            for image in images[classe]:
                image_bruitee = randmat.bruitage(image, dB)
                images_bruitees[classe].append(image_bruitee)
        for classe in images_bruitees.keys():
            preds = 0
            for image in images_bruitees[classe]:
                classe_num = classe
                preds += predireCE(image, pdf, classe_num)
                #print(dB, preds, classe_num)
            preds/=10
            #print(preds)
            accuracy[classe-1]+=[preds]
    
    
    classes = ["coins", "diag_g", "diag_d", "centre", "vline", "hline"]
    
    for i in range(6):
        plt.plot(dB_values,accuracy[i],label=classes[i])
    
    dossier = "png"
    os.makedirs(dossier, exist_ok=True)
    nom = "AccuracyCE"
    if not nom.endswith(".png"):
            nom = nom + ".png"
    chemin = os.path.join(dossier, nom)
    
    plt.xlabel('Signal dB')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en fonction du signal dB')
    plt.legend()
    plt.savefig(chemin, bbox_inches='tight', pad_inches=0)
    plt.show()
    return accuracy


#j'utilise les images du data Test et non pas du data Train 
images2 = kit.creer_dic_image(1001,2000)

# Valeurs de signal dB à tester
acc1 = plot_accuracy_dB(images2, 15, pdf)
print(acc1)


"""
ETAPE 5
Maintenant on cherche à améliorer la détection dans le signal
"""


def divergence_kl_pixel(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

def calculer_pixels_discriminants(pdf, command):
    n_classes = len(pdf)
    pixels_discriminants = []
    for i in range(8):
        for j in range(8):
            px = []
            #il y a peu être un bug par ici, en essayant d'optimiser, je pense avoir surajuster le code
            for k in range(n_classes):  # On parcours toutes les classes
                for l in range(n_classes):  # On compare avec toutes les autres classes
                    if k != l:  # On évite de comparer un pixel avec lui-même
                        px.append(divergence_kl_pixel(pdf[k][i][j], pdf[l][i][j]))
            moyenne_px = np.mean(px)  # Calcule la moyenne des divergences de KL pour ce pixel
            pixels_discriminants.append((moyenne_px, i, j))  # Ajoute la moyenne et les indices du pixel
    pixels_discriminants = sorted(pixels_discriminants, reverse=True)   # Trie par ordre décroissant
    if command == "enlever":
        return pixels_discriminants[27:]  # Retourner les pixels à enlever
    elif command == "discriminants":
        return pixels_discriminants[:27]  # Retourner les pixels les plus importants



# Exemple d'utilisation diverses et variés du retour de notre calculer_pixels_discriminants 
pixels_discriminants = calculer_pixels_discriminants(pdf, "enlever")
positions = [(k[1], k[2]) for k in pixels_discriminants]
# Nous affichons différents résultats utiles pour l'analyse
print(pixels_discriminants)
print(positions)
#On cherche à extraire pour réutiliser plus tard la liste en brut pour éviter de tout recalculer à chaque fois
li = [k[0] for k in positions]
lj = [k[1] for k in positions]
print(li)
print(lj)

def afficher_pixels_discriminants(image_shape, pixels_discriminants):
    image_discriminants = np.zeros(image_shape)
    for pixel in pixels_discriminants:
        i, j = pixel[1], pixel[2]
        image_discriminants[i, j] = 1 
    plt.imshow(image_discriminants, cmap='gray')
    plt.title('Modèle Embedding')
    plt.show()

# le plus intéressant !!!
img_shape = (8,8)
afficher_pixels_discriminants(img_shape, calculer_pixels_discriminants(pdf, "discriminants")) 


"""
ETAPE 6
On réutilise les pixels discriminant du Modèle Embedding
dans l'évaluation de nos courbes d'accuracy
"""

def predireCE(image_bruite, pdfs, classe_reelle):
    entropies = []
    pdf_img_br = kit.calcul_pdf_single(image_bruite)
    for pdf_classe in pdfs:
        entropie = sh.Calculer_Entropie_Croisee(pdf_img_br, pdf_classe)
        entropies.append(entropie)
    prediction = np.argmin(entropies) + 1
    if prediction == classe_reelle:
        return 1
    else:
        return 0

def remplacer_pixels(image, li, lj, epsilon):
    for i, j in zip(li, lj):
        image[i][j] = epsilon


def plot_accuracy_vs_dB_optimisee(images, n, pdf): #On améliore notre fonction précédente
    epsilon = 0.0000000000000001
    li = [0, 7, 7, 1, 3, 1, 3, 3, 0, 3, 6, 6, 4, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
    lj = [4, 4, 3, 3, 7, 4, 6, 1, 3, 0, 4, 3, 6, 6, 5, 2, 1, 7, 5, 2, 0, 7, 6, 1, 0, 7, 6, 1, 0, 7, 5, 2, 0, 6, 5, 2, 1]
    # On génère une liste de décibels 
    dB_values = list(range(0, n))
    accuracy = [[],[],[],[],[],[]] #à améliorer si un jour on traite plus de 6 classes !!!!!
    for dB in dB_values:
        images_bruitees = {}  # Créer un dictionnaire pour stocker les images bruitées pour chaque classe
        for classe in images.keys():
            images_bruitees[classe] = []
            # Bruitage de chaque image
            for image in images[classe]:
                remplacer_pixels(image, li, lj, epsilon)
                image_bruitee = randmat.bruitage(image, dB)
                # Remplacer les pixels spécifiés par epsilon
                images_bruitees[classe].append(image_bruitee)
        for classe in images_bruitees.keys():
            preds = 0
            for image in images_bruitees[classe]:
                classe_num = classe
                preds += predireCE(image, pdf, classe_num)
                #print(dB, preds, classe_num)
            preds /= 10
            #print(preds)
            accuracy[classe-1] += [preds]
    
    classes = ["coins", "diag_g", "diag_d", "centre", "vline", "hline"]
    
    for i in range(6):
        plt.plot(dB_values, accuracy[i], label=classes[i])
    
    dossier = "png"
    os.makedirs(dossier, exist_ok=True)
    nom = "AccuracyCE"
    if not nom.endswith(".png"):
            nom = nom + ".png"
    chemin = os.path.join(dossier, nom)
    
    plt.xlabel('Signal dB')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en fonction du signal dB')
    plt.legend()
    plt.savefig(chemin, bbox_inches='tight', pad_inches=0)
    plt.show()
    return accuracy

# Utilisation des images du data Test et non pas du data Train 
# Valeurs de signal dB à tester
acc2 = plot_accuracy_vs_dB_optimisee(images2, 15, pdf)
print(acc2)


"""
ETAPE 7: 
On évalue cette optimisation en va pourcentage
"""

def calculer_amelioration(Db_j_etoile, Db_j, nclasse, ndB):
    assert len(Db_j_etoile) == len(Db_j), "Les listes Db_j_etoile et Db_j n'ont pas la même longueur."
    ameliorations_percentage = []
    for i in range(nclasse):
        tmp = []
        for j in range(ndB):
            if Db_j[i][j] != 0: 
                aux = 100 * (Db_j_etoile[i][j] - Db_j[i][j]) / Db_j[i][j]
                tmp.append(aux)
            else:
                tmp.append(0)
        ameliorations_percentage.append(sum(tmp))
    return ameliorations_percentage
ameliorations = calculer_amelioration(acc2, acc1, 6, 15) # on aurait pu généraliser par une variables globales
print(ameliorations)



"""
ETAPE 8: 
On peut tracer à la fois nos courbes avec et sans optimisation 
pour un meilleurs apperçu.
"""
def plot_accuracy(dB_values, accuracy_1, accuracy_2):
    classes = ["coins", "diag_g", "diag_d", "centre", "vline", "hline"]
    couleurs = ["blue", "orange", "green", "red", "purple", "brown"]
    
    fig, ax = plt.subplots()
    
    for i, c in enumerate(classes):
        ax.plot(dB_values, accuracy_1[i], label=c, color=couleurs[i])
        ax.plot(dB_values, accuracy_2[i], label=c, linestyle="dashed", color=couleurs[i])
    
    dossier = "png"
    os.makedirs(dossier, exist_ok=True)
    nom = "AccuracyCE.png"
    chemin = os.path.join(dossier, nom)
    
    ax.set_xlabel('Signal dB')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy en fonction du signal dB')
    ax.legend()
    plt.savefig(chemin, bbox_inches='tight', pad_inches=0)
    plt.show()

plot_accuracy( list(range(0, 15)), acc1, acc2)


