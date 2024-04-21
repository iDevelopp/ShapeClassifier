import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import randmat
import shannon

"""
Attention, ce module est un peu brouillon, des noms de 
fonctions doivent etre changé et certaines mises à jour depuis le main.py
"""

# Affichage de l'image générée par la matrice
def afficher_image(image):
    plt.imshow(image, cmap='gray')
    plt.colorbar() #barre de couleur pour l'intensité des valeurs
    plt.show()

#Créer un fichier csv pour une seule matrice d'image
def creer_image(image, nom):
    afficher_image(image)
    with open(nom, 'w', newline='') as f:
        writer = csv.writer(f)
        for l in image :
            writer.writerow(l)

# Générer les fichiers CSV pour chaque classe d'images (ici 6)
def generer_image_classes(nbclasse, img_classe, nom_fichier):
    os.makedirs("img", exist_ok="True") 
    for i in range(nbclasse):
        dossier_classe = os.path.join("img", f"Classe_{i+1}")
        os.makedirs(dossier_classe,exist_ok=True)
        for j in range(1000,img_classe):
            filename = os.path.join(dossier_classe, f"image_{j+1}.csv")
            if i==0:
                creer_image(randmat.gen_coins(), filename)
            elif i==1:
                creer_image(randmat.gen_diag_droite(), filename)
            elif i==2:
                creer_image(randmat.gen_diag_gauche(), filename)
            elif i==3:
                creer_image(randmat.Get_Matrice_Centre(), filename)
            elif i==4:
                creer_image(randmat.gen_random_vline(), filename)
            elif i==5:
                creer_image(randmat.gen_random_hline(), filename)
    
    print(nom_fichier + " a été généré")

#Fonctions pour tester la génération de nos matrices pour n tirages
def tester_matrice(n):
    for i in range(n):
        afficher_image(randmat.gen_coins())
        afficher_image(randmat.gen_diag_gauche())
        afficher_image(randmat.gen_diag_droite())
        afficher_image(randmat.gen_centre())


def calcul_classe_moyenne(A):
    bary = np.mean(A, axis=0)  # pour chaque colonne
    cardinal = A.shape[0]  # Nombre total d'échantillons dans la classe
    modele_moyen = bary / cardinal  # Calcul du Modele_Moyen
    return modele_moyen


def calcul_pdf(data):
    pdf = []
    #data1 = copy.deepcopy(data)
    #print(data.values())
    for images in data.values():  # Parcours des images pour chaque classe
        res = np.zeros(images[0].shape, dtype=float)  # Initialisation de la somme des images
        for img in images:
            res += img  # Somme des images
        moyenne_images = res / len(images)  # Calcul de la moyenne des images
        somme_modele = np.sum(moyenne_images)  # Somme des valeurs du modèle moyen
        pdf_image = moyenne_images / somme_modele  # Calcul de la densité de probabilité
        pdf.append(pdf_image)
    return pdf


# Calculer l'entropie croisée entre chaque paire de PDF
def CE_pdf(dic_pdf):
    CE = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            epsilon = 10**(-16)  # Petite valeur constante
            CE[i,j] = shannon.Calculer_Entropie_Croisee(dic_pdf[i] + epsilon, dic_pdf[j] + epsilon).flatten()
    return CE



def tester(n):
    for i in range(n):
        kit.afficher_image(randmat.Get_Matrice_Centre())


def load_bruitee_from_csv(path: str, classe: int, image: int) -> np.ndarray:
    """Charge une image bruitée spécifique depuis un fichier CSV"""
    classe_folder = f"Classe_{classe}"
    image_filename = f"{classe}_{image}_bruite.csv"  
    image_path = os.path.join(path, classe_folder, image_filename)

    with open(image_path, 'r') as f:
        lines_read = csv.reader(f)
        image_data = []
        for line in lines_read:
            image_data.append([float(x) for x in line])

    return np.array(image_data, dtype=np.float64)


def load_bruitees_range_from_csv(path: str, classe: int, n: int) -> List[np.ndarray]:
    """Charge une plage d'images bruitées pour une classe spécifique depuis des fichiers CSV"""
    bruitees = []
    for i in range(n):
        image_bruitee = load_bruitee_from_csv(path, classe, i)
        bruitees.append(image_bruitee)
    return bruitees



def entropie_pdf(pdf_reference, pdf_bruitee):
    return shannon.entropie_croise(pdf_reference, pdf_bruitee)

def CE_img_br(images_bruitees, pdf):
    ce = []
    for classe, pdf_non_bruitee in enumerate(pdf, start=1):
        entropie_croisee_classe = entropie_pdf(pdf_non_bruitee, images_bruitees)
        ce.append(entropie_croisee_classe)
    return ce


# Charge les images depuis les fichiers CSV
def creer_dic_image(n1, n2):
    dic_images = {}
    for i in range(1, 7):
        dic_images[i] = []
        for j in range(n1, n2 + 1):
            dic_images[i].append(load_from_csv('img', i, j))
    return dic_images

#Calcule la fonction de densité de probabilité pour chaque classe
def creer_dic_pdf(data):
    dic_pdf = {}  
    for classe, lpixels in data.items():
        dic_pdf[classe] = calcul_pdf(lpixels)
    return dic_pdf


def creer_dic_image_bruite(n1, n2):
    dic_images = {}
    for i in range(1, 7):
        dic_images[i] = []
        for j in range(n1, n2 + 1):
            dic_images[i].append(load_bruitee_from_csv("images_bruitees", i, j))
    return dic_images


def afficher_matrice(mat, titre):
    dossier = "png"
    os.makedirs(dossier, exist_ok=True)
    nom = titre.replace(' ', '_')
    if not nom.endswith(".png"):
            nom = nom + ".png"
    chemin = os.path.join(dossier, nom)
    plt.imshow(mat, cmap='viridis')
    plt.colorbar()
    classes = ["coins", "diag_g", "diag_d", "centre", "vline", "hline"]
    #classes = ["hline", "vline", "coins", "diag_g", "diag_d", "centre"]
    tclasses = len(classes)
    plt.xticks(np.arange(tclasses), classes)
    plt.yticks(np.arange(tclasses), classes)
    plt.title(titre) 
    plt.savefig(chemin, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()



def save_to_csv_optimise(filename: str, data: "list[tuple[str, np.ndarray]]"):
    """Sauvegarde les données dans un fichier csv.
    
    Chaque ligne du fichier sera au format:
    label, px1, px2, ..., px64
    
    Avec label le nom de la forme
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        
        for label, matrix in data:
            tmp = [label] + ['1' if x else '' for x in matrix.flatten()]
            writer.writerow(tmp)


def load_from_csv_optimise(filename: str) -> "list[tuple[str, np.ndarray]]":
    """Charge les données depuis un fichier csv"""
    res = []
    with open(filename, 'r') as f:
        lines_read = csv.reader(f)

        for line in lines_read:
            label = line.pop(0)
            tmp = np.array([bool(x) for x in line])
            matrix = np.ndarray((8, 8), dtype=bool, buffer=tmp)
            res.append((label, matrix))
    
    return res


def load_from_csv(path: str, classe: int, image: int) -> np.ndarray:
    """Charge une image spécifique depuis un fichier CSV"""
    classe_folder = f"Classe_{classe}"
    image_filename = f"image_{image}.csv"
    image_path = os.path.join(path, classe_folder, image_filename)

    with open(image_path, 'r') as f:
        lines_read = csv.reader(f)
        image_data = []
        for line in lines_read:
            image_data.append([(float(x)) for x in line])

    return np.array(image_data, dtype=np.float64)

"""
Charge n image de chaque classe
"""
def charger_image(n):
    i=1
    while i<=6:
        j=1
        while j<=n:
            afficher_image(load_from_csv('img', i, j))
            j+=1
        i+=1


def charger_csv(n):
    i=1
    L=[]
    while i<=6:
        j=1
        while j<=n:
            L.append(load_from_csv('img', i, j))
            j+=1
        i+=1
    return L


def sauver_rendu(image, nom):
    dossier = "rendu"
    os.makedirs("rendu", exist_ok="True")
    if not nom.endswith(".csv"):
        nom = nom + ".csv"
    chemin = os.path.join(dossier, nom)
    creer_image(image, chemin)



def calcul_pdf_single(image_bruite):
    somme_image = np.sum(image_bruite)  # Somme des valeurs de l'image brute
    pdf_image = image_bruite / somme_image  # Calcul de la densité de probabilité
    return pdf_image


def predire(image_bruite, classes_images, classe_reelle):
    r = []
    pdf_img_br = calcul_pdf_single(image_bruite)
    for classe, images_classe in classes_images.items():
        pdf_classe = calcul_pdf_single(images_classe)
        entropie = shannon.Calculer_Entropie_Croisee(pdf_img_br, pdf_classe)
        r.append(entropie)
    print(r)
    #kit.afficher_matrice(r, "Entropies croisées entre classes notre image bruité et chaque classe")
    prediction = np.argmin(r)+1
    print("Classe prédite :", prediction)
    if prediction == classe_reelle:
        print("La prédiction est correcte!")
    else:
        print("La prédiction est incorrecte.")
    return prediction

