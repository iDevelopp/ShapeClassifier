import numpy as np
import math
import kit
import os 

epsilon = 0.0000000000000001

"""
Code propose par Mathis Sedkaoui

get_rdm_hline:
    Genere aleatoirement des lignes horizontales dans une matrice
    de taille 8x8, en fonction d'un certain seuil. Si la valeur
    aleatoire generee est superieure au seuil, la cellule est
    remplie par 1, sinon par 0. Ainsi, on creer des motifs
    de lignes aleatoires pour une image.

get_rdm_vline:
    Genere aleatoirement des lignes verticales en transposant
    la matrice generee par get_rdm_hline.
"""

def gen_random_vline(h: int=8, w: int=8, treshold: float=0.35) -> np.ndarray:
    """Genere une image contenant une ligne verticale.

    `treshold` est le seuil qui determine si un pixel est allume ou non"""
    # Bool car 0 ou 1 (pixel eteint ou allume)
    res = np.zeros((h, w), dtype=bool)
    
    # On prend les pixels du milieu (dec est la pour gerer le cas ou la largeur
    # est impaire)
    dec = 1 if not w & 1 else 0
    for col in range(w // 2 - dec, w // 2 + 1):
        for row in range(h):
            res[row, col] = (np.random.rand() > treshold)
    res = res.astype(float) + epsilon
    return res


def gen_random_hline(h: int=8, w: int=8, treshold: float=0.35) -> np.ndarray:
    """Genere une image contenant une ligne horizontale.
    `treshold` est le seuil qui determine si un pixel est allume ou non"""
    # L'attribut `T` fournit la transposee.
    # On inverse h et w pour le cas ou h != w
    return gen_random_vline(h=w, w=h, treshold=treshold).T


"""
fin code propose en cours.
On continue sur cette ligne directrice
"""

def gen_coins(h: int=8, w: int=8, seuil: float=0.35) -> np.ndarray:
    #Génère une image des coins aléatoires 
    r = np.zeros((h, w), dtype=bool)
    r[0,0] = np.random.rand() > seuil     #coin sup gauche
    r[0,w-1] = np.random.rand() > seuil   #coin sup droit
    r[h-1,0] = np.random.rand() > seuil   #coin inf gauche
    r[h-1,w-1] = np.random.rand() > seuil #coin inf droit
    return r.astype(float) + epsilon

def gen_diag_gauche(h: int=8, w: int=8, seuil: float=0.35) -> np.ndarray:
    #Génère une image contenant une diagonale de DROITE vers la GAUCHE 
    r = np.zeros((h, w), dtype=bool)
    i=0
    while i< min(h,w):
        r[i, w-1-i] = np.random.rand() > seuil
        i+=1
    return r.astype(float) + epsilon
        
def gen_diag_droite(h: int=8, w: int=8, seuil: float=0.35) -> np.ndarray:
    #Génère une image contenant une diagonale de GAUCHE vers la DROITE 
    r = np.zeros((h, w), dtype=bool)
    i=0
    while i< min(h,w):
        r[i, i] = np.random.rand() > seuil
        i+=1
    return r.astype(float) + epsilon

def gen_centre(h: int=8, w: int=8, seuil: float=0.35) -> np.ndarray:
    r = np.zeros((h, w), dtype=bool)
    centre = (h//2, w//2) #calcul le centre de la matrice
    r[centre[0], centre[1]] = np.random.rand() > seuil
    r[centre[0]-1, centre[1]] = np.random.rand() > seuil
    r[centre[0], centre[1]-1] = np.random.rand() > seuil
    r[centre[0]-1, centre[1]-1] = np.random.rand() > seuil
    return r.astype(float) + epsilon

def Get_Matrice_Centre(h: int=8, w: int=8, seuil: float=0.35) -> np.ndarray:
    r = np.zeros((h, w), dtype=bool)
    limite = min(h,w)-2
    i = 2
    while i < limite:
        j = 2
        while j < limite:
            r[i,j] = np.random.rand() > seuil
            j+=1
        i+=1
    return r.astype(float) + epsilon
 

#Bruité n matrices de la classe i
#def bruiter_mat(n, i):

def decoupage(img):
    """Découpe l'image en 16 zones de 2x2 pixels.
    
    Le format de retour sera une liste de 16 tuples au format (p1, p2, p3, p4)
    arrangés de cette manière dans la zone:
                p1  p2
                p3  p4
    """
    res = []
    for i in range(4):
        row = 2 * i
        for j in range(4):
            col = 2 * j
            res.append((img[row, col], img[row,col+1], img[row+1, col],
                        img[row+1, col+1]))

    return res

"""
Pour évaluer la robustesse du modèle avec le bruitage, 
nous évaluerons la prédiction par rapport au buit noté :
 xdB = 10*math.log10(img.size/pixels_bruite)
"""




#applique un bruitage selon le taux spécifié (10% par défaut)
def bruitage(image, taux_bruitage=0.10):
    
    taille = image.size #nbr total de pixels dans l'image
    n = int(64 / (10 ** (taux_bruitage/10))) 
    #n=taux_bruitage
    #print(n)

    # A: Shuffle (1:64)
    indices = np.arange(taille)
    np.random.shuffle(indices)

    # B: Prenez les n premiers (A)
    indices_bruit = indices[:n]

    
    img_bruitee = np.copy(image)
    #on calcul les coordonnées de la matrice par ligne-colone pour un indice i car la matrice est flatten ou aplati
    nbr_col = image.shape[1]
    for idx in indices_bruit:
        ligne = idx // nbr_col
        col = idx % nbr_col
        img_bruitee[ligne, col] = 1 - img_bruitee[ligne, col] # T(B) = 1 - T(B)
    
    return img_bruitee

def creer_image_bruitee(images):
    res = []
    for classe, img in images.items:
        img_bruitee = randmat.bruitage(img)
        res.append((classe, img_bruitee))
    nom_img_bruite = "matrice_images_bruite.csv"
    save_to_csv_optimise(nom_img_bruite, res)

def bruitage_et_enregistrer(images):
    os.makedirs("images_bruitees", exist_ok=True) 
    for identifiant, matrices_images in images.items():
        for index, matrice_image in enumerate(matrices_images):
            matrice_image = np.array(matrice_image)
            img_bruitee = bruitage(matrice_image, 0.10)
            dossier_classe = os.path.join("images_bruitees", f"Classe_{identifiant}")
            os.makedirs(dossier_classe, exist_ok=True)
            nom_fichier_bruite = os.path.join(dossier_classe, f"{identifiant}_{index}_bruite.csv")
            kit.creer_image(img_bruitee, nom_fichier_bruite)
