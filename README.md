# Mask-Detector
Machine Learning model to detect if a person is wearing a mask on the face

<h1 align="center">Face mask detector</h1> 

<div align= "center">
  <h4>Depuis la crise sanitaire, de nombreuses mesures ont été prises pour lutter contre le virus notamment le port du masque dans les espaces publiques. <br> 
    Ce projet avait pour but de tenter de reproduire les technologies récemment mises en place pour contrôler le port du masque
</h4>
</div>

## Sommaire

  - [Dataset](#Dataset)
  - [Video detector Set up](#VideoSetUp)
  - [Image detector Set up](#ImageSetUp)


## Dataset

Le dossier "dataset" contient les deux dossiers d'images utilisés pour le projet :
 - "Mask" : images avec des visages portant un masque
 - "NoMask" : images avec des visages sans masques.

*N.B : Les deux datasets ont été créés automatiquement via le fichier BingWebScrapping.ipynb. Bien qu'un filtre mannuel ait été fait il peut comporter des irrégularités.*

## Video Detector SetUp

Ce projet a été codé en python, et utilise différentes librairies. 

Pour faire fonctionner la détéction de masque via la webcam l'installation suivante est nécessaire :

- Créer un environnement virtuel python
- Télécharger requirements.txt
- Installer les versions des librairies nécessaires présentes dans requirements.txt
- Lancer video_live_detection.py
  

# Installation des librairies


