#### Description
L'Image Captionning est un vieux problème complexe ayant pour but de décrire une image par une courte phrase. Depuis l'émergence des réseaux de neurone profond permet aujourd'hui à un niveau d'abstraction suffisant pour aujourd'hui dire que le problème est résolue. Le but est d'essayer de résoudre par moi-même ce projet. Il se compose d'un encoder qui est un modèle inception v3 pré-entreiné et d'un décoder qui est un modèle de type LSTM que l'on entreinera.

#### Installation
##### Telecharger le projet
soit vous passez par ce [lien](https://github.com/piirios/Image-captionning) pour télécharger le projet
soit vous effectuer dans une console
```shell
git clone https://github.com/piirios/Image-captionning.git
```
##### Installer les dépendances
dans le dossier du projet effectuer
```shell
pip install -r requirement.txt
```
##### Dataset
telecharger les dossiers train2014, val2014 et train/val annotation 2014 sur ce [lien](https://cocodataset.org/#download)

ensuite deziper-les et dans le répertoire du projet effectuer:
```shell
python cli.py set_dataset_folder FOLDER
```
où FOLDER est le chemin du dossier racine du dataset cad. le chemin du  dossier parent de des dossiers dézipé. 
##### Dossier des poids
dans le répertoire du projet effectuer
```shell
python cli.py set_dataset_folder FOLDER
```
##### ressource de la langue
effectuez:
```shell
python -m spacy download en_core_web_sm
```
##### Entreinement
Pour entreinez un traducteur effectuer
```shell
python cli.py train
```
