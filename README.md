# ERATER

Ce dépôt contient le prototype de l'application "ERATER", un système de vision par ordinateur conçu pour la détection d'objets, le suivi des personnes et l'estimation de proximité en temps réel.

## 1. Organisation du projet

Le projet est structuré de la manière suivante :

- **`main.py`** : Point d'entrée principal de l'application (acquisition vidéo, inférence YOLOv8 et boucle d'affichage).
- **`config.json`** : Fichier de configuration contenant les paramètres modulables (seuils de détection, seuils de proximité, paramètres caméra).
- **`requirements.txt`** : Liste des dépendances et bibliothèques nécessaires au fonctionnement du projet.
- **`benchmark_metrics.csv`** : Fichier de journalisation enregistrant la télémétrie et les événements détectés.
- **`tests/`** : Dossier contenant les scénarios de tests unitaires et d'intégration (`test_pipeline.py`).

## 2. Installation et Exécution

### Prérequis
- Python 3.10 ou supérieur
- Une webcam fonctionnelle ou un flux vidéo compatible

### Procédure d'installation

1. **Cloner le dépôt GitHub :**
   ```bash
   git clone [https://github.com/olivertokam-ops/ERATER.git](https://github.com/olivertokam-ops/ERATER.git)
   cd ERATER

2. **
