# San_0AI

Une IA personnelle simple pour serveur Proxmox

## Installation

1. Cloner le repository
2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

## Utilisation

1. Démarrer le serveur:
```bash
python san_ai.py
```

2. Envoyer des requêtes à l'API:
```bash
curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"input":"bonjour"}'
```

## Configuration sur Proxmox

1. Créer un conteneur LXC sous Proxmox
2. Installer Python et pip
3. Cloner ce repository
4. Installer les dépendances
5. Configurer le service pour démarrer automatiquement
