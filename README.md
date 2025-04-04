# Chat avec plusieurs PDFs

Ce projet permet aux utilisateurs de télécharger plusieurs documents PDF et d'engager une conversation avec le contenu de ces documents. 
L'application utilise diverses bibliothèques et outils pour extraire du texte des PDF, diviser le texte en morceaux gérables, 
créer des embeddings vectoriels et configurer une chaîne de récupération conversationnelle avec mémoire.

## Fonctionnalités

- Télécharger et traiter plusieurs documents PDF.
- Extraire du texte des PDF et le diviser en morceaux.
- Créer des embeddings vectoriels en utilisant les modèles Hugging Face.
- Configurer une chaîne de récupération conversationnelle avec mémoire.
- Engager une conversation avec le contenu des PDF téléchargés.


## Prérequis

- Python 3.7 ou supérieur
- Streamlit
- PyPDF2
- LangChain
- Hugging Face Transformers
- FAISS
- dotenv

  ### Modifications clés :
- **Création de la clé API Groq** : J'ai ajouté une note importante concernant la création d'un compte sur Groq pour obtenir une clé API,
- nécessaire pour utiliser le modèle Llama via l'API.
  
Ainsi, vos utilisateurs sauront qu'ils doivent s'inscrire sur Groq et ajouter leur clé API dans le fichier `.env` pour utiliser le modèle Llama.

## Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/sidi-mohamed-ahmed-jiddou/chat-with-pdf.git
cd chat-with-pdf

