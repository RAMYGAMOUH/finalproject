import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Charger le modèle d'embedding
model = tf.keras.models.load_model('mon_modele_facenet.h5')  

# ... code précédent pour créer l'index FAISS ...

# Sauvegarder l'index FAISS
faiss.write_index(index, "index.faiss") 

# ... suite du code ...
# Charger l'index FAISS
index_path = 'index.faiss'
index = faiss.read_index(index_path)

def get_embedding(image):
    # Prétraiter l'image pour le modèle FaceNet
    image = np.array(image)
    image = tf.image.resize(image, (160, 160))  # Adapter à la taille requise par FaceNet
    image = (image / 255.0).astype(np.float32)  # Normaliser l'image
    embedding = model(image[np.newaxis, ...])  # Ajouter une dimension batch
    return embedding.numpy().flatten()

st.title('Reconnaissance Faciale')

uploaded_image = st.file_uploader("Choisissez une image...", type="png")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    embedding = get_embedding(image)

    # Recherche de visages similaires
    distances, indices = index.search(np.array([embedding]), k=1)  # k=1 pour le visage le plus proche
    
    st.write(f"Visage le plus similaire : Index {indices[0][0]}")
    st.write(f"Distance : {distances[0][0]}")
