import cv2
import os

# Define the path to the folder containing the images
image_folder = 'images'

# Define the desired size for the resized images
new_width = 224
new_height = 224

# Loop through all the images in the folder
for filename in os.listdir(image_folder):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    # Load the image
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))

    # Save the resized image
    cv2.imwrite(image_path, resized_img)

print("Images resized successfully!")


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import os

# Define the path to the folder containing the images
image_folder = 'images'

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Initialize InceptionResnetV1 for face embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Loop through all the images in the folder
for filename in os.listdir(image_folder):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    # Load the image
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face
    face, _ = mtcnn(img, return_prob=True)

    if face is not None:
      # Generate face embedding
      face = face.unsqueeze(0)
      embedding = resnet(face).detach().numpy()
      print(f"Embedding for {filename}: {embedding}")
    else:
      print(f"No face detected in {filename}")


import faiss
import numpy as np

# Define the path to the folder containing the images
image_folder = 'images'

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Initialize InceptionResnetV1 for face embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize FAISS index
index = faiss.IndexFlatL2(512)  # Assuming embedding dimension is 512

# Loop through all the images in the folder
for filename in os.listdir(image_folder):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    # Load the image
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face
    face, _ = mtcnn(img, return_prob=True)

    if face is not None:
      # Generate face embedding
      face = face.unsqueeze(0)
      embedding = resnet(face).detach().numpy()

      # Add embedding to FAISS index
      index.add(np.float32(embedding))

      print(f"Embedding for {filename} added to FAISS index.")
    else:
      print(f"No face detected in {filename}")


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
from google.colab import files
files.download('app.py')
