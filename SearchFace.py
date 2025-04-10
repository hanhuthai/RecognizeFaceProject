import cv2
import numpy as np
import onnxruntime as ort
import faiss
from numpy.linalg import norm
import mysql.connector
from insightface.app import FaceAnalysis  # Add this import

def cosine_similarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def l2_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)


def load_model(model_path):
    return cv2.dnn.readNetFromONNX(model_path)
#    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def crop_face(image):
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    face = faces[0]

    bbox = face.bbox.astype(int)
    cropped_face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Save the cropped face
    cv2.imwrite("face.jpg", cropped_face)
    print(f"Face saved to face.jpg")
    return cropped_face, face.kps  # Return keypoints for alignment

def extract_embedding(net, img):
    try:
        if img is None:
            raise ValueError("Could not read image")

        # Handle 112x112 images
        if img.shape[0] == 112 and img.shape[1] == 112:
            cropped = img.copy()
        else:
            src = img.copy()

            # Pad if height is less than 128
            if src.shape[0] < 128:
                top = (128 - src.shape[0]) // 2
                bottom = 128 - src.shape[0] - top
                src = cv2.copyMakeBorder(src, top, bottom, 0, 0,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))

            # Pad if width is less than 128
            if src.shape[1] < 128:
                left = (128 - src.shape[1]) // 2
                right = 128 - src.shape[1] - left
                src = cv2.copyMakeBorder(src, 0, 0, left, right,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))

            # Resize to 128x128
            resized = cv2.resize(src, (128, 128))

            #print("Resized image:", resized[:, :, 2])

            # Crop center 112x112
            a = (128 - 112) // 2
            b = (128 - 112) // 2 + 112
            cropped = resized[a:b, a:b]

        # Flip image horizontally
        flipped = cv2.flip(cropped, 1)
        # Create blobs
        cropped_blob = cv2.dnn.blobFromImage(cropped, scalefactor=1.0 / 255, size=(112, 112), mean=(0, 0, 0),
                                             swapRB=False,
                                             crop=False)
        flipped_blob = cv2.dnn.blobFromImage(flipped, scalefactor=1.0 / 255, size=(112, 112), mean=(0, 0, 0),
                                             swapRB=False,
                                             crop=False)
        # Forward pass
        net.setInput(cropped_blob)
        res1 = net.forward()

        net.setInput(flipped_blob)
        res2 = net.forward()

        # Return sum of embeddings
        return res1 + res2

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return np.zeros((1, 512), dtype=np.float32)

def get_embeddings_from_db(cursor):
    cursor.execute("SELECT faceObjId, embedding, frameId FROM face_object")
    rows = cursor.fetchall()
    faceObjIDS = []
    embeddings = []
    frameIds = []
    for row in rows:
        faceObjId, embedding_blob, frameId = row
        #print(f"Frame ID: {frameId}, Embedding Blob Size: {len(embedding_blob)}")  # Debugging line

        # Decode the PNG image back to a NumPy array
        embedding = np.frombuffer(embedding_blob, np.float32)

        faceObjIDS.append(faceObjId)
        embeddings.append(embedding)
        frameIds.append(frameId)

    return faceObjIDS, np.array(embeddings), frameIds

def find_similar_faces_cosine(input_embedding, embeddings, top_n=5):
    # Normalize the embeddings to unit vectors
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    #print("Embedding:", embeddings)
    input_embedding = input_embedding / np.linalg.norm(input_embedding)

    # Compute cosine similarity
    similarities = np.dot(embeddings, input_embedding)

    # Get the top N similar faces
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_similarities = similarities[top_indices]

    return top_indices, top_similarities

def get_image_paths_from_db(cursor, frameIds):
    format_strings = ','.join(['%s'] * len(frameIds))
    cursor.execute(f"SELECT frameId, imgPath FROM frame_metadata WHERE frameId IN ({format_strings})", tuple(frameIds))
    rows = cursor.fetchall()
    frameId_to_imgPath = {row[0]: row[1] for row in rows}
    return frameId_to_imgPath

def search_similar_faces(image_path, k=3):
    cursor = conn.cursor()
    model_path = "ArcFace.onnx"

    # Load model
    net = load_model(model_path)

    # Read image
    img = cv2.imread(image_path)

    cropped_face, keypoints = crop_face(img)

    input_embedding = extract_embedding(net, cropped_face)
    input_embedding = input_embedding.flatten()

    # Get embeddings from database
    ids, embeddings, frameIds = get_embeddings_from_db(cursor)

    # Find similar faces using cosine similarity
    similar_indices, similarities = find_similar_faces_cosine(input_embedding, embeddings)

    # Close cursor
    cursor.close()

    # Convert numpy.float32 to float
    similar_faces = [(ids[idx], float(sim)) for idx, sim in zip(similar_indices[:k], similarities[:k])]

    # Return top k similar faces
    return similar_faces

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="portal",
    port=3307
)

if __name__ == "__main__":
    cursor = conn.cursor()
    model_path = "ArcFace.onnx"
    image_path = "chinh-front.jpg"  # Path to cropped image
    #image_path= r"D:\ai-portal\frontend\static\media\eTheiaStorage\img\2 (5).jpg"  #thai
    image_path =r"D:\ai-portal\frontend\static\media\eTheiaStorage\img\thaifrontcokinh.jpg"

    # Load model
    net = load_model(model_path)

    # Read image
    img = cv2.imread(image_path)

    #crop image if full face
    cropped_face, keypoints = crop_face(img)

    input_embedding = extract_embedding(net, cropped_face)
    input_embedding = input_embedding.flatten()
    print("Embedding-originl:", input_embedding)

    #from db
    # Get embeddings from the database
    ids, embeddings, frameIds = get_embeddings_from_db(cursor)

    # Find the most similar images using Faiss
    similar_indices, distances = find_similar_faces_cosine(input_embedding, embeddings)

    # Print results, the smaller the distance, the more similar
    for idx, dist in zip(similar_indices, distances):
        print(f"ID: {ids[idx]}, Distance: {dist:.4f}")

    # Close connection
    cursor.close()
    conn.close()
