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
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def create_transformation_matrix(src_points, dst_points):
    if len(src_points) != 5 or len(dst_points) != 5:
        raise ValueError("Both src_points and dst_points must contain exactly 5 points")
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    return cv2.estimateAffinePartial2D(src_points, dst_points)[0]

def preprocess_image(img, transfer_mat=None):
    if img is None or not isinstance(img, np.ndarray) or len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Invalid input image or not RGB")
    if img.dtype != np.uint8 or img.min() < 0 or img.max() > 255:
        raise ValueError("Image must be uint8 with values from 0-255")

    face_size = (112, 112)
    target_size = (128, 128)

    # Align image if transfer_mat is provided
    if transfer_mat is not None:
        if transfer_mat.shape != (2, 3):
            raise ValueError("transfer_mat must be a 2x3 matrix")
        face_aligned = cv2.warpAffine(img, transfer_mat, face_size)
        cv2.imwrite("face_aligned.jpg", face_aligned)
        print(f"Face saved to face_aligned.jpg")
    else:
        face_aligned = img

    # Resize image
    if face_aligned.shape[:2] != face_size:
        src = face_aligned
        if src.shape[:2] != target_size:
            scale = min(target_size[0] / src.shape[0], target_size[1] / src.shape[1])
            new_w, new_h = int(src.shape[1] * scale), int(src.shape[0] * scale)
            src = cv2.resize(src, (new_w, new_h))
            top = (target_size[0] - new_h) // 2
            bottom = target_size[0] - new_h - top
            left = (target_size[1] - new_w) // 2
            right = target_size[1] - new_w - left
            src = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if src.shape[:2] != target_size:
                src = cv2.resize(src, target_size)
        a, b = (target_size[0] - face_size[0]) // 2, (target_size[0] - face_size[0]) // 2 + face_size[0]
        cropped = src[a:b, a:b]
    else:
        cropped = face_aligned

    # Convert color and flip
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    flipped = cv2.flip(cropped, 1)

    # Create blob
    cropped_blob = cv2.dnn.blobFromImage(cropped, 1.0 / 255, face_size, (0, 0, 0), swapRB=False, crop=False)
    flipped_blob = cv2.dnn.blobFromImage(flipped, 1.0 / 255, face_size, (0, 0, 0), swapRB=False, crop=False)

    return cropped_blob, flipped_blob

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

def extract_embedding(net, img, transfer_mat=None):
    if not hasattr(net, 'get_inputs') or not hasattr(net, 'run'):
        raise ValueError("Invalid model")

    try:
        cropped_face, keypoints = crop_face(img)  # Crop the face before preprocessing
        if transfer_mat is None:
            # Define destination points for alignment
            dst_points = [
                [30.2946 + 8.0, 51.6963],
                [65.5318 + 8.0, 51.5014],
                [48.0252 + 8.0, 71.7366],
                [33.5493 + 8.0, 92.3655],
                [62.7299 + 8.0, 92.2041]
            ]
            transfer_mat = create_transformation_matrix(keypoints, dst_points)
        transfer_mat= None
        cropped_blob, flipped_blob = preprocess_image(cropped_face, transfer_mat)
        input_name = net.get_inputs()[0].name

        # Run inference
        res1 = net.run(None, {input_name: cropped_blob})[0]
        res2 = net.run(None, {input_name: flipped_blob})[0]

        # Aggregate embeddings (not normalized to match C++)
        return res1 + res2
    except Exception as e:
        print("Error extracting embedding:", e)
        return np.zeros(512, dtype=np.float32)

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

    input_embedding = extract_embedding(net, img)
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
    image_path =r"D:\ai-portal\frontend\static\media\eTheiaStorage\database\ClusterA\3\20230209-075224-837855.png"

    # Load model
    net = load_model(model_path)

    # Read image
    img = cv2.imread(image_path)

    input_embedding = extract_embedding(net, img)
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
