import cv2
import numpy as np
import onnxruntime as ort

# Global variable to store the ONNX model
onnx_model = None

def init_model(model_path):
    global onnx_model
    if onnx_model is None:
        onnx_model = cv2.dnn.readNetFromONNX(model_path)

def extract_feature(img):
    try:
        global onnx_model
        if onnx_model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")

        # Read image
        img = cv2.imread(img)
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
        onnx_model.setInput(cropped_blob)
        res1 = onnx_model.forward()

        onnx_model.setInput(flipped_blob)
        res2 = onnx_model.forward()

        # Return sum of embeddings
        return res1 + res2

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return np.zeros((1, 512), dtype=np.float32)
