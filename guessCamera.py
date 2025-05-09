import tensorflow as tf
import keras
import numpy as np
import cv2

def guess(image_arr, model):
    image_resized = np.resize(image_arr, (100, 100))
    image_flat = image_resized.flatten()
    image_flat = image_flat / 255.0
    image_input = np.expand_dims(image_flat, axis=0)
    prediction_logits = model.predict(image_input, verbose=0)
    probability = tf.sigmoid(prediction_logits)[0][0].numpy()
    return probability

def open_camera(model):
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_img, (100, 100))
            captured_image_array = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            probability = guess(captured_image_array, model)
            isFemale = False if probability >= 0.5 else True
            rectangleColor = (147, 20, 255) if isFemale else (255, 0, 0)
            confidence = (1 - probability) if isFemale else probability
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangleColor, 3, cv2.LINE_4)
            cv2.putText(frame, f'{"female" if isFemale else "male"} - {confidence:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, rectangleColor, 3)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = keras.models.load_model("biometric_attribute_classification.keras")
    open_camera(model)
