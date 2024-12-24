import ssl
import cv2
import numpy as np
import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context

model = tf.keras.applications.MobileNetV2(weights='models/MobileNet V2 Weights 1.0 224.h5')

cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def predict(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)
    return decoded_predictions[0][0][1]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    object_name = predict(frame)
    cv2.putText(frame, object_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()