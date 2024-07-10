from django.shortcuts import render
import numpy as np
import cv2
import os

# Function to calculate distance between two vectors
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

# Function to implement KNN algorithm
def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Load dataset
dataset_path = "./face_dataset/"
face_data = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)

# Flatten each face image in face_dataset
face_dataset = np.array([face.flatten() for face in face_dataset])

face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

# Function to handle webcam feed and face recognition
def img_recognition(request):
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        if ret == False:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for face in faces:
            x, y, w, h = face
            offset = 5
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section, (100, 100))
            face_section_flatten = face_section.flatten().reshape(1, -1)
            out = knn(trainset, face_section_flatten)
            cv2.putText(frame, names[int(out)], (x, y-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.imshow("Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'img_recognition.html')  # You need to create this HTML template
