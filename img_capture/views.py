from django.shortcuts import render
import cv2
import numpy as np

def img_capture(request):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    skip = 0
    face_data = []
    dataset_path = "./face_dataset/"

    file_name = request.POST.get("person_name", "default")

    while True:
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret:
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces[0]
                offset = 5
                face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                face_selection = cv2.resize(face_offset, (100, 100))

                if skip % 10 == 0:
                    face_data.append(face_selection)
                    print(len(face_data))

                cv2.imshow("face", face_selection)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("faces", frame)
            skip += 1

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q') or len(face_data) >= 50:
            break

    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print(face_data.shape)

    np.save(dataset_path + file_name, face_data)
    print("Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'img_capture.html')  # Create this HTML template if needed
