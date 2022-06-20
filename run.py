import datetime
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import numpy as np
import telegram


currentname = "unknown"

encodingsP = "encodings.pickle"

# Variables
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cont = 0
salida = cv2.VideoWriter('videoSalida.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
actual = 0


print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
band = True

vs = VideoStream(src=0, framerate=20).start()

time.sleep(2.0)

fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640, height=480)
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: no se ha detectado movimiento"

    area_pts = np.array([[130, 160], [245, 165], [245, 480], [120, 480]])

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, 255, -1)
    imAux = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = fgbg.apply(imAux)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.erode(fgmask, None, iterations=3)

    conts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"  # if face is not recognized, then print Unknown

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(currentname)

        names.append(name)

        cv2.drawContours(frame, [area_pts], -1, color, 2)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)
        if name == 'Unknown':
            for cnt in conts:
                if cv2.contourArea(cnt) > 400:
                    # Dibuja en rectangulos los movimientos detectados
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    texto_estado = "Movimiento detectado"
                    color = (0, 0, 255)
                    cont += 1
    if cont > 20:
        # Una vez con la condicion empieza a grabar
        today = datetime.datetime.today()
        if band:
            minuto = today.minute
            band = False
        salida.write(frame)

        if today.minute > minuto+1:
            # Manda los mensajes a telegram
            telegram.sendMessage()
            salida.release()
            telegram.sendVideo()
            print("Video enviado")
            cont = 0
            band = True

    # display the image to our screen
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Facial Recognition is Running", frame)
    cv2.imshow('imAux', fgmask)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
