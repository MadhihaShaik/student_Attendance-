import cv2
import numpy as np
import face_recognition

imgMadhiha = face_recognition.load_image_file('Images_Attendance/madhiha.jpg')
imgMadhiha = cv2.cvtColor(imgMadhiha, cv2.COLOR_BGR2RGB)
imgshaik = face_recognition.load_image_file('Images_Attendance/shaik.jpg')
imgshaik = cv2.cvtColor(imgshaik, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgMadhiha)[0]
encodeModi = face_recognition.face_encodings(imgMadhiha)[0]
cv2.rectangle(imgMadhiha, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

facelocshaikimgshaik = face_recognition.face_locations(imgshaik)[0]
encodeshaikimgshaik = face_recognition.face_encodings(imgshaik)[0]
cv2.rectangle(imgshaik, (facelocshaikimgshaik[3], facelocshaikimgshaik[0]), (facelocshaikimgshaik[1], facelocshaikimgshaik[2]), (155, 0, 255), 2)

results = face_recognition.compare_faces([encodeModi], encodeshaikimgshaik)
faceDis = face_recognition.face_distance([encodeModi], encodeshaikimgshaik)
print(results, faceDis)
cv2.putText(imgshaik, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

cv2.imshow('madhiha', imgMadhiha)
cv2.imshow('shaik', imgshaik)
cv2.waitKey(0)