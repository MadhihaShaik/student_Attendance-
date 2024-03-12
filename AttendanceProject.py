import mysql.connector
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import os
import csv

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="attendance"
)
cursor = conn.cursor()

def load_student_data():
    cursor.execute('SELECT id, name, department FROM students_info')
    return cursor.fetchall()

students_info = load_student_data()

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(student_id, name, department):
    dString = datetime.now().strftime('%d/%m/%Y')
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)  # Move the file pointer to the beginning
        if not f.read(1):  # Check if the file is empty
            f.write('id,name,department,time,date\n')  # Write header if file is empty
        f.seek(0)  # Move the file pointer to the beginning
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header line
            entry = line.strip().split(',')
            if entry[1] == name and entry[4] == dString:  # Check if an entry for the student already exists for today
                print("Attendance already marked for today.")
                return  # Exit the function without adding a new entry
        time_now = datetime.now().strftime('%H:%M:%S')
        f.write(f'{student_id},{name},{department},{time_now},{dString}\n')
        print("Attendance marked successfully.")

# Load images from the Images_Attendance folder
path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)
print('Encoding Complete')
print('encodeListKnown length:', len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture a frame from the webcam.")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    if not facesCurFrame:
        print("No faces detected in the current frame.")
        continue

    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            student_info = students_info[matchIndex]
            name = student_info[1].upper()
            student_id = student_info[0]
            department = student_info[2]

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, f"Name: {name}", (x1+6, y2-30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"ID: {student_id}", (x1+6, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Department: {department}", (x1+6, y2+10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

            markAttendance(student_id, name, department)

    cv2.imshow('webcam', img)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()
