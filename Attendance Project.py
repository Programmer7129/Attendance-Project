import face_recognition as fr
import cv2
import numpy as np
import os
from datetime import datetime as dt

path = 'ImagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        nameList = []
        for line in mydatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = dt.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)  Use this command if you want to get names of the image result in output terminal.
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4 , x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FORMATTER_FMT_NUMPY, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)  # Make sure to attach a webcam to your device otherwise this won't work.
    cv2.waitKey(1)


# Great Work. Now you may ask how to add more images to recognize more people
# Well, it's very easy, you just need to add respective images of the person
# in Images Attendance Folder to recognize them.
