import cv2 as cv #for processing of images
import os #for joining paths 
import face_recognition as fr #for encoding,compare,location
import numpy as np #array processing package

folder = ['Data']
dir = r'F:\Chandni'

def names():
    people = []
    for f in folder:
        folder_path = os.path.join(dir,f)

        for img in os.listdir(folder_path): #accessing all the photos from Data
            people.append(str(img[:len(img)-4]))
    return people

def find_encodings():
    encoding_list = []
    for f in folder:
        imge = os.path.join(dir,f)


        for img in os.listdir(imge):
            img_path = os.path.join(imge,img)
            # print(img_path)
            img = cv.imread(img_path)
            encoding_list.append(fr.face_encodings(img)[0])
            # cv.imshow('test',img)
            # cv.waitKey(0)
    return encoding_list


encoding_list = find_encodings()
people = names()
# print(people)

#for local image----------->
test = cv.imread('Modi.png')
test_encoding = fr.face_encodings(test)[0]
flag = 0

size = len(encoding_list)
for i in range (0,size):
    result = fr.compare_faces([encoding_list[i]],test_encoding)[0]
    if(result == True):
        flag = 1
        test = cv.resize(test,(500,500),interpolation=cv.INTER_CUBIC)
        test_face_loc = fr.face_locations(test)[0]
        cv.rectangle(test,(test_face_loc[3],test_face_loc[0]),(test_face_loc[1],test_face_loc[2]),(0,255,0),2)
        cv.putText(test,str(people[i]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
        cv.imshow('test',test)
        cv.waitKey(0)
        break

if(flag == 0):
    test = cv.resize(test,(500,500),interpolation=cv.INTER_CUBIC)
    test_face_loc = fr.face_locations(test)[0]
    cv.rectangle(test,(test_face_loc[3],test_face_loc[0]),(test_face_loc[1],test_face_loc[2]),(0,0,255),2)
    cv.putText(test,str("unknown"),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)
    cv.imshow('test',test)
    cv.waitKey(0)


# from camera -------->
# img = cv.VideoCapture(0)
# while True:
#     isTrue , frame = img.read()
#     #cv.imshow('img',frame)
#     frame_enc = fr.face_encodings(frame)
#     for enc in frame_enc:
#         result = fr.compare_faces(encoding_list,enc)
#         dist = fr.face_distance(encoding_list,enc)
#         mindis = np.argmin(dist) #index of min index
#         if(dist[mindis] > 0.5):
#             face_loc = fr.face_locations(frame)
#             for loc in face_loc:
#                 cv.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),(0,0,255),2)
#                 cv.putText(frame,str("unknown"),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)
#                 cv.imshow('frame',frame)
#         else:
#             face_loc = fr.face_locations(frame)
#             for loc in face_loc:
#                 cv.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),(0,255,0),2)
#                 cv.putText(frame,str(people[mindis]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
#                 cv.imshow('frame',frame)
#     if(cv.waitKey(20) & 0xFF == ord('q')):
#         break
