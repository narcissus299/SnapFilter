import numpy as np
import cv2

# Read the image and convert to gray
img = cv2.imread('face5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sunglasses = cv2.imread('sunglasses.png')

# now we can try to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h) = sorted(faces, key=lambda face: face[2]*face[3])[-1] #Might have more than one face -> choose the largest
    
roi_gray = gray[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray) # Might have more than two 
eyes_in_image = sorted(eyes, key=lambda eye: eye[2]*eye[3])[-2:] #Find the two largest detections since they're probably the eye

#Create boundry boxes
# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# roi_color = img[y:y+h, x:x+w]
# for (ex,ey,ew,eh) in eyes_in_image:
#     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

(lx,ly,lw,lh),(rx,ry,rw,rh) = sorted(eyes_in_image, key=lambda eye: eye[0]) #Get left and right eye respectively

#Getting real distances (since original lx,ly,rx,ry were wrt roi)
lx+= x
ly+= y
rx+= x
ry+= y

distored_sunglasses = cv2.resize(sunglasses, (rx+rw - lx, lh))

sepia_kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
img = cv2.transform(img,sepia_kernel)

for i in range(ly,ly+lh):
	for j in range(lx, rx+rw):
		val_sunglasses = distored_sunglasses[i-ly, j-lx]
		if np.all(val_sunglasses != 255):
			img[i,j] = val_sunglasses


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()