import numpy as np
import cv2

SCALE_FACTOR = 1.3
DISPLAY_BOUNDRY_BOX = False

# Read the image and convert to gray
img = cv2.imread('face1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sunglasses = cv2.imread('sunglasses.png') #Feature to be added

# now we can try to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h) = sorted(faces, key=lambda face: face[2]*face[3])[-1] #Might have more than one face -> choose the largest
    
roi_gray = gray[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray) # Might have more than two 
eyes_in_image = sorted(eyes, key=lambda eye: eye[2]*eye[3])[-2:] #Find the two largest detections since they're probably the eye

#Create boundry boxes
if DISPLAY_BOUNDRY_BOX:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_color = img[y:y+h, x:x+w]
	for (ex,ey,ew,eh) in eyes_in_image:
	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

(lx,ly,lw,lh),(rx,ry,rw,rh) = sorted(eyes_in_image, key=lambda eye: eye[0]) #Get left and right eye respectively

#Getting real distances (since original lx,ly,rx,ry were wrt roi)
lx+= x
ly+= y
rx+= x
ry+= y

size_x = int(SCALE_FACTOR* (rx+rw-lx))
size_y = int(SCALE_FACTOR* (lh))

distored_sunglasses = cv2.resize(sunglasses, (size_x, size_y))

pixel_scale_x = int((size_x- (rx+rw-lx)) /2)
pixel_scale_y = int((size_y- (lh)) /2)

sepia_kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
img_sepia = cv2.transform(img,sepia_kernel) #Speia image

for i in range(ly -pixel_scale_y,ly+lh +pixel_scale_y):
	for j in range(lx -pixel_scale_x, rx+rw +pixel_scale_x):
		val_sunglasses = distored_sunglasses[i-(ly-pixel_scale_y), j-(lx-pixel_scale_x)]
		if np.all(val_sunglasses < 250):
			img_sepia[i,j] = val_sunglasses


cv2.imshow('img',img_sepia)
cv2.waitKey(0)
cv2.destroyAllWindows()