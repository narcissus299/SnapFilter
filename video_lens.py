import numpy as np
import cv2

SCALE_FACTOR_GLASS = 1.5
SCALE_FACTOR_JNT = 1.5
SCALE_FACTOR_HAT = 1.5
DISPLAY_BOUNDRY_BOX = True

sunglasses = cv2.imread('thug.jpg') 
jnt = cv2.imread('jnt.jpg')

# now we can try to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, img = cap.read()

	img = cv2.medianBlur(img,3)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if len(faces) > 0:

		filter_applied = False

		#(x,y,w,h) = sorted(faces, key=lambda face: face[2]*face[3])[-1] #Might have more than one face -> choose the largest
		for (x,y,w,h) in faces:
			
			if DISPLAY_BOUNDRY_BOX:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			    
			roi_gray = gray[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray) # Might have more than two 
			
			
			if len(eyes) >= 2:
				eyes_in_image = sorted(eyes, key=lambda eye: eye[2]*eye[3])[-2:] #Find the two largest detections since they're probably the eye

				#Create boundry boxes
				if DISPLAY_BOUNDRY_BOX:
					roi_color = img[y:y+h, x:x+w]
					for (ex,ey,ew,eh) in eyes_in_image:
					    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

				(lx,ly,lw,lh),(rx,ry,rw,rh) = sorted(eyes_in_image, key=lambda eye: eye[0]) #Get left and right eye respectively

				roi_gray_mouth = roi_gray[ly+10:, 0:] #TODO

				#Getting real distances (since original lx,ly,rx,ry were wrt roi)
				lx+= x
				ly+= y
				rx+= x
				ry+= y

				size_x = int(SCALE_FACTOR_GLASS* (rx+rw-lx))
				size_y = int(SCALE_FACTOR_GLASS* (lh))

				distored_sunglasses = cv2.resize(sunglasses, (size_x, size_y))

				pixel_scale_x = int((size_x- (rx+rw-lx)) /2)
				pixel_scale_y = int((size_y- (lh)) /2)

				if not filter_applied:
					#Sepia filter
					sepia_kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
					filtered_img = cv2.transform(img,sepia_kernel)
					
					filter_applied = True

				for i in range(ly -pixel_scale_y,ly+lh +pixel_scale_y):
					for j in range(lx -pixel_scale_x, rx+rw +pixel_scale_x):
						val_sunglasses = distored_sunglasses[i-(ly-pixel_scale_y), j-(lx-pixel_scale_x)]
						if np.all(val_sunglasses < 250):
							filtered_img[i,j] = val_sunglasses

				mouth = mouth_cascade.detectMultiScale(roi_gray_mouth)
				
				#Mouth
				if len(mouth) > 0:
					mouth_in_image = sorted(mouth, key=lambda m: m[2]*m[3])[-1]

					[mx,my,mw,mh] = mouth_in_image

					mx += x
					my += (ly+ry)/2

					#Ratio of the mouth over which it is to be applied
					mc1 = 0.5
					mc2 = 0.8

					mc3 = 0.3
					mc4 = 0.7

					jw_real  = int(SCALE_FACTOR_JNT*(mc2-mc1)*mw)

					distorted_jnt = cv2.resize(jnt, (jw_real, int(1.25*jw_real) )) #0.8 magic number to conserve ratio of joint image

					# for i in range(mx + int(mc1*mw), mx+ int(mc1*mw)+ jw_real):
					# 	for j in range(my + int(mc3*mh), my + int(mc3*mh) + int(0.8*jw_real)):
							
					# 		val_jnt = distorted_jnt[i-(mx + int(mc1*mw)), j-(my + int(mc3*mh))]
							
					# 		if np.all(val_jnt < 253): #since opencv refuses to work with pngs
					# 			filtered_img[i,j] = val_jnt

					if DISPLAY_BOUNDRY_BOX:
						cv2.rectangle(filtered_img,(mx,my),(mx+mw,my+mh),(0,0,255),2)
						#For joint
						cv2.rectangle(filtered_img,(mx + int(mc1*mw), my + int(mc3*mh)),(mx+ int(mc1*mw)+ jw_real, my + int(mc3*mh) + int(1.25*jw_real) ),(0,255,0),2)

				#For Hat
				hx = x
				hy = y - int(0.3*h)
				hw = w
				hh = int(0.3*h)

				if DISPLAY_BOUNDRY_BOX:
					cv2.rectangle(filtered_img,(hx,hy),(hx+hw,hy+hh),(255,0,255),2)




				img = filtered_img
				
			cv2.imshow('frame',img)
	else:
		cv2.imshow('frame',img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()