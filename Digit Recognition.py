import cv2
import numpy as np
import tensorflow as tf

m_new = tf.keras.models.load_model('Digit.h5')

img = np.ones((600,600),dtype='uint8')*255
# 255 - White , 0 - Black
img[100:500,100:500] = 0
windowName = 'Digits Project'
cv2.namedWindow(windowName)
def draw_type(event,x,y,a,b):
# event - Mouse Events = Left click, Right CLick, Mouse Move
# x and y are the centres of the shape
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state = True
        cv2.circle(img,(x,y),10,(255,255,255),-1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if state == True:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
    else:
        state = False

state = False # Flags in Microcontroller
cv2.setMouseCallback(windowName,draw_type)
while True:
    cv2.imshow(windowName,img)
    key = cv2.waitKey(1) # waiting for one milli second
    if key == ord('q'): 
        break
    elif key == ord('c'):
        img[100:500,100:500] = 0
    elif key == ord('p'):
        final_image = img[100:500,100:500]
        i = cv2.resize(final_image,(28,28)).reshape(1,28,28) # 
        print(m_new.predict_classes(i))
cv2.destroyAllWindows()
