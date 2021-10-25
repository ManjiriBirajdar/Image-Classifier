# Camera Capture + Preview
# @Source: https://github.com/rohit--jain/AI-Demo

from gpiozero import Button
from picamera import PiCamera
from datetime import datetime
from signal import pause
 
preview_button = Button(23)
capture_button = Button(21)
image_Capture = PiCamera()

resol_X = 2560
resol_Y = 1920
enable_Run = True
#global no_Pics = 0
 
def store_Image():
    timestamp = datetime.now().isoformat()
    image_Capture.resolution = (resol_X, resol_Y)
    image_Capture.capture('/home/pi/Documents/code/dataset/%s.jpg' % timestamp)
    #no_Pics = no_Pics + 1
 
#print("Press Preview / Capture Button for Action...") 
while(enable_Run):    
    preview_button.when_pressed = image_Capture.start_preview 
    preview_button.wait_for_press()  
    preview_button.when_released = image_Capture.stop_preview
    #print("Camera Preview Stopped")    
    
    #print("Camera Preview Started")
    #capture_button.when_pressed = image_Capture.start_preview 
    capture_button.wait_for_press()
    capture_button.when_pressed = store_Image
    #print("Camera Shot taken!")

#print("Total Number of Pictures Clicked: "+ str(no_Pics))
pause()
