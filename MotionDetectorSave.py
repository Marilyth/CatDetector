import picamera
import time
import io
import cv2
import imutils
import numpy as np
import datetime
import imageio
import os

DIFFERENCE=40
MINPX2=4000
GIFPERFRAME=1
WAITTIME=1
TIMELAPSE=10
CYCLES=-1

def take_picture(camera, name='test.jpg', is_stream=True):
    stream = io.BytesIO()
    if is_stream:
        camera.capture(stream, 'jpeg')
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, 1)
        return image

    camera.capture("/var/www/html/catDetector/" + name)
    return None

def save_picture(image, name='test.jpg'):
    cv2.imwrite("/var/www/html/catDetector/" + name, image)

def save_pictures(images, delta, name='test.jpg'):
    print("Saving chain image")
    vis = np.concatenate(images, axis=0)
    d_vis = np.concatenate(delta, axis=0)
    save_picture(vis, name+"chain.jpg")
    save_picture(d_vis, name+"delta.jpg")
    for i, image in enumerate(images):
        images[i] = image[:, :, ::-1]
    print("Saving gif")
    imageio.mimsave("/var/www/html/catDetector/" + name + ".gif", images, duration=GIFPERFRAME)
    print("Optimizing gif")
    os.system("sudo gifsicle -O3 --lossy=35 /var/www/html/catDetector/"+name+".gif -o /var/www/html/catDetector/"+name+".gif")

def process_image(frame, last_frame_gray=None, last_frame=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if last_frame_gray is not None:
        frameDelta = cv2.absdiff(last_frame, frame)
        frameDelta = np.sum(frameDelta, axis = 2)/(3*255)
        print("Starting")
        print(frameDelta)
        print("Ended")
        thresh = cv2.threshold(frameDelta, 254, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return gray, thresh, cnts, frameDelta
    
    return gray

def main():
    cycle = 0
    frame_stack = []
    d_frame_stack = []
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 720)
        camera.framerate = 30
        camera.rotation = 180
        time.sleep(2)
        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g

        frame = take_picture(camera)
        save_picture(frame)
        #return
        last_frame_gray = process_image(frame)
        last_frame = frame
        last_snap = time.time()

        while CYCLES < 0 or cycle < CYCLES:
            if len(frame_stack) > 0 and time.time() - last_snap > 30:
                print("Motion stopped, saving " + str(len(frame_stack)) + " images")
                save_pictures(frame_stack, d_frame_stack, datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
                frame_stack = []
                d_frame_stack = []
            cycle += 1
            #print("Currently on cycle " + str(cycle))
            current_frame = take_picture(camera)
            gray, thresh, contours, delta = process_image(current_frame, last_frame_gray, last_frame)
            last_frame_gray = gray
            last_frame = current_frame
            hits = 0
            for contour in contours:
                if cv2.contourArea(contour) >= MINPX2:
                    hits += 1
            if hits > 0:
                last_snap = time.time()
                date = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                cv2.putText(current_frame, date, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                frame_stack.append(current_frame)
                d_frame_stack.append(thresh)
                #save_picture(current_frame, date+".jpg")
                for contour in contours:
                    if cv2.contourArea(contour) >= MINPX2:
                        print("Found contour of size: " + str(cv2.contourArea(contour)) + "px^2")
            
            save_picture(current_frame)
            time.sleep(WAITTIME)

main()
