import picamera
import time
import io
import cv2
import imutils
import numpy as np
import datetime
import imageio
import os

DIFFERENCE=10
MINPX2=1500
GIFPERFRAME=1
WAITTIME=1
TIMELAPSE=30
CYCLES=-1

class data_point:
    def __init__(self, hit_pixels, avg_colour, hour):
        self.hit_pixels = hit_pixels
        self.avg_colour = avg_colour
        self.hour = hour
    
    def __str__(self):
        return str(self.hit_pixels) + "\t" + str(self.avg_colour) + "\t" + str(self.hour)

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

def save_pictures(images, delta, pixelcount, name='test.jpg'):
    print("Saving chain image")
    vis = np.concatenate(images, axis=0)
    d_vis = np.concatenate(delta, axis=0)
    save_picture(vis, name+"chain.jpg")
    save_picture(d_vis, name+"delta.jpg")
    with open('/var/www/html/catDetector/data.txt', 'a') as file:
        for hit in pixelcount:
            file.write(str(hit) + "\n")
        file.write("\t\t\t\t" + str(np.mean([p.hit_pixels for p in pixelcount])) + "\n")
    for i, image in enumerate(images):
        images[i] = image[:, :, ::-1]
    print("Saving gif")
    imageio.mimsave("/var/www/html/catDetector/" + name + ".gif", images, duration=GIFPERFRAME)
    print("Optimizing gif")
    os.system("sudo gifsicle -O3 --lossy=35 /var/www/html/catDetector/"+name+".gif -o /var/www/html/catDetector/"+name+".gif")

def process_image(frame, last_frame=None):
    if last_frame is not None:
        #Convert to YUV without Y(brightness is not important) to represent human colour perception
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:,:,1:]
        yuv_2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2YUV)[:,:,1:]
        frameDelta = cv2.absdiff(yuv, yuv_2)
        frameDelta = np.sum(frameDelta, axis = 2)/2
        frameDelta = cv2.GaussianBlur(frameDelta,(21,21),0)
        thresh = cv2.threshold(frameDelta, DIFFERENCE, 1, cv2.THRESH_BINARY)[1]
        return thresh, frameDelta
    
    return frame

def main():
    cycle = 0
    frame_stack = []
    d_frame_stack = []
    hits_stack = []
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
        last_frame = frame
        last_snap = time.time()

        while CYCLES < 0 or cycle < CYCLES:
            if len(frame_stack) > 0 and (time.time() - last_snap > 60) or len(frame_stack) >= TIMELAPSE:
                if len(frame_stack) > 1:
                    print("Motion stopped, saving " + str(len(frame_stack)) + " images")
                    save_pictures(frame_stack, d_frame_stack, hits_stack, datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
                frame_stack = []
                d_frame_stack = []
                hits_stack = []
            cycle += 1
            current_frame = take_picture(camera)
            thresh, delta = process_image(current_frame, last_frame)
            if time.time() - last_snap > 60:
                last_frame = current_frame
            hits = np.sum(thresh)
            if hits > MINPX2:
                print(str(hits) + " pixels found")
                avg_colour = np.mean(current_frame[thresh.nonzero()], axis=0).astype(int)
                last_snap = time.time()
                frame_stack.append(current_frame)
                d_frame_stack.append(delta)
                hits_stack.append(data_point(int(hits), avg_colour, datetime.datetime.now().strftime("%H")))
            
            last_frame = current_frame
            
            save_picture(current_frame)
            time.sleep(WAITTIME)

main()
