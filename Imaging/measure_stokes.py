from vmbpy import VmbSystem,PixelFormat,VmbCameraError
from os import abort
import numpy as np
import pandas as pd
import thorlabs_apt as apt
from tqdm import tqdm
from time import sleep

exposure_time_us=1000*3.8
n_img=16
save_dir="./dataset/"
cam_id="DEV_1AB22C046DED"
QWP_fast=0

def get_camera(camera_id):
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            return cams[0]

def setup_cam(cam):
    exposure_time = cam.ExposureTime
    exposure_time.set(exposure_time_us)
    gain=cam.Gain
    gain.set(0.0)
    cam.set_pixel_format(PixelFormat.Mono12)

#print(get_camera(False))

fo=np.load(save_dir+"alphabeta.npz")
alpha_array=fo["arr_0"]
beta_array=fo["arr_1"]
print("alpha and beta array loaded.")

motor1=apt.Motor(27266776)
motor2=apt.Motor(27006280)
print("Moter loaded.")

result=np.zeros((100, 1216, 1936), dtype=np.uint16)
with VmbSystem.get_instance():
    with get_camera(cam_id) as cam:
        setup_cam(cam)
        print("Camera loaded.")

        for i in tqdm(range(alpha_array.shape[0])):
            motor1.move_to(-alpha_array[i],blocking=True)
            motor2.move_to(-beta_array[i]+QWP_fast,blocking=True)
            for frame in cam.get_frame_generator(limit=n_img):
                result[i%100]=result[i%100]+frame.as_numpy_ndarray()[:,:,0].astype(np.uint16)
            if i%100==99:
                np.savez_compressed(save_dir+"data%d.npz"%(i//100),result)
                result=np.zeros((100, 1216, 1936), dtype=np.uint16)

print("Motor back to zero")
motor1.move_to(0,blocking=True)
motor2.move_to(0,blocking=True)
input("All finished. Press ENTER to exit...")
