import os
import sys

def configure_path():
    absolute_path_to_dlls = r"C:\Users\Ultrafast\Documents\Thorlabs\thorcam_dlls\64_lib"
    os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
    try:
        os.add_dll_directory(absolute_path_to_dlls)
    except AttributeError:
        pass
configure_path()

import numpy as np
import pandas as pd
import thorlabs_apt as apt
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from tqdm import tqdm
from time import sleep


sdk=TLCameraSDK()
sleep(0.1)
available_cameras = sdk.discover_available_cameras()
sleep(0.1)
camera=sdk.open_camera(available_cameras[0])
print("Camera loaded.")
motor1=apt.Motor(27266776)
motor2=apt.Motor(27005394)
print("Moter loaded.")

import pyvisa
from ThorlabsPM100 import ThorlabsPM100
rm = pyvisa.ResourceManager()
rm.list_resources()
inst = rm.open_resource('USB0::0x1313::0x8078::P0040907::INSTR', timeout=1)
powermeter = ThorlabsPM100(inst=inst)
powermeter.configure.scalar.power()
powermeter.sense.power.dc.range.auto = 1 # auto range on, 1 = on, 0 = off
powermeter.sense.correction.wavelength = 425
print("Powermeter loaded.")

class Measurement:
    def __init__(self,n_shot,expose_ms,motor1=motor1,motor2=motor2,QWP_fast=-2,
                 camera=camera,powermeter=powermeter):
        print("New measurement created.")
        self.QWP_fast=QWP_fast
        self.motor1=motor1
        self.motor2=motor2
        self.camera=camera
        self.shot_time=expose_ms/1000
        self.n_shot=n_shot

        self.camera.exposure_time_us = round(self.shot_time*1000000)
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.image_poll_timeout_ms = 1000
        self.camera.arm(2)
        self.camera.issue_software_trigger()

        self.powermeter=powermeter

    def end_measurement(self):
        self.motor1.move_to(0,blocking=True)
        self.motor2.move_to(0,blocking=True)
        print("Motors back to zero")
        self.camera.disarm()
        sleep(0.1)
        self.camera.dispose()
        sleep(0.1)
        print("Camera off")
        
    # Get raw intensities.
    def get_img(self):
        
        frame = self.camera.get_pending_frame_or_null()
        result = np.copy(frame.image_buffer)
        for i in range(1,self.n_shot):
            frame = self.camera.get_pending_frame_or_null()
            result = result + np.copy(frame.image_buffer)
        result=result

        return result
        
    def measure_train(self,n_alpha,n_beta,save_dir="./training_set/",skip_until=None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        alpha_array=np.linspace(0,90,n_alpha+1)[:-1]
        beta_array=np.linspace(0,180,n_beta+1)[:-1]

        if skip_until is not None:
            for i in range(n_alpha):
                if alpha_array[i]>skip_until:
                    alpha_array=alpha_array[i:]
                    break
        
        for i in range(n_alpha):
            cur_result=np.zeros((n_beta,1080,1440),dtype=np.uint16)
            cur_power=np.zeros(n_beta,dtype=np.float32)
            self.motor2.move_to(-alpha_array[i],blocking=True)
            print("Measuring %.2f"%alpha_array[i])
            for j in tqdm(range(n_beta)):
                self.motor1.move_to(-beta_array[j]+self.QWP_fast,blocking=True)
                cur_result[j]=self.get_img()
                cur_power[j]=self.powermeter.read
            np.savez_compressed(save_dir+"alpha%.2f.npz"%alpha_array[i],cur_result,cur_power)

    def measure_test(self,save_dir):
        fo=np.load(save_dir+"alphabeta.npz")
        alpha_array=fo["arr_0"]
        beta_array=fo["arr_1"]
        
        cur_result=np.zeros((alpha_array.shape[0],1080,1440),dtype=np.uint16)
        cur_power=np.zeros(alpha_array.shape[0],dtype=np.float32)
        for i in tqdm(range(alpha_array.shape[0])):
            self.motor2.move_to(-alpha_array[i],blocking=True)
            self.motor1.move_to(-beta_array[i]+self.QWP_fast,blocking=True)
            cur_result[i]=self.get_img()
            cur_power[i]=self.powermeter.read
        np.savez_compressed(save_dir+"data.npz",cur_result,cur_power)

msm=Measurement(64,21)
msm.measure_train(36,36)
msm.measure_test("./val_set/")
msm.measure_test("./test_set/")
msm.end_measurement()
sdk.dispose()
sleep(0.1)
print("finished")

