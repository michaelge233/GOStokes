from os import abort
from vmbpy import VmbSystem,PixelFormat,VmbCameraError
import numpy as np
import matplotlib.pyplot as plt


exposure_time_us=1000*3.8
n_img=16
save_dir="./img/obj1.npz"
cam_id="DEV_1AB22C046DED"


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

result=np.zeros((1216, 1936), dtype=np.uint16)
with VmbSystem.get_instance():
    with get_camera(cam_id) as cam:
        setup_cam(cam)
        print("Camera loaded.")
        for frame in cam.get_frame_generator(limit=n_img):
            result=result+frame.as_numpy_ndarray()[:,:,0].astype(np.uint16)
        np.savez_compressed(save_dir,result)
plt.imshow(result)
plt.show()
print("All finished.")
