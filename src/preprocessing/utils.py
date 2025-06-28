import cv2
import numpy as np

def calculate_extrinsic(rvec, tvec):
    Rc = cv2.Rodrigues(rvec)[0].T
    tc = tvec.reshape([3,1])
    extrinsic = np.concatenate([Rc,tc], axis=1)
    return extrinsic


if __name__ == '__main__':

    rvec = np.array([[1.2022354494810004], [0.0], [2.9024531564007323]])
    tvec = np.array([-1.7744795289584235e-08, 0.9999999999999999, 2.5031579784263953])
    extrinsic = calculate_extrinsic(rvec, tvec)
    
    print(extrinsic.tolist())
