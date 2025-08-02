import cv2
import numpy as np


def img_range(x):
    y = (x + abs(x.min()))
    x = y / y.max()
    x = x * 255
    x = x.astype(np.uint8)
    return x


def DFT(img, offset=11, mode='HP'):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    if mode == 'LP':
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - offset:crow + offset, ccol - offset:ccol + offset] = 1
    if mode == 'HP':
        mask = np.ones((rows, cols), np.uint8)*255
        x, y = crow - 3*offset//2, ccol - 3*offset//2
        ###################
        mask[:, ccol - offset//2:ccol + offset//2] = 0
        mask[crow - offset//2:crow + offset//2, :] = 0
        cv2.rectangle(mask, (x, y), (rows-x, cols-y), 0, -1)
        mask = mask//255
        mask = np.dstack([mask, mask])
        ######################

    # apply mask and inverse DFT
    dft_shift_masked = dft_shift * mask
    inv_masked = np.fft.ifftshift(dft_shift_masked)
    imginv_masked = cv2.idft(inv_masked)
    img_dft = cv2.magnitude(imginv_masked[:, :, 0], imginv_masked[:, :, 1])
    return img_dft
