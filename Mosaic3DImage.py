import pandas as pd
import os
from astropy.io import fits
import numpy as np
import time
import gc
gc.collect()

"""
Changeable Parameters:
shape = (1665, 181, 61) — a larger box for the mosaic.
ref_center = [30, 90] — the reference center. Each flux-calibrated frame 
(using either your defined center or the OSIRIS-defined center) will be aligned to this ref_center.
"""

# === Config ===
DATA_PATH = "/Users/mylin/ngc1068/keck/NGC1068-Anne-Calibration/mylin/"
CSV_FILE = os.path.join(DATA_PATH, "center_N1068_225_mylin.txt")
myldf = pd.read_csv(CSV_FILE, delimiter=" ")
ref_center = [30, 90]  # [x, y]
threshold = 1.7 # threshold for stdev but used thresh*thresh for Var 
output_FileName = "mylin-NGC1068-kbb-Refactoring.fits"

# === Load input data ===
fileNames = myldf['FileName']
Xcenter = myldf['Xcen']
Ycenter = myldf['Ycen']

# === Initialize cubes ===
shape = (1665, 181, 61, len(fileNames)) # [z, y, x, FileNumber]
cube_shape = shape[:3] # [z, y, x]
all_data = np.zeros(shape, dtype=np.float32)
all_IntAuxData = np.zeros(shape, dtype=np.float32)
all_noise_data = np.zeros(shape, dtype=np.float32)
noise_data = np.zeros(cube_shape, dtype=np.float32)
first_mean_data = np.zeros(cube_shape, dtype=np.float32)
stacked_count_cube = np.zeros(cube_shape, dtype=np.float32)
deviation_data = np.zeros(shape, dtype=np.float32)
noise_variance = np.zeros(shape, dtype=np.float32)
second_mean_data = np.zeros(cube_shape, dtype=np.float32)
second_mean_good_count = np.zeros(cube_shape, dtype=np.float32)

# === First loop: accumulate mean and count ===
for idx, fname in enumerate(fileNames):
    print(f'First loop: {idx}')
    fits_path = os.path.join(DATA_PATH, f"mylin_{fname}")
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        noise = hdul[1].data
        int_aux = hdul[2].data
        int_aux[int_aux == 9] = 1
        data *= int_aux
    
    # Compute offset from image edge left, right, up, down
    z, y, x = data.shape
    offset_left = round(Xcenter[idx])
    offset_right = x - offset_left
    offset_down = round(Ycenter[idx])
    offset_up = y - offset_down
    
    ys = slice(ref_center[1] - offset_down, ref_center[1] + offset_up)
    xs = slice(ref_center[0] - offset_left, ref_center[0] + offset_right)
    
    # Put data into 4D (z,y,x,fileNumber)
    all_data[:, ys, xs, idx] = data
    all_IntAuxData[:, ys, xs, idx] = int_aux
    all_noise_data[:, ys, xs, idx] = noise
    
    # Put data into the parent image & calculate first mean
    first_mean_data[:, ys, xs] += data
    stacked_count_cube[:, ys, xs] += int_aux
    
    # Calculate Noise
    noise_data[:, ys, xs] += noise ** 2

    del data
    del noise
    del int_aux

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    first_mean_data = np.true_divide(first_mean_data, stacked_count_cube)
    noise_data = np.true_divide(np.sqrt(noise_data), stacked_count_cube)
    first_mean_data[np.isnan(first_mean_data)] = 0
    noise_data[np.isnan(first_mean_data)] = 0

# === Second loop: compute squared deviations and counts ===
for idx in range(len(fileNames)):
    print(f'Second loop: {idx}')
    valid_mask = all_IntAuxData[:, :, :, idx] > 0
    diff = all_data[:, :, :, idx] - first_mean_data
    deviation_data[:, :, :, idx][valid_mask] = diff[valid_mask] ** 2
    noise_variance[:, :, :, idx][valid_mask] = all_noise_data[:, :, :, idx][valid_mask] ** 2

    del valid_mask
    del diff

# Calculate the variance (data - mean)**2 / N 
with np.errstate(divide='ignore', invalid='ignore'):
    variance_data = np.sum(deviation_data + noise_variance, axis=3) / stacked_count_cube
    variance_data[np.isnan(variance_data)] = 0
    
del first_mean_data
# === Third pass: filtered average using deviation clip ===   
for idx in range(len(fileNames)):
    print(f'Third loop: {idx}')
    valid_mask = (all_IntAuxData[:, :, :, idx] > 0) & \
                 (deviation_data[:, :, :, idx] < (threshold ** 2) * variance_data)

    second_mean_data[valid_mask] += all_data[:, :, :, idx][valid_mask]
    second_mean_good_count[valid_mask] += 1

    del valid_mask

# Final average and cutout
with np.errstate(divide='ignore', invalid='ignore'):
    output = np.true_divide(second_mean_data, second_mean_good_count)
    output[np.isnan(output)] = 0

# Create HDUs
primary_hdu = fits.PrimaryHDU(output)  # First HDU, usually the main data
variance_hdu = fits.ImageHDU(variance_data, name="VARIANCE")
noise_hdu = fits.ImageHDU(noise_data, name="NOISE")
count_hdu = fits.ImageHDU(second_mean_good_count, name="MASK")

# Combine into an HDUList
hdulist = fits.HDUList([primary_hdu, variance_hdu, noise_hdu, count_hdu])

# Write to file
hdulist.writeto(output_FileName, overwrite=True)