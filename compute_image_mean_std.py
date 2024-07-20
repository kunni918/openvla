import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append((os.path.basename(filename), np.array(img)))
    return images

def compute_mean_std_per_image(images):
    for filename, img in images:
        # Compute mean and std for each channel
        mean = [np.mean(img[:, :, i]) for i in range(3)]
        std = [np.std(img[:, :, i]) for i in range(3)]
        print(f"{filename}, shape {img.shape}: Mean - {mean}, Std - {std}")

folder_path = '.'
images = load_images_from_folder(folder_path)

compute_mean_std_per_image(images)