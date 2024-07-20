import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
# import rlds
from PIL import Image


# load_dataset_path = '/home/kunni/plusai_ws/openvla/datasets/austin_buds_dataset_converted_externally_to_rlds/0.1.0'
load_dataset_path = '/home/kunni/plusai_ws/openvla/datasets/cmu_franka_exploration_dataset_converted_externally_to_rlds/0.1.0'


loaded_dataset = tfds.builder_from_directory(load_dataset_path).as_dataset(split='all')

for e in loaded_dataset:
  for s in e['steps']:
    image = s['observation']['image']
    image = image.numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    print(s['action'])
    image = Image.fromarray(image, mode='RGB')
    image.save("cmu_franka.png", "PNG")