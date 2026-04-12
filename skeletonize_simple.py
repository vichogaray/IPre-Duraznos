import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize

if len(sys.argv) < 2:
    print("Uso: python skeletonize_simple.py <ruta_imagen>")
    sys.exit(1)

path = sys.argv[1]
img = imread(path, as_gray=True)
binary = img > 0.5
skeleton = skeletonize(binary)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(binary, cmap="gray")
axes[0].set_title("Máscara binaria")
axes[0].axis("off")

axes[1].imshow(skeleton, cmap="gray")
axes[1].set_title("Esqueleto")
axes[1].axis("off")

plt.tight_layout()
plt.show()
