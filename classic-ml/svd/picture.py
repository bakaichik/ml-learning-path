import numpy as np
import matplotlib.pyplot as plt
# Скачивание картинки.
from PIL import Image
import requests

# Ссылка на картинку.
url = "https://code.s3.yandex.net/math-adult/explanations/linal/1/svd/guinea-pig.png"

# Скачивание и сохранение картинки в переменную.
im = Image.open(requests.get(url, stream=True).raw)
# Превращение картинки в NumPy-массив.
guinea_pig = np.array(im).astype(float)

def show_image(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

U, s, VT = np.linalg.svd(guinea_pig, full_matrices=False)


X_approx = U[:, 0:402] @ S[0:402, 0:402] @ VT[0:402, :]
show_image(X_approx)