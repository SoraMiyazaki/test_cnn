import cv2
import numpy as np

fn_1 = "C:/Users/mizutani/Desktop/CNN_pra/wave_image/us/case10/case101057.jpg"
fn_2 = "C:/Users/mizutani/Desktop/CNN_pra/wave_image/ud/case10/case101057.jpg"

img = cv2.imread(fn_2, 0)

img = img[200:700, :] # 画像のトリミング(ndarray)

# リサイズ
row, col = img.shape
size = (col // 2, row // 2)
img_resize = cv2.resize(img, size)

img_1 = img_resize[:, 0:500]
cv2.imwrite("resize2.jpg", img_1)