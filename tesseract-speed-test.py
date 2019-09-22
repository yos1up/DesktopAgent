from PIL import Image
import time
import numpy as np

import pytesseract
import pyocr

img = Image.open('tesseract_image.png')
print(img)
print(img.height, img.width)

t0 = time.time()
txt = pytesseract.image_to_string(img, lang='eng', config='--tessdata-dir /usr/local/Cellar/tesseract/4.1.0/share/tessdata_fast')
print(time.time() - t0, txt)

t0 = time.time()
txt = pytesseract.image_to_string(img, lang='eng', config='--tessdata-dir /usr/local/Cellar/tesseract/4.1.0/share/tessdata')
print(time.time() - t0, txt)

tools = pyocr.get_available_tools()

t0 = time.time()
txt = tools[0].image_to_string(img, lang='eng')
print(time.time() - t0, txt)

t0 = time.time()
txt = tools[1].image_to_string(img, lang='eng')
print(time.time() - t0, txt)

# 実行時間はそんなに大きく変わらない．

# pytesseract は img が array でも動作．また config オプションもある．
# pyocr はそれらが無い（？）

ary = np.array(img)
ary_large = np.tile(ary, [5, 5, 1])
img_large = Image.fromarray(ary_large)
print(img_large.height, img_large.width)

t0 = time.time()
txt = tools[1].image_to_string(img_large, lang='eng')
print(time.time() - t0, txt)

# まとめるほど速い．


