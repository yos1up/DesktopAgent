import numpy as np

import pyocr
# import pytesseract

import concurrent.futures
from PIL import Image

tools = pyocr.get_available_tools()
tools = {tool.get_name(): tool for tool in tools}
if 'Tesseract (C-API)' in tools.keys():
    tool = tools['Tesseract (C-API)']
elif 'Tesseract (sh)' in tools.keys():
    tool = tools['Tesseract (sh)']
else:
    raise RuntimeError

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def concat_image(images, axis=0):
    ws = np.array([im.width for im in images])
    hs = np.array([im.height for im in images])
    if axis == 0:
        ret = Image.new('RGB', (np.max(ws), np.sum(hs)))
        offs = 0
        for im in images:
            ret.paste(im, (0, offs))
            offs += im.height
    elif axis == 1:
        ret = Image.new('RGB', (np.sum(ws), np.max(hs)))
        offs = 0
        for im in images:
            ret.paste(im, (offs, 0))
            offs += im.height
    else:
        raise ValueError            
    return ret

# ====================================================

class NonBlockingBatchOCRMaster:
    def __init__(self):
        self.queries = []
    def query(self, image, slave_ref):
        self.queries.append({"image":image, "slave_ref":slave_ref})
        if len(self.queries) >= 10:
            # self._ocr_batch(self.queries) # これを別スレッドにする予定　
            executor.submit(self._ocr_batch, self.queries)
            self.queries = []
    def _ocr_batch(self, queries):
        # concated_image を作る
        concated_image = concat_image([q["image"] for q in queries], axis=0)
        res = tool.image_to_string(concated_image, lang='eng')
        # res = pytesseract.image_to_string(concated_image, lang='eng', config='--psm 8')
        # フリーズした・・・
        # res をパースして各 slave_ref の result に代入．
        lines = res.split('\n')
        for i, q in enumerate(queries):
            txt = lines[i] if len(lines) > i else "" # 個数違ったらどうするの？
            digits = ''.join([c for c in txt if c in '0123456789'])
            value = int(digits) if digits != "" else np.nan
            q["slave_ref"].result = {
                "text": txt,
                "value": value,
            }

non_blocking_batch_ocr_master = NonBlockingBatchOCRMaster()

class NonBlockingBatchOCRSlave:
    def __init__(self, img):
        """
        img の OCR が開始され，終了次第，結果が self.result に格納される．
        """
        self.result = None
        non_blocking_batch_ocr_master.query(img, self)

# ====================================================

class BlockingBatchOCRMaster:
    def __init__(self):
        self.queries = []
    def query(self, image, slave_ref):
        self.queries.append({"image":image, "slave_ref":slave_ref})
        if len(self.queries) >= 10:
            # concated_image を作る
            res = tool.image_to_string(concat_image([q["image"] for q in self.queries], axis=0), lang='eng')
            # res をパースして各 slave_ref に代入．
            lines = res.split('\n')
            for i, q in enumerate(self.queries):
                txt = lines[i] if len(lines) > i else "" # 個数違ったらどうするの？
                digits = ''.join([c for c in txt if c in '0123456789'])
                value = int(digits) if digits != "" else np.nan
                q["slave_ref"].result = {
                    "text": txt,
                    "value": value,
                }
            self.queries = []
blocking_batch_ocr_master = BlockingBatchOCRMaster()

class BlockingBatchOCRSlave:
    def __init__(self, img):
        """
        img の OCR が開始され，終了次第，結果が self.result に格納される．
        """
        self.result = None
        blocking_batch_ocr_master.query(img, self)

# ====================================================

class NonBlockingOCR:
    """
    「非同期に」「まとめて」OCR処理をしてくれるクラス．
    まずは「非同期に」だけでも実装しよう．
    """
    def __init__(self, img):
        """
        img の OCR が開始され，終了次第，結果が self.result に格納される．
        """
        self.result = None
        self.img = img
        self.state = executor.submit(self._ocr)

    def _ocr(self):
        txt = tool.image_to_string(self.img, lang='eng')
        digits = ''.join([c for c in txt if c in '0123456789'])
        value = int(digits) if digits != "" else np.nan
        self.result = {
            "text": txt,
            "value": value,
        }
        return 0

# ====================================================

class BlockingOCR:
    """
    普通のOCR．
    """
    def __init__(self, img):
        """
        img の OCR が開始され，終了次第，結果が self.result に格納される．
        """
        self.img = img
        self.state = None
        self._ocr()

    def _ocr(self):
        txt = tool.image_to_string(self.img, lang='eng')
        digits = ''.join([c for c in txt if c in '0123456789'])
        value = int(digits) if digits != "" else np.nan
        self.result = {
            "text": txt,
            "value": value,
        }



        
