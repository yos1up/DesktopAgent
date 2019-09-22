import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm_notebook as tqdm
from collections import Counter, defaultdict, deque
import os, sys, glob, copy, json, time, pickle

from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, \
    optimizers, serializers, utils, Variable, dataset, datasets, using_config, training, iterators
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
import chainer

import mss
from PIL import Image

import pyautogui as pgui

# import pytesseract
# import pyocr
from ocr import NonBlockingOCR, BlockingOCR, BlockingBatchOCRSlave, NonBlockingBatchOCRSlave

class Timer:
    def tic(self):
        self.t0 = time.time()
    def toc(self):
        return time.time() - self.t0
_timer = Timer()
_tic, _toc = _timer.tic, _timer.toc


class DesktopTrainer:
    """
    Agent と環境の間を仲介する．
    """
    def __init__(self, agent, ltrb=(0, 0, 640, 480), ltrb_score=(0, 0, 100, 20),
            action=[[],['left'],['up'],['right'],['down']], wait=None, press_mode='trigger'):
        assert press_mode in ['trigger', 'hold']

        self.sct = mss.mss()
        self.ltrb = ltrb
        self.ltrb_score = ltrb_score
        self.action = action
        self.agent = agent 
        self.screen_info_log = deque()
        self.gameover = False
        self.loopcnt = 0
        self.wait = wait
        self.press_mode = press_mode

        self.reward_log_by_action = defaultdict(list)
        
    def run(self):
        """
        学習を開始する．
        """
        print("game started")
        while True:
            _tic()
            info = self._get_screen_info()
            time_scr = _toc()
            _tic()
            selected_action = self.agent.determine(info["screen_array"])
            time_det = _toc()
            info["selected_action"] = selected_action
            # update key pressing w.r.t. selected_action
            if self.press_mode == 'hold':
                if len(self.screen_info_log) >= 1:
                    last_action = self.screen_info_log[-1]["selected_action"]
                    on_keys = set(self.action[selected_action]) - set(self.action[last_action])
                    off_keys = set(self.action[last_action]) - set(self.action[selected_action])
                else:
                    on_keys, off_keys = self.action[selected_action], []
                for k in off_keys:
                    pgui.keyUp(k)
                for k in on_keys:
                    pgui.keyDown(k)
            elif self.press_mode == 'trigger':
                for k in self.action[selected_action]:
                    pgui.press(k)
            else:
                raise ValueError
            # https://pyautogui.readthedocs.io/en/latest/introduction.html
            
            # logging screen info
            self.screen_info_log.append(info)
            # このログは増えていくが，スコアが読み取られて結果が Agent に通達されると，古いものから削除されていく．

            pre_score, post_score, action, reward = np.nan, np.nan, np.nan, np.nan
            if len(self.screen_info_log) >= 20:
                # 古い方から 20 個連続のスナップショットを解析対象とする．
                screen_info_log_oldest = list(itertools.islice(self.screen_info_log, 0, 20))
                score_ocr_results = [log["score_ocr"].result for log in screen_info_log_oldest]
                # スコア読み取りが全て完了している場合
                if all([s is not None for s in score_ocr_results]):
                    # ゲームが続いているかの判定． とりあえず，スコアが読み取れたかどうかを使って判定する．
                    score = np.array([r["value"] for r in score_ocr_results])
                    if not self.gameover and all(np.isnan(score)):
                        self.gameover = True
                        # TODO: ここで agent に負の observation を伝える？
                    elif self.gameover and not any(np.isnan(score)):
                        self.gameover = False
                    
                    # observation の処理．
                    if np.count_nonzero(np.isnan(score)) * 2 <= len(score):
                        pre_score = np.nanmedian(score[:-1])
                        post_score = np.nanmedian(score[1:])
                        if not self.gameover:
                            # agent に observation を伝える．
                            r = post_score - pre_score
                            reward = np.sign(r) * np.log(1 + np.abs(r))
                            key_frame = len(score) // 2 - 1
                            s0 = screen_info_log_oldest[key_frame]["screen_array"]
                            action = screen_info_log_oldest[key_frame]["selected_action"]
                            s1 = screen_info_log_oldest[key_frame+1]["screen_array"]
                            self.agent.observe([s0, action, reward, s1])
                            # この observation は最新より少し古いが，学習に使うだけなので問題はない．
                            self.reward_log_by_action[action].append(reward)
                    # 一番古いスナップショットは削除する．
                    self.screen_info_log.popleft()
                    

            report_str = '\r\033[K' # 復帰して行削除．
            report_str += '🌞🌙'[self.gameover]
            report_str += '  [{}]:  📈  {} => {} 💪  {} 💰  {:.2f} [scr] {:.2f}s [det] {:.2f}s [queued] {}'.format(
                self.loopcnt, pre_score, post_score, action, reward, time_scr, time_det, len(self.screen_info_log)) 

            for k, v in self.reward_log_by_action.items():
                report_str += "[{}: {:.2f}±{:.2f}]".format(k, np.mean(v), np.std(v))

            # show report_str   
            print(report_str, end='')
            self.loopcnt += 1
            if self.wait is not None:
                time.sleep(self.wait)


    
    @staticmethod
    def _update_log(dq, item, maxlen):
        dq.append(item)
        if len(dq) > maxlen: dq.popleft()
    
    def _get_screen_info(self):
        # 130 x 50 程度の画像の ocr に 0.25 秒もかかるのをなんとかしたい
        screen_ary = np.array(self.sct.grab(self.ltrb))[...,:3].transpose(2,0,1) # (color[BGR], height, width) 
        # score_str = pytesseract.image_to_string(np.array(self.sct.grab(self.ltrb_score)), lang='eng', config='--oem 1 --psm 7')
        # score_digits = ''.join([s for s in score_str if s in '0123456789'])
        # score = int(score_digits) if score_digits != "" else np.nan
        score_sct = self.sct.grab(self.ltrb_score)
        score_img = Image.frombytes("RGB", score_sct.size, score_sct.bgra, "raw", "BGRX")
        timestamp = time.time()
        return {
            "screen_array": screen_ary,
            "score_ocr": NonBlockingBatchOCRSlave(score_img),
            # "score_string": score_str,
            # "score": score,
            "timestamp": timestamp          
        }
