import pyautogui as pgui


def input_box_position():
    input('左上端にマウスカーソルを置いて Enter キーを押してください（このウィンドウはアクティブのまま）:')
    x0, y0 = pgui.position()
    input('右下端にマウスカーソルを置いて Enter キーを押してください（このウィンドウはアクティブのまま）:')
    x1, y1 = pgui.position()
    print('OK')
    return (x0, y0, x1, y1)