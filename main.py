from InferenceEngine import InferenceEngine
import cv2 as cv
import numpy as np
from PIL import Image,ImageFont,ImageDraw


def show_text(img,text,pos):
    img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    font = ImageFont.truetype(font='msyh.ttc', size=18)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 0, 0))  # PIL中RGB=(255,0,0)表示红色
    img_cv = np.array(img_pil)                         # PIL图片转换为numpy
    img = cv.cvtColor(img_cv, cv.COLOR_RGB2BGR)      # PIL格式转换为OpenCV的BGR格式
    return img


if __name__ == "__main__":
    engine = InferenceEngine(conf_thres=0.5)
    image = cv.imread("data/images/bus.jpg")
    capture = cv.VideoCapture(0)
    area = 20.0  # 视觉范围内的面积

    while True:
        ref, frame = capture.read()
        if not ref:
            print("Fail to capture the frame.")
            continue
        result = engine.detect(frame)
        n = 0
        for i in result:
            if i['class'] == 'person':
                n = n+1
                position = i['position']
                cv.rectangle(frame, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]),
                             (0, 255, 0), 1, 4)
        density = n/area
        monitor_text = "监测到 {} 人，当前视野内人群密度为 {:.2f} 人/平方米。".format(n, density)
        frame = show_text(frame, monitor_text, (10, 10))
        cv.imshow('PC Camera', frame)

        c = cv.waitKey(1) & 0xff
        if c == 27:
            cv.destroyAllWindows()
            break
