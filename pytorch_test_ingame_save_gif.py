from pytorch_model import create_model
from image_drawer import ImageDrawer
from capture_screenshot import capture_screenshot

import os
import sys
import argparse
import time
import colorsys

import torch
import numpy as np
from mss import mss
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, models, transforms


def draw_box(img, label, x1, y1, x2, y2, color, thickness=1, text_color=[230, 230, 230]):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    label_box_thickness = thickness / 3.0
    text_size = cv2.getTextSize(label, 0, fontScale=label_box_thickness, thickness=thickness)[0]

    offset = 3

    label_box_point = (x1 + text_size[0], y1 - text_size[1] - offset)
    img = cv2.rectangle(img, (x1, y1), label_box_point, color, thickness=-1)
    img = cv2.putText(img, label, (x1, y1 - offset), cv2.FONT_HERSHEY_SIMPLEX, label_box_thickness, text_color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


score_threshold = 0.8


def main():
    parser = argparse.ArgumentParser(description='Detect ingame using model')
    parser.add_argument('model_path', help='path to model weights (.pt)')
    parser.add_argument('num_classes', type=int, help='number of classes (149)', default=149)
    parser.add_argument('gif_output', help='name of output gif')

    args = parser.parse_args()

    model_path = args.model_path
    num_classes = args.num_classes
    gif_output = args.gif_output

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = create_model(num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    icons_path = 'league_icons/'
    image_drawer = ImageDrawer(icons_path + 'champions', icons_path + 'minimap', icons_path + 'fog',
                               icons_path + 'misc', resize=(256, 256))

    champion_to_color = {k: colorsys.hsv_to_rgb(v / num_classes, 1.0, 1.0) for k, v in image_drawer.champion_to_id.items()}

    frames = []

    while len(frames) < 1000:
        screenshot = capture_screenshot()
        h, w, c = screenshot.shape
        minimap_ratio = 800 / 1080
        minimap_x = int(minimap_ratio * h)

        minimap_size = h - minimap_x
        minimap = screenshot[-minimap_size:, -minimap_size:]

        h, w, c = minimap.shape
        left = 0
        right = 0
        top = 0
        bottom = 0

        for x in range(w):
            y = int(h / 2)
            r, g, b = minimap[y][x]
            if r == 0 and g == 0 and b == 0:
                left = x
                break

        for x in range(w - 1, 0, -1):
            y = int(h / 2)
            r, g, b = minimap[y][x]
            if r == 0 and g == 0 and b == 0:
                right = x
                break

        for y in range(h):
            x = int(w / 2)
            r, g, b = minimap[y][x]
            if r == 0 and g == 0 and b == 0:
                top = y
                break

        for y in range(h - 1, 0, -1):
            x = int(w / 2)
            r, g, b = minimap[y][x]
            if r == 0 and g == 0 and b == 0:
                bottom = y
                break

        minimap = minimap[top - 1:bottom + 1, left - 1:right + 1]

        h, w, c = minimap.shape
        if h == 0 or w == 0:
            print('Could not detect game')
            continue
        minimap = cv2.resize(minimap, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        orig_minimap = minimap
        minimap = minimap.copy().transpose((2, 0, 1)).astype(np.float)
        minimap /= 255

        img = torch.from_numpy(minimap)
        img = img.type(torch.float32)

        with torch.no_grad():
            img = img.to(device)
            predictions = model([img])
            img = img.cpu().numpy()

            for prediction in predictions:
                boxes = prediction['boxes'].cpu().numpy().tolist()
                labels = prediction['labels'].cpu().numpy().tolist()
                scores = prediction['scores'].cpu().numpy().tolist()
                output = ''

                if len(boxes) == 0:
                    continue

                img = img.transpose((1, 2, 0))
                for label, box, score in zip(labels, boxes, scores):
                    if score > score_threshold:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        label = image_drawer.id_to_champion[label]

                        img = img.copy()
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1)

                        text = '{} {:.2f}'.format(label, score)
                        output += text + '\n'
                        # print(text)

                        color = champion_to_color[label]
                        img = draw_box(img, label, x1, y1, x2, y2, color)

                img = img.transpose((2, 0, 1))

                img = torch.from_numpy(img)
                pil_img = transforms.ToPILImage()(img)
                frames.append(pil_img)
                print(len(frames))

    frames[0].save(gif_output, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    main()
