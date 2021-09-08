from image_drawer import ImageDrawer
from capture_screenshot import capture_screenshot

import os
import sys
import argparse
import time

import numpy as np
import torch
from torchvision import datasets, models, transforms
import cv2


def main():
    parser = argparse.ArgumentParser(description='Take a screenshot every x seconds')
    parser.add_argument('delay', type=float, help='delay between each screenshot', default=5.0)

    args = parser.parse_args()

    delay = args.delay

    icons_path = 'league_icons/'
    image_drawer = ImageDrawer(icons_path + 'champions', icons_path + 'minimap', icons_path + 'fog',
                               icons_path + 'misc', resize=(256, 256))

    i = 0
    while True:
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

        pil_img = transforms.ToPILImage()(img)
        pil_img.save('{}.png'.format(i))

        time.sleep(delay)
        i += 1


if __name__ == "__main__":
    main()
