import os
import sys
sys.path.insert(0, '..')
import argparse
import time
import colorsys

from pytorch_model import create_model
from image_drawer import ImageDrawer
from capture_screenshot import capture_screenshot

import xml.etree.ElementTree as ET
import torch
import numpy as np
from mss import mss
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, models, transforms
from PIL import Image


class CocoBox:
    def __init__(self, name, xmin, ymin, xmax, ymax, score = 0.0):
        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score

    def in_range(self, threshold, other_box):
        if abs(other_box.xmin - self.xmin) < threshold and \
            abs(other_box.ymin - self.ymin) < threshold and \
            abs(other_box.xmax - self.xmax) < threshold and \
            abs(other_box.ymax - self.ymax) < threshold:
                return True
        return False

    def calculate_iou(self, other_box):
        xmin = max(other_box.xmin, self.xmin)
        ymin = max(other_box.ymin, self.ymin)
        xmax = min(other_box.xmax, self.xmax)
        ymax = min(other_box.ymax, self.ymax)

        intersection = (xmax - xmin) * (ymax - ymin)

        box1_area = (other_box.xmax - other_box.xmin) * (other_box.ymax - other_box.ymin)
        box2_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)

        union = box1_area + box2_area - intersection

        iou = intersection / union
        return iou

def draw_box(img, label, x1, y1, x2, y2, color, thickness = 1, text_color = [230, 230, 230]):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    label_box_thickness = thickness / 3.0
    text_size = cv2.getTextSize(label, 0, fontScale=label_box_thickness, thickness=thickness)[0]

    offset = 3

    label_box_point = (x1 + text_size[0], y1 - text_size[1] - offset)
    img = cv2.rectangle(img, (x1, y1), label_box_point, color, thickness=-1)
    img = cv2.putText(img, label, (x1, y1 - offset), cv2.FONT_HERSHEY_SIMPLEX, label_box_thickness, text_color, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser(description='Detect ingame using model')
    parser.add_argument('model_path', help='path to model weights (.pt)')
    parser.add_argument('num_classes', type=int, help='number of classes (149)', default=149)
    parser.add_argument('image_path', help='path to folder of images')
    parser.add_argument('label_path', help='path to folder of labels')

    args = parser.parse_args()

    model_path = args.model_path
    num_classes = args.num_classes
    image_path = args.image_path
    label_path = args.label_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = create_model(num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    icons_path = '../league_icons/'
    image_drawer = ImageDrawer(icons_path + 'champions', icons_path + 'minimap', icons_path + 'fog',
                               icons_path + 'misc', resize=(256, 256))

    fig = plt.figure(figsize=(25, 25))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    plt.ion()

    champion_to_color = {k: colorsys.hsv_to_rgb(v / num_classes, 1.0, 1.0) for k, v in image_drawer.champion_to_id.items()}
    score_threshold = 0.5

    for f in os.listdir(image_path):
        ax1.set_title(f)
        ax2.set_title('truth')
        ax3.set_title('prediction')

        base_name = f[:f.find(".")]
        full_label_path = os.path.join(label_path, base_name + '.xml')
        if not os.path.exists(full_label_path):
            continue

        full_image_path = os.path.join(image_path, f)
        minimap = np.array(Image.open(full_image_path))
        h, w, c = minimap.shape

        root = ET.parse(full_label_path).getroot()

        # parse expected champion locations
        expected_champion_boxes = {}
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            box = CocoBox(name, xmin, ymin, xmax, ymax)
            expected_champion_boxes[name] = box

        orig_minimap = minimap
        minimap = minimap.copy().transpose((2, 0, 1)).astype(np.float)
        minimap /= 255

        img = torch.from_numpy(minimap)
        img = img.type(torch.float32)

        extra_champion_boxes = {}
        with torch.no_grad():
            img = img.to(device)
            predictions = model([img])
            img = img.cpu().numpy()
            expected_img = img.copy()

            for prediction in predictions:
                boxes = prediction['boxes'].cpu().numpy().tolist()
                labels = prediction['labels'].cpu().numpy().tolist()
                scores = prediction['scores'].cpu().numpy().tolist()
                output = ''

                img = img.transpose((1, 2, 0))
                for label, box, score in zip(labels, boxes, scores):
                    x1, y1, x2, y2 = box
                    if score > score_threshold:
                        actual_box = CocoBox(label, x1, y1, x2, y2, score)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        label = image_drawer.id_to_champion[label]

                        expected_champion_box = expected_champion_boxes.get(label)
                        if expected_champion_box:
                            name = expected_champion_box.name
                            xmin = expected_champion_box.xmin
                            ymin = expected_champion_box.ymin
                            xmax = expected_champion_box.xmax
                            ymax = expected_champion_box.ymax

                            img = img.copy()

                            text = '{} {:.2f}'.format(label, score)
                            output += text + '\n'
                            print(text)

                            color = champion_to_color[label]
                            img = draw_box(img, label, x1, y1, x2, y2, color)
                        else:
                            print("{} Champion {} not in champion boxes ({}) ({})".format(full_image_path, label, score, box))

                img = img.transpose((2, 0, 1))

            img = torch.from_numpy(img)
            pil_img = transforms.ToPILImage()(img)
            pil_img.save(os.path.join('labeled_images', f))

            expected_img = expected_img.transpose((1, 2, 0))
            for label, box in expected_champion_boxes.items():
                x1 = int(box.xmin)
                y1 = int(box.ymin)
                x2 = int(box.xmax)
                y2 = int(box.ymax)
                
                expected_img = expected_img.copy()
                
                color = champion_to_color[label]
                expected_img = draw_box(expected_img, label, x1, y1, x2, y2, color)

            expected_img = expected_img.transpose((2, 0, 1))
            expected_img = torch.from_numpy(expected_img)
            expected_img = transforms.ToPILImage()(expected_img)

            ax1.imshow(orig_minimap, cmap='gray', interpolation='none')
            ax2.imshow(expected_img, cmap='gray', interpolation='none')
            ax3.imshow(pil_img, cmap='gray', interpolation='none')

            fig.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)
            for i, ax in enumerate(fig.axes):
                ax.clear()

if __name__ == "__main__":
    main()
