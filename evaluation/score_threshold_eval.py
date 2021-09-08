import os
import sys
sys.path.insert(0, '..')
import argparse
import time

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

    score_threshold = 0.0
    iou_threshold = 0.5

    true_positives_arr = []
    false_positives_arr = []
    false_negatives_arr = []
    extra_champions_detected_arr = []
    precisions = []
    recalls = []
    accuracy = []
    accuracy_box = []
    while score_threshold < 1.0:
        box_threshold = 5
        total_images = 0
        total_champions_in_images = 0
        true_positives = 0
        total_champions_detected_in_boxes = 0
        total_champions_not_detected = 0

        false_positives = 0
        false_negatives = 0
        extra_champions_detected = 0

        for f in os.listdir(image_path):
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

            total_champions_in_images += len(expected_champion_boxes)

            orig_minimap = minimap
            minimap = minimap.copy().transpose((2, 0, 1)).astype(np.float)
            minimap /= 255

            img = torch.from_numpy(minimap)
            img = img.type(torch.float32)

            extra_champion_boxes = {}
            with torch.no_grad():
                img = img.to(device)
                predictions = model([img])

                for prediction in predictions:
                    boxes = prediction['boxes'].cpu().numpy().tolist()
                    labels = prediction['labels'].cpu().numpy().tolist()
                    scores = prediction['scores'].cpu().numpy().tolist()
                    output = ''

                    for label, box, score in zip(labels, boxes, scores):
                        if score > score_threshold:
                            x1, y1, x2, y2 = box
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

                                if expected_champion_box.in_range(box_threshold, actual_box):
                                    total_champions_detected_in_boxes += 1

                                if expected_champion_box.calculate_iou(actual_box) > iou_threshold:
                                    true_positives += 1
                                else:
                                    false_positives += 1
                                
                                del expected_champion_boxes[label]
                            else:
                                #print("{} Champion {} not in champion boxes".format(full_image_path, label))
                                extra_champion_boxes[label] = actual_box

            false_negatives += len(expected_champion_boxes)
            extra_champions_detected += len(extra_champion_boxes)
            total_images += 1

        total_champions_not_detected = false_positives + false_negatives

        print('Total images: {}'.format(total_images))
        print('Total champions detected images: {}'.format(total_champions_in_images))
        print('Total champions detected: {}'.format(true_positives))
        print('Total champions detected in boxes: {}'.format(total_champions_detected_in_boxes))
        print('Total champions not detected: {}'.format(total_champions_not_detected))

        print('False positives (detecting object when it is not there): {}'.format(false_positives))
        print('False negatives (not detecting object when it is there): {}'.format(false_negatives))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print('Precision (how accurate predictions are): {}'.format(precision))
        print('Recall (how well at detecting objects): {}'.format(recall))

        print('Total champions detected percentage: {}'.format(true_positives / total_champions_in_images))
        print('Total champions detected percentage in {} pixel radius: {}'.format(box_threshold, total_champions_detected_in_boxes / total_champions_in_images))
        print('Total champions not detected percentage: {}'.format(total_champions_not_detected / total_champions_in_images))

        true_positives_arr.append(true_positives)
        false_positives_arr.append(false_positives)
        false_negatives_arr.append(false_negatives)
        extra_champions_detected_arr.append(extra_champions_detected)
        
        precisions.append(precision)
        recalls.append(recall)
        accuracy.append(true_positives / total_champions_in_images)
        accuracy_box.append(total_champions_detected_in_boxes / total_champions_in_images)
        score_threshold += 0.05

    print(true_positives_arr)
    print(false_positives_arr)
    print(false_negatives_arr)
    print(extra_champions_detected_arr)

    print(precisions)
    print(recalls)
    print(accuracy)
    print(accuracy_box)

if __name__ == "__main__":
    main()
