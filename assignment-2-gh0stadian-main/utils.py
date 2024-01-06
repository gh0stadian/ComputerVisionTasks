import cv2
import numpy as np
import matplotlib
from math import ceil
from matplotlib import pyplot as plt
import os

max_values_for_color = {
    "hsv": {
        "histSize": {
            "h": [180],
            "s": [256],
            "v": [256]
        },
        "ranges": {
            "h": [0, 180],
            "s": [0, 256],
            "v": [0, 256]
        },
    },
    "lab": {
        "histSize": {
            "l": [255],
            "a": [255],
            "b": [255]
        },
        "ranges": {
            "l": [1, 256],
            "a": [1, 256],
            "b": [1, 256]
        },
    },
    "hls": {
        "histSize": {
            "h": [180],
            "l": [256],
            "s": [256]
        },
        "ranges": {
            "h": [0, 180],
            "l": [0, 256],
            "s": [0, 256]
        },
    },
}


def show_histogram(bgr_img, models):
    base = ceil(len(models) / 2) * 100 + 21
    figure = plt.figure(figsize=(20, 3 * ceil(len(models) / 2)))
    colors = ['b', 'g', 'r']
    for i, model in enumerate(models):
        ax = figure.add_subplot(base + i)
        if model == "greyscale":
            ax.set_title("greyscale")
            img_grey = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
            ax.plot(hist, color='k')

        elif model == "rgb":
            ax.set_title("rgb")
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([bgr_img], [i], None, [256], [0, 256])
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "hsv":
            ax.set_title("hsv")
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            for i, col in enumerate(['h', 's', 'v']):
                hist = cv2.calcHist([hsv_img], [i], None,
                                    max_values_for_color['hsv']['histSize'][col],
                                    max_values_for_color['hsv']['ranges'][col]
                                    )
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "lab":
            ax.set_title("lab")
            lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
            for i, col in enumerate(['l', 'a', 'b']):
                hist = cv2.calcHist([lab_img], [i], None,
                                    max_values_for_color['lab']['histSize'][col],
                                    max_values_for_color['lab']['ranges'][col]
                                    )
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "ycrcb":
            ax.set_title("ycrcb")
            ycrcb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
            for i, col in enumerate(['y', 'cr', 'cb']):
                hist = cv2.calcHist([ycrcb_img], [i], None, [256], [0, 256])
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "xyz":
            ax.set_title("xyz")
            xyz_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2XYZ)
            for i, col in enumerate(['x', 'y', 'z']):
                hist = cv2.calcHist([xyz_img], [i], None, [256], [0, 256])
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "luv":
            ax.set_title("luv")
            luv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Luv)
            for i, col in enumerate(['l', 'u', 'v']):
                hist = cv2.calcHist([luv_img], [i], None, [256], [0, 256])
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")

        elif model == "hls":
            ax.set_title("hls")
            hls_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
            for i, col in enumerate(['h', 'l', 's']):
                hist = cv2.calcHist([hls_img], [i], None,
                                    max_values_for_color['hls']['histSize'][col],
                                    max_values_for_color['hls']['ranges'][col]
                                    )
                ax.plot(hist, color=colors[i], label=col)
                ax.legend(loc="upper right")
        else:
            print("Wrong model name")


def equalize_comparision(images, labels, color_to_plot_conversion):
    fig, axes = plt.subplots(nrows=len(images), ncols=4, figsize=(20, len(images) * 3))
    labels.insert(0, "base")
    for i, (row, image) in enumerate(zip(axes, images)):
        for j, (col, label) in enumerate(zip(row, labels)):
            if i == 0:
                col.set_title(label)

            if j == 0:
                col.imshow(cv2.cvtColor(image, color_to_plot_conversion))
            else:
                img_to_eq = image.copy()
                img_to_eq[:, :, j - 1] = cv2.equalizeHist(img_to_eq[:, :, j - 1])
                col.imshow(cv2.cvtColor(img_to_eq, color_to_plot_conversion))


def equalize_comparision_separate_channels(images, labels, color_to_plot_conversion):
    fig, axes = plt.subplots(nrows=len(images), ncols=7, figsize=(20, len(images) * 3))
    labels.insert(0, "base")
    for i, (row, image) in enumerate(zip(axes, images)):
        for j, label in enumerate(labels):
            if j == 0:
                row[0].imshow(cv2.cvtColor(image, color_to_plot_conversion))
                continue

            if i == 0:
                row[(j*2)-1].set_title(label)
                row[(j*2)].set_title(label+"_eq")

            img_to_eq = image.copy()
            row[(j*2)-1].imshow(img_to_eq[:, :, j - 1], cmap="gray")

            img_to_eq[:, :, j - 1] = cv2.equalizeHist(img_to_eq[:, :, j - 1])
            row[(j*2)].imshow(img_to_eq[:, :, j - 1], cmap="gray")


def show_best_equalizations(images):
    fig, axes = plt.subplots(nrows=len(images), ncols=4, figsize=(20, len(images) * 3))
    labels = ["base", "hsv_v", "ycrcb_y", "lab_l"]
    for i, (row, image) in enumerate(zip(axes, images)):
        if i == 0:
            for j, col in enumerate(row):
                col.set_title(labels[j])

        row[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])
        row[1].imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))

        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image_ycrcb[:, :, 0] = cv2.equalizeHist(image_ycrcb[:, :, 0])
        row[2].imshow(cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB))

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image_lab[:, :, 0] = cv2.equalizeHist(image_lab[:, :, 0])
        row[3].imshow(cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB))
