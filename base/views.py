from django.shortcuts import render
import cv2
from sklearn.cluster import KMeans
import scipy as sp
import numpy as np
# # import porespy as ps
import matplotlib.pyplot as plt
from pathlib import Path
import base64

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# import pandas as pd
# Create your views here.
# colours = []
# Color_percent = []


def home(request):
    context = {}
    # colours.clear()
    # Color_percent.clear()
    if request.method == 'POST':

        k = request.POST.get('no_colour')
        imgByte = request.FILES['bgImg'].read()
        ext = request.FILES['bgImg'].name.split('.')[-1]

        nparr = np.fromstring(imgByte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(img.shape)
        jpg_as_text = base64.b64encode(imgByte).decode('utf-8')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_feature = img_rgb.reshape(
            ((img_rgb.shape[0]*img_rgb.shape[1]), img_rgb.shape[2]))
        x = img_rgb_feature

        clf = KMeans(n_clusters=int(k))
        clf.fit(img_rgb_feature)

        hist = centroid_histogram(clf)
        bar, colours, Color_percent = plot_colors(hist, clf.cluster_centers_)

        labels = []
        sizes = []
        Colors = []
        for color_dict in Color_percent:
            Colors.append(tuple(color_dict['color']))
            labels.append(color_dict['label'])
            sizes.append(color_dict['percent'])

        context = {'k': k, 'path': BASE_DIR, 'img': jpg_as_text,
                   'Colors': Colors, 'labels': labels, 'sizes': sizes}
        return render(request, 'base/home.html', context=context)
    return render(request, 'base/home.html', context=context)


def centroid_histogram(clt):
    # get the no of different clusters and create a histogram
    # group to cluster according to the no of pixels assigned
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram,
    # so that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart
    # representing the relative frequency
    # of per colors
    startX = 0
    bar = np.zeros((50, 300, 3), dtype="uint8")
    colours = []
    Color_percent = []

    # loop over the percentage of each cluster
    # and the color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        colours.append(color.astype("uint8").tolist())
        Color_percent.append({"label": color.astype("uint8").tolist(
        ), "color": color.astype("uint8").tolist(), "percent": percent})

    return bar, colours, Color_percent
