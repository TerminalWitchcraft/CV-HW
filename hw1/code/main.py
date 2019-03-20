#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : b1.py
# Author            : Hitesh Paul <hp1293@gmail.com>
# Date              : 24.02.2019
# Last Modified Date: 28.02.2019
# Last Modified By  : Hitesh Paul <hp1293@gmail.com>

import os
import random
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio
from collections import defaultdict, Counter, OrderedDict
from PIL import Image

# Load the image
COLORMAP = {
        -1: "rgb(99,99,99)",
        0: "rgb(215,48,39)",
        1: "rgb(102,189,99)",
        2: "rgb(43,130,189)"
        }
NAMEMAP = {
        -1: "Gray",
        0: "Red",
        1: "Green",
        2: "Blue"
        }

def get_info(im):
    # Show the bands contained in the image
    print("The image has following bands: ", im.getbands())
    # Seperate out the bands
    # print out information about the bands
    print(im.getextrema())
    print("The width of the image is: ", im.width)
    print("The height of the image is: ", im.height)

def plot(data, filename, title, titlex, titley, modes,
        auto_open=True, gray=False, save=False):
    """
    Function to plot data. data is an array of (x,y)
    """
    if not os.path.exists("html"):
        os.mkdir("html")
    traces = []
    for i in range(len(modes)):
        color = COLORMAP[-1] if gray else COLORMAP[modes[i]]
        name = NAMEMAP[-1] if gray else NAMEMAP[modes[i]]
        trace = go.Bar(x=data[modes[i]][0], y=data[modes[i]][1],
                marker={"line": {"color": color}, "color": color},
                name=name)
        traces.append(trace)

    layout = go.Layout(title=title,
            xaxis=dict(title=titlex),
            yaxis=dict(title=titley))
    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, auto_open=auto_open, filename="html/" + filename+".html", image_filename="images/" + filename)
    if save:
        if not os.path.exists("charts"):
            os.mkdir("charts")
        pio.write_image(fig, 'charts/' + filename + ".jpeg", width=1366, height=768, scale=2)


def histogram(im, mode=[0], denom=1, cummulate=False):
    """
    Returns the histogram for the given image and mode
    0 -> Red, 1-> Green, 2->Blue
    """
    ret_data = []
    for key in mode:
        band = np.array(im.getdata(key))
        data = defaultdict(int)
        for item in band:
            data[item] += 1
        x = []
        y = []
        for i in range(256):
            x.append(i)
            if cummulate:
                cummulative_sum = 0
                for j in range(i+1):
                    cummulative_sum += data[j] / denom
                y.append(cummulative_sum)
            else:
                y.append(data[i] / denom)
        ret_data.append((x,y))
    return ret_data

def normalize(grey_im, mode=[0], gray=True):
    """
    Normalize the histogram according to Leibnitz rule
    """
    arr = np.array(grey_im)
    denom = grey_im.width * grey_im.height
    data = histogram(grey_im, mode=mode, denom=1, cummulate=True)
    y = data[0][1]


    # First method
    vf = np.vectorize(lambda x: ((y[x] - np.nanmin(arr) ) / (denom - np.nanmin(arr))) * np.amax(arr) )
    # ff = np.vectorize(lambda x: np.nanmax(arr) * y[x])
    norm_im = vf(arr)
    norm_im = Image.fromarray(norm_im.astype(np.uint8))
    # norm_im.show()
    return norm_im


def pdf(im, mode=[0]):
    """
    Function to calculate pdf of the given image
    Returns histogram / (width * height)
    """
    denom = im.width * im.height
    return histogram(im, mode, denom=denom)

def cdf(im, mode=[0]):
    """
    Function to calculate cumulative distribution function from histogram
    """
    denom = im.width * im.height
    return histogram(im, mode, denom=denom, cummulate=True)

def grayscale(img):
    """
    Converts the given color image to grayscale
    """
    im2arr = np.array(img)
    r_band = im2arr[:,:,0]
    g_band = im2arr[:,:,1]
    b_band = im2arr[:,:,2]
    l = 0.3 * r_band + 0.59 * g_band + 0.11 * b_band
    grey_im = Image.fromarray(l.astype(np.uint8))
    # grey_im.show()
    return grey_im
    

def main(filename):
    if not os.path.exists("images"):
        os.mkdir("images")
    im = Image.open(filename)
    get_info(im)
    filename = filename[:len(filename)-4]
    save = True

    # Part 1 of assignment
    hist_data = histogram(im, [0,1,2])
    plot(hist_data, filename=filename+"_hist_r", title="Plot of distribution for Red",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0], auto_open=False, save=save)

    plot(hist_data, filename=filename+"_hist_g", title="Plot of distribution for Green",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[1], auto_open=False, save=save)

    plot(hist_data, filename=filename+"_hist_b", title="Plot of distribution for Blue",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[2], auto_open=False, save=save)

    plot(hist_data, filename=filename+"_hist", title="Plot of distribution for channels[Click on the legend to isolate traces]",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0,1,2], auto_open=False, save=save)

    # Next, create Grayscale images
    grey_im = grayscale(im)
    grey_im.save("images/" + filename + "_grey.jpeg")
    hist_data_gray = histogram(grey_im, [0])
    plot(hist_data_gray, filename=filename+"_hist_gray", title="Plot of distribution for Gray channel",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0], auto_open=False, save=save, gray=True)

    # Next, plot the pdf
    pdf_grey = pdf(grey_im, [0])
    plot(pdf_grey, filename=filename+"_pdf_grey", title="Plot of probability distribution function",
            titlex="Intensity Values (0-255)",
            titley="Probability",
            modes=[0], auto_open=False, save=save, gray=True)

    # Plot the cdf
    cdf_grey = cdf(grey_im, [0])
    plot(cdf_grey, filename=filename+"_cdf_grey", title="Plot of Cummulative distribution function",
            titlex="Intensity Values (0-255)",
            titley="Cummulative",
            modes=[0], auto_open=False, save=save, gray=True)

    # Plot the normalized histogram
    norm_im = normalize(grey_im)
    norm_im = normalize(norm_im)
    norm_im.save("images/" + filename + "_norm.jpeg")
    hist_grey_norm = histogram(norm_im)
    plot(hist_grey_norm, filename=filename+"_norm_hist_gray", title="Plot of Normalized histogram",
            titlex="Intensity Values (0-255)",
            titley="Cummulative",
            modes=[0], auto_open=False, save=save, gray=True)
    cdf_grey_norm = cdf(norm_im, [0])
    plot(cdf_grey_norm, filename=filename+"_norm_cdf_grey", title="Plot of Cummulative distribution function",
            titlex="Intensity Values (0-255)",
            titley="Cummulative",
            modes=[0], auto_open=False, save=save, gray=True)


def manual_threshold(im, threshold):
    """
    Manually set the threshold and return a new image
    """
    arr = np.array(im)
    arr[arr > threshold] = 255
    arr[arr <= threshold] = 0
    return Image.fromarray(arr)

    
def threshold(filenames, thresholds):
    """
    Second part of assignment
    """
    for filename in filenames:
        for item in thresholds:
            im = Image.open(filename)
            imt = manual_threshold(im, item)
            imt.save("images/" + filename[:len(filename) -4 ] + "_t_{}.png".format(str(item)))

def otsu(filenames):
    """
    Second part of the assignment. Otsu's automatic threshold
    detection algorithm
    """
    for filename in filenames:
        im = Image.open(filename)
        arr = np.array(im)
        arr += 1
        hist_data = pdf(im)

        hist_temp = {}
        for key, value in zip(hist_data[0][0], hist_data[0][1]):
            # print(key, value)
            if value > 0:
                hist_temp[key] = value

        optim_k = 0
        max_b = 0.0
        x = []
        y = []
        for k in range(0, 256):
            c0 = [x for x in range(k+1) if x in hist_temp]
            c1 = [x for x in range(k+1, 256) if x in hist_temp]
            if not c0 or not c1: continue
            omega0 = omegak= sum([hist_temp[x] for x in c0])
            omega1 =  sum([hist_temp[x] for x in c1])

            u0 = sum([i * hist_temp[i] for i in c0]) / omega0
            u1 = sum([i * hist_temp[i] for i in c1]) / omega1
            ut = sum([i*hist_temp[i] for i in hist_temp]) 
            var = (omega0 * omega1) * ((u0-u1) * (u0-u1))
            x.append(k)
            y.append(var)
            if var > max_b:
                max_b = var
                optim_k = k
        print("The best k for {} is: ".format(filename), optim_k)
        print("The max variance  for {} is: ".format(filename), max_b)
        im1 = manual_threshold(im, optim_k)
        im1.save("images/" + filename[: len(filename) - 4] + "_otsu.jpeg")

        hist_data_gray = histogram(im1, [0])
        plot(hist_data_gray, filename=filename[: len(filename) - 4]+"_hist_gray", title="Plot of distribution for Gray channel",
                titlex="Intensity Values (0-255)",
                titley="Number of Pixels",
                modes=[0], auto_open=False, save=True, gray=True)
        plot([(x,y)], filename=filename[: len(filename) - 4] + "_otsu", title="Plot of variance with respect to intensity level",
                titlex="Intensity Values (0-255)",
                titley="Variance",
                modes=[0], auto_open=False, save=True, gray=True)

def prepare():
    """Prepare directories"""
    import shutil
    for item in ["images", "charts", "html"]:
        try:
            shutil.rmtree(item)
        except:
            pass


if __name__ == "__main__":
    prepare()
    files = ["b2_a.png", "b2_b.png", "b2_c.png"]
    main("b1.png")
    threshold(files, [80, 125, 230])
    otsu(files)
