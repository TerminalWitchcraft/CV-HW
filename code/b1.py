#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : b1.py
# Author            : Hitesh Paul <hp1293@gmail.com>
# Date              : 24.02.2019
# Last Modified Date: 24.02.2019
# Last Modified By  : Hitesh Paul <hp1293@gmail.com>

import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
from collections import defaultdict, Counter, OrderedDict
from PIL import Image

# Load the image
im = Image.open("../b1.png")
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

def histogram(im, mode=[0], gray=False):
    """
    Returns the histogram for the given image and mode
    0 -> Red, 1-> Green, 2->Blue
    """
    traces = []
    for key in mode:
        band = np.array(im.getdata(key))
        color = COLORMAP[-1] if gray else COLORMAP[key]
        name = NAMEMAP[-1] if gray else NAMEMAP[key]
        data = defaultdict(int)
        for item in band:
            data[item] += 1
        x = []
        y = []
        for i in range(256):
            x.append(i)
            y.append(data[i])
        trace = go.Bar(x=x, y=y, 
                marker={"line": {"color": color}, "color": color},
                name=name,
                )
        traces.append(trace)

    layout = go.Layout(
        title='Plot of distribution for channels[Click on the legend to isolate traces]',
        xaxis=dict(
            title='Intensity Values (0-255)',
            ),
        yaxis=dict(
            title='Number of Pixels',
            )
        )
    fig = go.Figure(data = traces, layout=layout)
    # trace = go.Histogram(x=band)
    plotly.offline.plot(fig, auto_open=True, filename="hist_main" + ".html")

def normalize(grey_im, mode=[0], gray=True):
    """
    Normalize the histogram according to Leibnitz rule
    """
    arr = np.array(grey_im)
    g = np.zeros_like(arr)
    denom = grey_im.width * grey_im.height
    traces = []
    for key in mode:
        band = np.array(im.getdata(key))
        color = COLORMAP[-1] if gray else COLORMAP[key]
        name = NAMEMAP[-1] if gray else NAMEMAP[key]
        data = defaultdict(int)
        for item in band:
            data[item] += 1
        x = []
        y = []
        for i in range(256):
            x.append(i)
            cumulative_sum = 0.0
            for j in range(i+1):
                cumulative_sum += data[j] / denom
            y.append(cumulative_sum)
    min_y = np.nanmin(arr)
    gf = np.vectorize(lambda x: x+1 if x > 0 else x)
    arr2 = gf(arr)
    vf = np.vectorize(lambda x:((y[x] - min_y) / (denom - min_y)) * np.nanmax(arr) )
    out_im = vf(arr2)
    print(out_im.shape, out_im.size, out_im)
    round_im = np.round(out_im)
    norm_im = Image.fromarray(round_im.astype(np.uint8))
    norm_im.show()
    # if save_as: grey_im.save(save_as)
    return norm_im


def pdf(im, mode=[0], gray=False):
    """
    Function to calculate pdf of the given image
    Returns histogram / (width * height)
    """
    denom = im.width * im.height
    traces = []
    for key in mode:
        band = np.array(im.getdata(key))
        color = COLORMAP[-1] if gray else COLORMAP[key]
        name = NAMEMAP[-1] if gray else NAMEMAP[key]
        data = defaultdict(int)
        for item in band:
            data[item] += 1
        x = []
        y = []
        for i in range(256):
            x.append(i)
            y.append(data[i] / denom)
        trace = go.Bar(x=x, y=y, 
                marker={"line": {"color": color}, "color": color},
                name=name,
                )
        traces.append(trace)

    layout = go.Layout(
        title='Probability distribution funciton[Click on the legend to isolate traces]',
        xaxis=dict(
            title='Intensity Values (0-255)',
            ),
        yaxis=dict(
            title='Number of Pixels',
            )
        )
    fig = go.Figure(data = traces, layout=layout)
    # trace = go.Histogram(x=band)
    plotly.offline.plot(fig, auto_open=True, filename="pdf_main" + ".html")

def cdf(im, mode=[0], gray=False):
    """
    Function to calculate cumulative distribution function from histogram
    """
    denom = im.width * im.height
    traces = []
    for key in mode:
        band = np.array(im.getdata(key))
        color = COLORMAP[-1] if gray else COLORMAP[key]
        name = NAMEMAP[-1] if gray else NAMEMAP[key]
        data = defaultdict(int)
        for item in band:
            data[item] += 1
        x = []
        y = []
        for i in range(256):
            x.append(i)
            cumulative_sum = 0.0
            for j in range(i+1):
                cumulative_sum += data[j] / denom
            y.append(cumulative_sum)
        trace = go.Bar(x=x, y=y, 
                marker={"line": {"color": color}, "color": color},
                name=name,
                )
        traces.append(trace)

    layout = go.Layout(
        title='Cumulative distribution funciton[Click on the legend to isolate traces]',
        xaxis=dict(
            title='Intensity Values (0-255)',
            ),
        yaxis=dict(
            title='Number of Pixels',
            )
        )
    fig = go.Figure(data = traces, layout=layout)
    # trace = go.Histogram(x=band)
    plotly.offline.plot(fig, auto_open=True, filename="cdf_main" + ".html")

def grayscale(img, save_as=None):
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
    if save_as: grey_im.save(save_as)
    return grey_im
    

def main(filename):
    im = Image.open(filename)
    get_info(im)
    # histogram(im, [0,1,2])
    grey_im = grayscale(im, save_as="grey.png")
    print(grey_im.getbands())
    norm_im = normalize(grey_im)
    histogram(norm_im, [0], gray=True)
    # cdf(im, mode=[0,1,2])

if __name__ == "__main__":
    main("../b1.png")
