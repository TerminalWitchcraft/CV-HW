#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : b1.py
# Author            : Hitesh Paul <hp1293@gmail.com>
# Date              : 24.02.2019
# Last Modified Date: 24.02.2019
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
    plotly.offline.plot(fig, auto_open=auto_open, filename=filename+".html")
    if save:
        if not os.path.exists("images"):
            os.mkdir("images")
        pio.write_image(fig, 'images/' + filename + ".jpeg", width=1366, height=768, scale=2)


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

def normalize(grey_im, mode=[0], gray=True, save_as=False):
    """
    Normalize the histogram according to Leibnitz rule
    """
    arr = np.array(grey_im)
    g = np.zeros_like(arr)
    denom = grey_im.width * grey_im.height
    data = histogram(grey_im, mode=mode, denom=1, cummulate=True)
    y = data[0][1]


    # First method
    vf = np.vectorize(lambda x: ((y[x] - np.nanmin(arr) ) / (denom - np.nanmin(arr))) * np.amax(arr) )
    # ff = np.vectorize(lambda x: np.nanmax(arr) * y[x])
    norm_im = vf(arr)
    norm_im = Image.fromarray(norm_im.astype(np.uint8))
    norm_im.show()
    if save_as: norm_im.save(save_as)
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
    im = Image.open(filename)
    get_info(im)

    # Part 1 of assignment
    hist_data = histogram(im, [0,1,2])
    plot(hist_data, filename=filename+"histogram_r", title="Plot of distribution for Red",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0], auto_open=False, save=True)

    plot(hist_data, filename=filename+"histogram_g", title="Plot of distribution for Green",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[1], auto_open=False, save=True)

    plot(hist_data, filename=filename+"histogram_b", title="Plot of distribution for Blue",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[2], auto_open=False, save=True)

    plot(hist_data, filename=filename+"histogram", title="Plot of distribution for channels[Click on the legend to isolate traces]",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0,1,2], auto_open=False, save=True)

    # Next, create Grayscale images
    grey_im = grayscale(im)
    hist_data_gray = histogram(grey_im, [0])
    plot(hist_data_gray, filename=filename+"histogram_gray", title="Plot of distribution for Gray channel",
            titlex="Intensity Values (0-255)",
            titley="Number of Pixels",
            modes=[0], auto_open=False, save=True, gray=True)

    # Next, plot the pdf
    pdf_grey = pdf(grey_im, [0])
    plot(pdf_grey, filename=filename+"pdf_grey", title="Plot of probability distribution function",
            titlex="Intensity Values (0-255)",
            titley="Probability",
            modes=[0], auto_open=False, save=True, gray=True)

    # Plot the cdf
    cdf_grey = cdf(grey_im, [0])
    plot(cdf_grey, filename=filename+"cdf_grey", title="Plot of Cummulative distribution function",
            titlex="Intensity Values (0-255)",
            titley="Cummulative",
            modes=[0], auto_open=False, save=True, gray=True)

    # Plot the normalized histogram
    grey_im.save("grey.jpeg")
    norm_im = normalize(grey_im, save_as="norm.jpeg")
    hist_grey_norm = histogram(grey_im)
    plot(hist_grey_norm, filename=filename+"norm_hist_gray", title="Plot of Normalized cdf",
            titlex="Intensity Values (0-255)",
            titley="Cummulative",
            modes=[0], auto_open=False, save=True, gray=True)


def manual_threshold(im, threshold):
    """
    Manually set the threshold and return a new image
    """
    arr = np.array(im)
    arr[arr > threshold] = 255
    arr[arr <= threshold] = 0
    return Image.fromarray(arr)

    
def threshold():
    """
    Second part of assignment
    """
    b2a = Image.open("../b2_a.png")
    b2b = Image.open("../b2_b.png")
    b2c = Image.open("../b2_c.png")
    im1 = manual_threshold(b2a, 125)
    im1.save("b2_at.png")
    im2 = manual_threshold(b2b, 230)
    im2.save("b2_bt.png")
    im3 = manual_threshold(b2c, 125)
    im3.save("b2_ct.png")

def otsu(filenames):
    """
    Second part of the assignment. Otsu's automatic threshold
    detection algorithm
    """
    for filename in filenames:
        im = Image.open(filename)
        arr = np.array(im)
        arr += 1
        print(get_info(im))
        print(arr.shape, arr.size)
        hist_data = pdf(im)

        # plot(hist_data, filename="b2_a", title="Plot of Histogram",
        #         titlex="Intensity Values (0-255)",
        #         titley="Number of pixels",
        #         modes=[0], auto_open=False, save=False, gray=True)
        hist_temp = {}
        for key, value in zip(hist_data[0][0], hist_data[0][1]):
            # print(key, value)
            if value > 0:
                hist_temp[key] = value

        optim_k = 0
        max_b = 0.0
        # for k in range(0, 256):
        #     # k = random.randint(0,255)
        #     # print("Randomly initialized k is: ", k)
        #     c0 = [x for x in range(k+1) if x in hist_temp]
        #     c1 = [x for x in range(k+1, 256) if x in hist_temp]
        #     if not c0 or not c1: continue
        #     # print(c0)
        #     # print(c1)
        #     omega0 = omegak= sum([hist_temp[x] for x in c0])
        #     omega1 = 1 - omega0
        #     # print(omega0)
        #     uk = sum([i*hist_temp[i] for i in c0])
        #     ut = sum([i*hist_temp[i] for i in hist_temp]) 

        #     u0 = uk / omegak
        #     u1 = (ut-uk) / 1 - omega1
        #     print("LHS: ", omega0*u0 + omega1*u1, " RHS: ", ut)
        #     # print("RHS", ut)
        #     var0 = sum([((i-u0) * (i-u0) * hist_temp[i]) / omega0 for i in c0])
        #     var1 = sum([((i-u1) * (i-u1) * hist_temp[i]) / omega1 for i in c1])

        #     varW = omega0 * var0 + omega1 * var1
        #     varB = (omega0 * omega1) * ((u0 - u1) ** 2)
        #     varB2 = ((ut * omegak - uk) ** 2) / omegak * (1-omegak)
        #     # varT = sum([((i-ut)**2) * hist_temp[i] for i in range(256)])
        x = []
        y = []
        for k in range(0, 256):
            # k = random.randint(0,255)
            # print("Randomly initialized k is: ", k)
            c0 = [x for x in range(k+1) if x in hist_temp]
            c1 = [x for x in range(k+1, 256) if x in hist_temp]
            if not c0 or not c1: continue
            # print(c0)
            # print(c1)
            omega0 = omegak= sum([hist_temp[x] for x in c0])
            omega1 =  sum([hist_temp[x] for x in c1])
            # print(omega0)
            # uk = sum([i*hist_temp[i] for i in c0])

            u0 = sum([i * hist_temp[i] for i in c0]) / omega0
            u1 = sum([i * hist_temp[i] for i in c1]) / omega1
            ut = sum([i*hist_temp[i] for i in hist_temp]) 
            var = (omega0 * omega1) * ((u0-u1) * (u0-u1))
            print("LHS: ", omega0*u0 + omega1*u1, " RHS: ", ut, " Var: ", var)
            # print("RHS", ut)
            # var0 = sum([((i-u0) * (i-u0) * hist_temp[i]) / omega0 for i in c0])
            # var1 = sum([((i-u1) * (i-u1) * hist_temp[i]) / omega1 for i in c1])

            # varW = omega0 * var0 + omega1 * var1
            # varB = (omega0 * omega1) * ((u0 - u1) ** 2)
            # varB2 = ((ut * omegak - uk) ** 2) / omegak * (1-omegak)
            # varT = sum([((i-ut)**2) * hist_temp[i] for i in range(256)])
            x.append(k)
            y.append(var)
            if var > max_b:
                print("Current Maxb: ", max_b)
                print("Greater")
                max_b = var
                print("Updated maxb", max_b)
                optim_k = k
        print("The best k is: ", optim_k)
        im1 = manual_threshold(im, optim_k)
        # im1.show()
        plot([(x,y)], filename=filename + "_otsu", title="Plot of variance with respect to intensity level",
                titlex="Intensity Values (0-255)",
                titley="Variance",
                modes=[0], auto_open=True, save=True, gray=True)
            # print(varW + varB)
            # print(varT)


if __name__ == "__main__":
    # main("../b1.png")
    # threshold()
    otsu(["../b2_a.png", "../b2_b.png", "../b2_c.png"])
