import cv2
import math
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np

from grab import Grab

class Point(object):
    def __init__(self):
        pass

    def __call__(self, point): 
        x, y = point
        return np.array([x, y, 1]).reshape(3,1)

class Transform(object):

    def __init__(self, point, inverse=False):
        p = Point()
        self.point = p(point)
        self.inverse = inverse
        self.get_multiplier = lambda x: np.linalg.inv(x) if self.inverse else x

    @classmethod
    def init(cls, point, inverse):
        return cls(point, inverse)

    def inverse_on(self):
        self.inverse = True
        return self

    def translate(self, x, y):
        a = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
            ], dtype=np.float32) if not self.inverse else \
        np.array([
            [1, 0, y],
            [0, 1, x],
            [0, 0, 1]
            ], dtype=np.float32)
        # print(self.get_multiplier(a))
        self.point = np.matmul(self.get_multiplier(a), self.point)
        return self

    def rotate(self, theta):
        rads = theta * np.pi/180
        a = np.array([
            [np.cos(rads), np.sin(rads), 0],
            [-np.sin(rads), np.cos(rads), 0],
            [0, 0, 1]
            ], dtype=np.float32)
        self.point = np.matmul(self.get_multiplier(a), self.point)
        return self

    def scale(self, x, y):
        a = np.array([
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1]
            ], dtype=np.float32)
        self.point = np.matmul(self.get_multiplier(a), self.point)
        return self

    def get(self):
        return tuple(self.point[:2])

    def get_lowhigh(self):
        x_low = math.floor(self.point[0])
        x_high = math.ceil(self.point[0])

        y_low = math.floor(self.point[1])
        y_high = math.ceil(self.point[1])

        print(x_low, x_high, y_low, y_high)

    def get_orig(self):
        return self.point[0][0], self.point[1][0]

    def get_point(self):
        return int(self.point[0]), int(self.point[1])

    def show(self):
        print("\n")
        print(self.point)

def bilinear(x, y, img_arr):
    x_low = math.floor(x)
    x_high = math.ceil(x)

    y_low = math.floor(y)
    y_high = math.ceil(y)

    int_r = 0.
    int_g = 0.
    int_b = 0.

    final = []
    first = np.array([x_high - x, x-x_low], dtype=np.float32).reshape((1,2))
    third = np.array([y_high -y, y-y_low], dtype=np.float32).reshape((2,1))
    print("\nThis is point: ", x, y)
    for i in range(3):
        second = np.array([
            [img_arr[x_low, y_low, i], img_arr[x_low, y_high, i]],
            [img_arr[x_high, y_low, i], img_arr[x_high, y_high, i]],
            ], dtype=np.float32).reshape((2,2))
        temp = np.matmul(first, second)
        print("First: ", first)
        print("Second: ", second)
        print("Result: ", temp)
        temp = np.matmul(temp, third)
        final.append(float(temp[0]))
    return final

def partA():
    img = Image.open("./daoko.jpg")
    img_arr = np.array(img)
    new_arr = np.zeros_like(img_arr)
    img.show()
    for index in np.ndindex(img_arr.shape[:2]):
        a = Transform.init(index, inverse=True).translate(20, 10).rotate(45).get_point()
        x, y = a
        if x >= 0. and y >= 0.:
            try:
                new_arr[index] = img_arr[a]
            except:
                pass
    img2 = Image.fromarray(new_arr)
    img2.show()



def partA2():
    img = Image.open("./daoko.jpg")
    img_arr = np.array(img)
    new_arr = np.zeros_like(img_arr)
    img.show()
    for index in np.ndindex(img_arr.shape[:2]):
        x, y = Transform.init(index, inverse=True).translate(20, 10).get_orig()
        if x >= 0. and y >= 0.:
            if x - math.floor(x) == 0. and y - math.floor(y) == 0.:
                new_arr[index] = img_arr[int(x),int(y)]
            else:
                intensity = bilinear(x, y, img_arr)
                # print(intensity)
                try:
                    new_arr[index] = intensity
                except:
                    pass
    img2 = Image.fromarray(new_arr)
    img2.show()

def partB():
    img = Image.open("./daoko.jpg")
    img_copy = img.copy()
    g = Grab(img_copy)
    g.run()

def main():
    partA2()

if __name__ == "__main__":
    main()
