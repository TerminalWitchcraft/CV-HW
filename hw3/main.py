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

    def shearX(self, x):
        a = np.array([
            [1, x, 0],
            [0, 1, 0],
            [0, 0, 1]
            ], dtype=np.float32)
        self.point = np.matmul(self.get_multiplier(a), self.point)
        return self

    def shearY(self, y):
        a = np.array([
            [1, 0, 0],
            [y, 1, 0],
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
    if float(x_high) - x == 0. and int(x_high + 1) != img_arr.shape[0]:
        x_high = x_high + 1

    y_low = math.floor(y)
    y_high = math.ceil(y)
    if float(y_high) - y == 0. and int(y_high + 1) != img_arr.shape[1]:
        y_high = y_high + 1

    int_r = 0.
    int_g = 0.
    int_b = 0.

    final = []
    first = np.array([x_high - x, x-x_low], dtype=np.float32).reshape((1,2))
    third = np.array([y_high -y, y-y_low], dtype=np.float32).reshape((2,1))
    for i in range(3):
        second = np.array([
            [img_arr[x_low, y_low, i], img_arr[x_low, y_high, i]],
            [img_arr[x_high, y_low, i], img_arr[x_high, y_high, i]],
            ], dtype=np.float32).reshape((2,2))
        temp = np.matmul(first, second)
        temp = np.matmul(temp, third)
        final.append(float(temp[0]))
    return final

def partA():
    img = Image.open("./daoko.jpg")
    img_arr = np.array(img)
    new_arr = np.zeros_like(img_arr)
    img.show()
    for index in np.ndindex(img_arr.shape[:2]):
        a = Transform.init(index, inverse=True).shearY(-0.5).get_point()
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
    # img.show()
    for index in np.ndindex(img_arr.shape[:2]):
        x, y = Transform.init(index, inverse=True).shearY(0.3).get_orig()
        if x >= 0. and y >= 0.:
            if x - math.floor(x) == 0. and y - math.floor(y) == 0.:
                new_arr[index] = img_arr[int(x),int(y)]
            else:
                intensity = bilinear(x, y, img_arr)
                try:
                    new_arr[index] = intensity
                except:
                    pass
    img2 = Image.fromarray(new_arr)
    img2.show()

def calculate_affine(points1, points2):
    X = np.array([
        [points1[0][0], points1[0][1], 1, 0, 0, 0],
        [0, 0, 0, points1[0][0], points1[0][1], 1],
        [points1[1][0], points1[1][1], 1, 0, 0, 0],
        [0, 0, 0, points1[1][0], points1[1][1], 1],
        [points1[2][0], points1[2][1], 1, 0, 0, 0],
        [0, 0, 0, points1[2][0], points1[2][1], 1]
        ], dtype=np.float32).reshape((6,6))
    second = np.array([
        points2[0][0],
        points2[0][1],
        points2[1][0],
        points2[1][1],
        points2[2][0],
        points2[2][1]
        ], dtype=np.float32).reshape((6,1))
    a = np.matmul(np.linalg.inv(X), second)
    return a

def calculate_affine_unconstrained(points1, points2):
    X = np.array([
        [points1[0][0], points1[0][1], 1, 0, 0, 0],
        [0, 0, 0, points1[0][0], points1[0][1], 1],
        [points1[1][0], points1[1][1], 1, 0, 0, 0],
        [0, 0, 0, points1[1][0], points1[1][1], 1],
        [points1[2][0], points1[2][1], 1, 0, 0, 0],
        [0, 0, 0, points1[2][0], points1[2][1], 1]
        ], dtype=np.float32).reshape((6,6))
    for i in range(3, len(points1)):
        temp = np.array([
            [points1[i][0], points1[i][1], 1, 0, 0, 0],
            [0, 0, 0, points1[i][0], points1[i][1], 1],
            ])
        X = np.vstack((X, temp))

    # print(X)
    second = np.array([
        points2[0][0],
        points2[0][1],
        points2[1][0],
        points2[1][1],
        points2[2][0],
        points2[2][1]
        ], dtype=np.float32).reshape((6,1))
    for j in range(3, len(points2)):
        second = np.vstack((second, np.array([points2[j][0]]).reshape((1,1)) ))
        second = np.vstack((second, np.array([points2[j][1]]).reshape((1,1)) ))
    # print("scon", second)
    # a = np.matmul(np.linalg.inv(X), second)
    a = np.linalg.lstsq(X, second)
    return a[0]

def partB():
    img = Image.open("./pic1.jpg")
    img_copy = img.copy()
    g = Grab(img_copy)
    g.run()
    points1 = g.get_points()

    img = Image.open("./pic2.jpg")
    img_copy = img.copy()
    g = Grab(img_copy)
    g.run()
    points2 = g.get_points()
    a = calculate_affine(points1, points2)
    a = np.array([
        [a[0], a[1], a[2]],
        [a[3], a[4], a[5]],
        [0, 0, 1]
        ], dtype=np.float32).reshape((3,3))
    a = np.linalg.inv(a)

    img1 = Image.open("./pic1.jpg") 
    img1 = np.array(img1)
    new_arr = np.zeros_like(img1)
    for index in np.ndindex(img1.shape[:2]):
        b = np.matmul(a, np.array([index[0], index[1], 1], dtype=np.float32).reshape((3,1)))
        if b[0][0] >= 0. and b[1][0] >= 0.:
            try:
                new_arr[index] = img1[int(b[0][0]), int(b[1][0])] 
            except:
                pass
    final_img = Image.fromarray(new_arr)
    final_img.show()

def partB2():
    num_points = 5
    img = Image.open("./pic1.jpg")
    img_copy = img.copy()
    g = Grab(img_copy, limit=num_points)
    g.run()
    points1 = g.get_points()

    img = Image.open("./pic2.jpg")
    img_copy = img.copy()
    g = Grab(img_copy, limit=num_points)
    g.run()
    points2 = g.get_points()
    a = calculate_affine_unconstrained(points1, points2)
    a = np.array([
        [a[0], a[1], a[2]],
        [a[3], a[4], a[5]],
        [0, 0, 1]
        ], dtype=np.float32).reshape((3,3))
    a = np.linalg.inv(a)

    img1 = Image.open("./pic1.jpg") 
    img1 = np.array(img1)
    new_arr = np.zeros_like(img1)
    for index in np.ndindex(img1.shape[:2]):
        b = np.matmul(a, np.array([index[0], index[1], 1], dtype=np.float32).reshape((3,1)))
        if b[0][0] >= 0. and b[1][0] >= 0.:
            try:
                new_arr[index] = img1[int(b[0][0]), int(b[1][0])] 
            except:
                pass
    final_img = Image.fromarray(new_arr)
    final_img.show()

def main():
    partB2()

if __name__ == "__main__":
    main()
