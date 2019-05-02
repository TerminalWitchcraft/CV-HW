import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

from grab import Grab

class Point(object):
    def __init__(self):
        pass

    def __call__(self, point): 
        x, y = point
        return np.array([x, y, 1]).reshape(3,1)

class Transform(object):

    def __init__(self, point):
        p = Point()
        self.point = p(point)
        self.inverse = False

    @classmethod
    def init(cls, point):
        return cls(point)

    def inverse_on(self):
        self.inverse = True
        return self

    def get_multiplier(self, a):
        if self.inverse:
            return np.linalg.inv(a)
        else:
            return a

    def translate(self, x, y):
        a = np.array([
            [1, 0, y],
            [0, 1, x],
            [0, 0, 1]
            ], dtype=np.float32)
        self.point = np.matmul(self.get_multiplier(a), self.point)
        return self

    def rotate(self, theta):
        rads = theta * np.pi/180
        a = np.array([
            [np.cos(rads), -np.sin(rads), 0],
            [np.sin(rads), np.cos(rads), 0],
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

    def get_point(self):
        return int(self.point[0]), int(self.point[1])

    def show(self):
        print("\n")
        print(self.point)

def partA():
    img = Image.open("./daoko.jpg")
    img_arr = np.array(img)
    new_arr = np.zeros_like(img_arr)
    img.show()
    for index in np.ndindex(img_arr.shape[:2]):
        a = Transform.init(index).translate(50,100).get_point()
        # print(img_arr[index])
        # print(new_arr[index])
        try:
            new_arr[a] = img_arr[index]
        except:
            pass
        # print(new_arr[a])
        pass
    img2 = Image.fromarray(new_arr)
    img2.show()

def partB():
    img = Image.open("./daoko.jpg")
    img_copy = img.copy()
    g = Grab(img_copy)
    g.run()

def main():
    partA()

if __name__ == "__main__":
    main()
