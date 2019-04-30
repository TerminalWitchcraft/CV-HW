from PIL import Image
import numpy as np

class Point(object):
    def __init__(self):
        pass

    def __call__(self, x, y): 
        return np.array([x, y, 1]).reshape(3,1)

class Transform(object):

    def __init__(self, x, y):
        p = Point()
        self.point = p(x, y)
        self.inverse = False

    @classmethod
    def init(cls, x, y):
        return cls(x,y)

    def inverse(self):
        self.inverse = True
        return self

    def get_multiplier(self, a):
        if self.inverse:
            return np.linalg.inv(a)
        else:
            return a

    def translate(self, x, y):
        a = np.array([
            [1, 0, x],
            [0, 1, y],
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
        return self.point

    def get_point(self):
        return self.point[0], self.point[1]

    def show(self):
        print("\n")
        print(self.point)

def main():
    img = Image.open("./daoko.jpg")
    # img.show()
    img_arr = np.array(img)
    a = Transform.init(1,1).scale(2,2).translate(-3, 5)
    print(a.get().I)

if __name__ == "__main__":
    main()
