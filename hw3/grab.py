from PIL import ImageDraw
from matplotlib import pyplot as plt

class Grab(object):
    def __init__(self, img):
        self.img = img
        self.points = []
        self.plt = plt
        self.radius = 5
        self.exited = False
        self.plt.xticks([])
        self.plt.yticks([])

    def onClick(self, event):
        if event.xdata != None and event.ydata != None:
            self.validate_and_push((event.xdata, event.ydata))

    def onKeyPress(self, event):
        self.plt.close()

    def validate_and_push(self, data):
        if len(self.points) < 3:
            print("Coordinates of selected point: ", data)
            self.points.append(data)
            if len(self.points) == 3:
                self.plt.close()
                self.exited = True

        elif len(self.points) == 3:
            self.plt.close()
            self.exited = True
        else:
            self.plt.close()
            self.exited = True


    def get_radius(self, point):
        x_final = y_final = 0.
        diff_x = diff_y = 0.
        x, y = point

        if x - self.radius < 0:
            diff_x = x - 0.
        if y - self.radius < 0:
            diff_y = y - 0.

        diff = max(diff_x, diff_y)
        if diff == 0.:
            return (x - self.radius, y - self.radius,
                    x + self.radius, y + self.radius)
        else:
            min_diff = min(diff_x, diff_y)
            return (x - min_diff, y - min_diff, 
                    x + min_diff, y + min_diff)

    def get_output(self, fill=False):
        draw = ImageDraw.Draw(self.img)
        for point in self.points:
            draw.ellipse(self.get_radius(point), fill=(255, 0, 0))

    def get_points(self):
        return self.points

    def show(self):
        self.plt.imshow(self.img)
        self.plt.show()

    def run(self):
        self.plt.imshow(self.img)
        fig = self.plt.gcf()
        fig.canvas.set_window_title("Select three points using mouse")

        cid = fig.canvas.mpl_connect('button_press_event', self.onClick)
        cid2 = fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.plt.show(block=False)
        try:
            print("The window will be open for 60 seconds or until you select 3 points")
            self.plt.pause(60)
        except Exception:
            pass
        if self.exited:
            self.get_output()
            self.show()
        else:
            raise AssertionError("The image did not exit successfully")

