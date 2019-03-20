import sys
import numpy as np
from PIL import Image


def box_filter(img, val, k=3):
    """Box filter the image"""

    # Construct the filter with normalization
    mask = np.full((k,k), fill_value=val / (k*k*val), dtype=np.float64)
    print("The sum of the mask is: ", np.sum(mask))

    # Validate image dimensions
    if k > img.width or k > img.height:
        print("Invalid image dimensions...Exiting")
        sys.exit(0)

    # run the convolution
    img_arr = np.array(img)
    print(img_arr.shape)
    new_arr = []
    for j in range(img.height - k + 1):
        temp_data = []
        for i in range(img.width - k + 1):
            temp_data.append(np.sum(np.multiply(mask, img_arr[j:j+k, i: i+k])))
        new_arr.append(temp_data)
    print(len(new_arr))
    new_arr = np.array(new_arr)
    print(new_arr.shape)



def partA1():
    img = Image.open("./b2_a.png")
    box_filter(img, 9, k=7)

if __name__ == "__main__":
    partA1()
