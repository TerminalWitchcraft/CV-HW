import sys
import numpy as np
from PIL import Image
from old import otsu


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
    new_arr = []
    for j in range(img.height - k + 1):
        temp_data = []
        for i in range(img.width - k + 1):
            temp_data.append(np.sum(np.multiply(mask, img_arr[j:j+k, i: i+k])))
        new_arr.append(temp_data)
    new_arr = np.array(new_arr)
    img = Image.fromarray(new_arr)
    return img

def gaussian(img, mask):
    """Gaussian filter"""
    # Frist part, x direction
    new_arr = []
    img_arr = np.array(img)
    # print("The image size is: ", img_arr.shape)
    x = mask.shape[1]
    for j in range(img.height):
        temp_data = []
        for i in range(img.width - x + 1):
            temp_data.append(np.sum(np.multiply(mask, img_arr[j, i:i+x])))
        new_arr.append(temp_data)
    new_arr = np.array(new_arr)
    # print("After first op: ", new_arr.shape)

    # Second part, y direction
    new_arr2 = []
    mask = mask.T
    y = mask.shape[0]
    # print(y)
    for j in range(new_arr.shape[1]):
        temp_data2 = []
        for i in range(new_arr.shape[0] - y + 1):
            temp_data2.append(np.sum(np.multiply(mask.T, new_arr[i:i+y, j])))
        new_arr2.append(temp_data2)
    new_arr2 = np.array(new_arr2)
    # print("After second op: ", new_arr2.shape)
    return Image.fromarray(new_arr2.T)

def laplacian(img, mask, threshold=False):
    """Laplacian filter"""
    print("The sum of the mask is: ", np.sum(mask))
    k = mask.shape[0]

    # Validate image dimensions
    if k > img.width or k > img.height:
        print("Invalid image dimensions...Exiting")
        sys.exit(0)

    # run the convolution
    img_arr = np.array(img, dtype=np.float64)
    new_arr = []
    for j in range(img.height - k + 1):
        temp_data = []
        for i in range(img.width - k + 1):
            temp_data.append(np.sum(np.multiply(mask, img_arr[j:j+k, i: i+k])))
        new_arr.append(temp_data)
    new_arr = np.array(new_arr)
    if threshold:
        return new_arr
    low = np.amin(new_arr)
    high = np.amax(new_arr)
    op = np.vectorize(lambda x: ((x-low) / (high - low)) * (128 - 0))
    new_arr = op(new_arr)
    img = Image.fromarray(new_arr)
    return img

def scale(img_arr, low, high):
    nlow = np.amin(img_arr)
    nhigh = np.amax(img_arr)
    op = np.vectorize(lambda x: ( (x-nlow) / (nhigh - nlow) ) * ( high - low ))
    return op(img_arr)

def threshold(img):
    img_arr = img
    print(img_arr.dtype)
    op = np.vectorize(lambda x: 0 if x <= 0 else 255)
    img_arr = op(img_arr)
    print(img_arr.shape, img_arr.dtype, np.amax(img_arr), np.amin(img_arr))
    return scale(img_arr, 0, 255)
    # return Image.fromarray(img_arr) 

def pattern_match(img_file, sub_img):
    """pattern match the image"""
    # Construct the filter with normalization
    img = otsu(img_file)
    bin_img = binarize(np.array(img), 0, 1)
    # img.show()
    sub_img = otsu(sub_img)
    bin_sub_img = binarize(np.array(sub_img), -1, 1)

    mask = bin_sub_img
    mask_x, mask_y = mask.shape

    # run the convolution
    img_arr = bin_img
    img_x, img_y = bin_img.shape
    print("Image dimensions: ", img_x, img_y)
    print("Mask shape", mask_x, mask_y)
    print(np.amax(img_arr))
    print(np.amax(mask))

    new_arr = []
    for j in range(img_x - mask_x + 1):
        temp_data = []
        for i in range(img_y - mask_y + 1):
            temp_data.append(np.sum(np.multiply(mask, img_arr[j: j+mask_x, i:i+mask_y])))
        new_arr.append(temp_data)
    new_arr = np.array(new_arr)
    print(new_arr.shape)
    new_arr = scale(new_arr, 0, 255)
    img = Image.fromarray(new_arr)
    return img

def binarize(img_arr, low, high):
    op = np.vectorize(lambda x: low if x == 0 else high)
    return op(img_arr)

def partA1():
    img = Image.open("./b2_a.png")

    # img3 = box_filter(img, 1, k=3)
    # img3 = img3.convert('RGB')
    # img3.save("out_3.jpeg")

    # img5 = box_filter(img, 1, k=5)
    # img5 = img5.convert('RGB')
    # img5.save("out_5.jpeg")

    # gmask = np.array([0.03, 0.07, 0.12, 0.18, 0.20, 0.18, 0.12, 0.07, 0.03]).reshape((1, 9))
    # gimg = gaussian(img, gmask)
    # gimg = gimg.convert("RGB")
    # gimg.save("gimg.png")

    # lmask = np.array([[0, -1 ,0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    # limg = laplacian(gimg, lmask)
    # limg = limg.convert("RGB")
    # limg.save("limp.png")

    # limg_arr = laplacian(gimg, lmask, True)
    # limg_arr = threshold(limg_arr)
    # final_limg = Image.fromarray(limg_arr)
    # final_limg = final_limg.convert("RGB")
    # final_limg.save("limp_f.png")
    pat_img = pattern_match("keys.png", "temp.png")
    pat_img = pat_img.convert("RGB")
    pat_img.save("patt_match.png")


if __name__ == "__main__":
    partA1()
