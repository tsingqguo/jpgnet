import numpy as np
# def crop_center(image):
#     w = image.size[0]
#     h = image.size[1]
#     size = min(w, h) // 2
#     x, y = w // 2, h // 2
#
#     new_image = image.crop((x - size, y-size, x + size, y+size))  # (left, upper, right, lower)
#
#     return new_image


def crop_center(image):
    w = image.size[0]
    h = image.size[1]
    size = min(w, h) // 2
    x, y = w // 2, h // 2

    new_image = image.crop((x - size, y-size, x + size, y+size))  # (left, upper, right, lower)

    return new_image