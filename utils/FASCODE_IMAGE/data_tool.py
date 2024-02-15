import os

dir = "parsing/"
file_list = os.listdir(dir)

for idx, img in enumerate(file_list):
    os.rename(dir+img, dir+img.split("_gray")[0] + ".png")