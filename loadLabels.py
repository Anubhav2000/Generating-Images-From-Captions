import os
from PIL import Image

root = "F:\\Games\\Sem Project Dataset\\birds\\images"
save = "F:\Games\Sem Project Dataset\IMAGES"
count = 0
for dir in os.listdir(root):
    if not os.path.exists(os.path.join(save, dir)):
        os.makedirs(os.path.join(save, dir))
    for file in os.listdir(os.path.join(root, dir)):
        count = count + 1
        im = Image.open((os.path.join(root, dir, file)))
        size = 64, 64
        im = im.resize(size, Image.ANTIALIAS)
        im.save(os.path.join(save, dir, file), "JPEG")

print(count)
