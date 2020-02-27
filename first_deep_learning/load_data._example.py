# 子类定义  和上面介绍有点小区别，多了一个图像路径参数，因为我的txt中只有文件名！！！
import torch.utils.data as data
import torchvision.transforms as transforms
from os import listdir
from os.path import join
from PIL import Image


def image_file_name(txtfile,image_dir):
    image_filenames = []
    image_classes = []
    with open(txtfile,'r') as f:
        for line in f:
            image_filenames.append(image_dir+line.split()[0])
            image_classes.append(int(line.split()[1]))

    return image_filenames,image_classes

def load_img(filepath):
    with open(filepath, 'rb') as f:
        img = Image.open(filepath)
        return img.convert('RGB')


class DatasetFromTxt(data.Dataset):
    def __init__(self, txtfile, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromTxt, self).__init__()

        self.image_filenames, self.image_classes = image_file_name(txtfile, image_dir)

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = self.image_classes[index]
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)