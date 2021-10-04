from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import glob
import random
import os
import torch
import numpy as np
from PIL import Image
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        # self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
        tmp_list = []
        top_1_dir = os.listdir(root)
        for tmp_dir in top_1_dir:
            top_2_dir = os.listdir(os.path.join(root+tmp_dir))
            for tmp_sub_dir in top_2_dir:
                final_file = os.listdir(os.path.join(root+tmp_dir+"/"+tmp_sub_dir))
                ten_arr = []
                for file in final_file:
                    if len(file.split('.')) == 2:
                        ten_arr.append(os.path.join(root+tmp_dir+"/"+tmp_sub_dir+"/"+file))
                tmp_list.append(ten_arr)



        #print(len(tmp_list))
        self.tenImageFiles = tmp_list
        self.firstImageFiles = [img[0] for img in self.tenImageFiles]


    def __getitem__(self, index):
        # print(len(self.tenImageFiles), len(self.firstImageFiles))
        # for dir in ()
        # print("index: " + str(index))
        img_A = Image.open(self.firstImageFiles[index])
        img_B = [Image.open(self.tenImageFiles[index][i]) for i in range(10)]
        # print("len of img_b:" +str(len(img_B)))
        # w, h = img.size
        # img_A = img.crop((0, 0, w / 2, h))
        # img_B = img.crop((w / 2, 0, w, h))

        img_A = Image.fromarray(np.array(img_A), "L")
        img_B = [Image.fromarray(np.array(img), "L") for img in img_B]
        # img_B = Image.fromarray(np.array(img_B), "L")

        img_A = self.transform(img_A)
        img_B = torch.stack([self.transform(img) for img in img_B])
        # img_B = self.transform(img_B)
        img_B = torch.squeeze(img_B)
        #print(img_A.shape)
        #print(img_B.shape)
        return {"A": img_A, "B": img_B}


    def __len__(self):
        return len(self.tenImageFiles)