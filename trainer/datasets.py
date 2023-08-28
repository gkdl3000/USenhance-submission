import glob
import json
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None):

        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[index % len(self.files_B)]
        imgname = A_path

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        item_A = self.transform2(Image.open(A_path).convert('L'))

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        item_B = self.transform2(Image.open(B_path).convert('L'))
            
        return {'A': item_A, 'B': item_B, 'imgname': imgname}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None):

        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[index % len(self.files_B)]
        imgname = A_path

        item_A = self.transform(Image.open(A_path).convert('L'))
        item_B = self.transform(Image.open(B_path).convert('L'))
        return {'A': item_A, 'B': item_B, 'imgname': imgname}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    

class InferDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/*" % root))
        
    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        imgname = A_path
        item_A = self.transform(Image.open(A_path).convert('L'))

        return {'A': item_A, 'imgname': imgname}
    
    def __len__(self):
        return len(self.files_A)
