import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from torchvision import transforms
# Dummy dataset of pairs
class SiameseDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path 
        self.transform = transform
        self.datasets = ImageFolder(folder_path )
        self.images , self.labels = zip(*self.datasets.imgs)
        self.labels = np.array(self.labels)
        self.images , self.labels = self._generate_pairs()
    def _generate_pairs(self): 
        images = []
        labels_ = []
        for idx in range(len(self.images)):
            image_1 = self.images[idx]
            labels = np.where(self.labels == self.labels[idx] , 1 , 0)
            for i_idx in range(self.labels.shape[0]):
                image2 = self.images[i_idx]
                label = 1 if labels[idx] == labels[i_idx] else 0
                images.append((image_1 , image2))
                labels_.append(label)
        return images , labels_
    def class_to_idx(self):
        return self.datasets.class_to_idx
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image1 , image2 = self.images[idx]
        label = self.labels[idx]
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        return (self.transform(image1) , self.transform(image2)) , label
        # Current iamges
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SiameseDataset( "/media/baihaqi/Data_Temp/EdgeFace/datap_prepprocessing/NCT_Save_2/train", transform=transform)
    np.unique(dataset.labels)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)
    print(len(dl))


