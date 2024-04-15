import glob, os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from topo_ddpm_pipeline import TopoDDPMPipeline

class TopoLoader(Dataset):
    def __init__( self, 
                  data_dir,
                  img_fomat='jpg',
                  img_size=128,
                  batch_size=8,
                  shuffle = True,
                ):

        self.img_fomat = img_fomat
        self.x_paths = sorted(glob.glob(f'{data_dir}/*{self.img_fomat}'))
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def path2arr(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)            
        img = cv2.resize(img, dsize=(self.img_size, self.img_size))
        img_arr = np.array(img)
        return img_arr
    
    def calculate_volume_fraction(self, img_arr):
        vf = sum(sum(row) for row in img_arr) / (len(img_arr) * len(img_arr[0]))
        return 1 - (vf / 255.)
    
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):

        # img preprocess
        x = self.path2arr(self.x_paths[idx])
        vf = self.calculate_volume_fraction(x)
        x = x.astype(np.float16) / 127.5 - 1.
        
        # to tensor
        x = torch.tensor(x, dtype=torch.float16).unsqueeze(0)
        return x, vf

    def get_loader(self):
        return  DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    def show_loader(self, loader, idx=0):
        for i, batch in enumerate(loader):
            if i == idx :  # n-1번째 batch까지 반복
                xs, vfs = batch
                break
        n_batch = len(xs)
        
        plt.figure(figsize=(2*n_batch, 2))
        for n in range(n_batch):
            plt.subplot(1,n_batch,n+1)
            plt.imshow(xs[n,0], cmap='gray')
            plt.title(np.round(vfs[n].numpy(),2))
            plt.axis('off')
        plt.show()
        
def main():
    
    pp = TopoLoader('../db_pattern/DB/wheel_org/')
    loader = pp.get_loader()

    for x in loader:
        print(x.shape)
        print('x sample: ', x[0,:,20,:])
        break
        
if __name__ == "__main__":
    main()