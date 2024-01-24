from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
from PIL import Image
from random import sample


class single_image(Dataset):
    def __init__(self, path = None,ext = 'jpg',center_crop = None,random_crop = None,sampling = None):       
        self.img_path = path 
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.sampling = sampling
        self.ext = ext
    
        # for some simple binay geological images saved as text files
        if self.ext == 'txt': 
            self.img = np.loadtxt(self.img_path) #normalized
            self.img = Image.fromarray(self.img)
        else:
            self.img =Image.open(self.img_path)
        
        if center_crop:
            self.trans_crop = transforms.Compose([transforms.CenterCrop(center_crop),
                                    ])
        elif random_crop:
            self.trans_crop = transforms.Compose([transforms.RandomCrop(random_crop),
                                    ])
        else:
            self.trans_crop = None
            
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
            
    def __len__(self):
        if self.sampling:
            return self.sampling
        else:
            return 10000

    def __getitem__(self, idx):

        if self.trans_crop:
            img = self.trans_crop(self.img)
        img = self.transform(img)

        return {0: img}


class multiple_images(Dataset):
    def __init__(self, path = None,ext = 'txt'
                 ,center_crop = None,random_crop = None
                 ,resize = None,sampling = None):      
        
        # folder path of images       
        self.path = path 
         
        self.ext =ext
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.sampling = sampling
        self.resize = resize
        

        self.img_list = os.listdir(path)
        if sampling:
            self.img_list = sample(self.img_list,sampling)
    
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        if center_crop:
            self.trans_crop = transforms.Compose([transforms.CenterCrop(center_crop),
                                    transforms.Resize(64)
                                    ])

            #self.img = Image.fromarray(self.img)
        elif random_crop:
            self.trans_crop = transforms.Compose([transforms.RandomCrop(random_crop),
                                    #transforms.Resize(64)
                                    ])
            #self.img = Image.fromarray(self.img)

        else:
            self.trans_crop = None
        
        if resize:
            self.trans_resize= transforms.Compose([transforms.Resize(resize),
                                    #transforms.Resize(64)
                                    ])
        else:
            self.trans_resize = None


        #print(full_img_path)
        #self.img = img
        #print(self.img)
        #exit()

    def __len__(self):
        if self.sampling:
            return self.sampling
        else:
            return len(self.img_list)

    def __getitem__(self, idx):

        #full_img_path = self.img_path
        #img =Image.open(full_img_path)
        img_name = self.img_list[idx]
        full_img_path = os.path.join(self.path,img_name)
        
        img =Image.open(full_img_path)

        if self.resize is not None:
        #    h,w= self.resize 
        #    img = image_resize(img,width = w,height=h)
            img = self.trans_resize(img)

        
        if self.trans_crop:
            img = self.trans_crop(img)
        img = self.transform1(img)

        return {0: img}