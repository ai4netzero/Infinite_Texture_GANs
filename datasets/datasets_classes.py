from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
import glob
from PIL import Image,ImageOps,ImageEnhance,ImageFile
import matplotlib.pyplot as plt
import torch
import pandas as pd 
import ast 
from utils import *
from random import sample

#ImageFile.LOAD_TRUNCATED_IMAGES = True

class Channels(Dataset):
    def __init__(self, path = None,csv_path = None,ext = 'txt'
                    ,sampling = None,center_crop = None,random_crop = None):

        # folder path of images       
        self.path = path 
        # csv path containing meta data
        self.csv_path = csv_path
        self.ext =ext
        self.center_crop = center_crop
        self.random_crop = random_crop

        if csv_path ==None: # read from csv file if found
            self.img_list = os.listdir(path)
            if sampling:
                self.img_list = sample(self.img_list,sampling)
        else:
            self.img_list = pd.read_csv(csv_path)
            if sampling:
                self.img_list = self.img_list.sample(sampling)
        #self.img_list = self.labels.name.to_list()





        if self.ext != 'txt' and self.ext != 'grdecl': # png,jpg
            self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        else:
            self.transform1 = None

        if center_crop:
            self.trans_crop = transforms.Compose([transforms.CenterCrop(center_crop),
                                    transforms.Resize(64)
                                    ])
            #self.trans_center_crop = transforms.CenterCrop(center_crop)
        elif random_crop:
            self.trans_crop = transforms.Compose([transforms.RandomCrop(random_crop),
                                    transforms.Resize(64)
                                    ])
        else:
            self.trans_crop = None


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.csv_path:
            img_name = self.img_list.iloc[idx]['name']
        else:
            img_name = self.img_list[idx]
        full_img_path = os.path.join(self.path,img_name)
        #full_img_path = self.img_list.iloc[idx]['name']

        if self.ext == 'txt': 
            img = np.loadtxt(full_img_path) #normalized
            if self.trans_crop is not None:
                img = Image.fromarray(img)
                img = self.trans_crop(img)
                img = transforms.ToTensor()(np.array(img)) #(img)
                img = img*1.0
                img[img<=0] = -1
                img[img>0] = 1
            
                #img = torch.from_numpy(img).float()
            else:
                img = torch.from_numpy(img).unsqueeze(0).float()

        elif self.ext == 'png' or self.ext == 'jpg':
            img =Image.open(full_img_path)
            if self.center_crop:
                img = self.trans_crop(img)
            if self.transform1:
                img = self.transform1(img)
        elif self.ext == 'grdecl':
            img =read_gcl(full_img_path) 
            img = normalize_by_replace(img)
            img = torch.from_numpy(img).unsqueeze(0).float()
        elif self.ext =='npy':
            img = np.load(full_img_path) # (D,H,W) in case of 3D
            img = np.expand_dims(img,axis=0) #(1,D,H,W)
            img[img==2] = -1.0
            img[img==0] = 1.0
            #img = img.astype(float)
            #print(img)





  

        if self.csv_path == None:
            return {0: img}
        else:
            if 'label_path' in self.img_list:
                l_p = self.img_list.iloc[idx]['label_path']
                l = np.load(l_p)
            elif 'label' in self.img_list:
                l = l_p = self.img_list.iloc[idx]['label']
                if type(l) == str:
                    l = ast.literal_eval(l)
                    #l = from_np_array(l)
            else:
                return {0: img}
                
            l = torch.tensor(l).float()
            return {0: img, 1: l}


class single_image(Dataset):
    def __init__(self, path = None,ext = 'txt',center_crop = None,random_crop = None,sampling = None):       
        self.img_path = path 
        self.ext =ext
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.sampling = sampling
        full_img_path = self.img_path

        #self.img_list = self.labels.name.to_list()
        if self.ext == 'txt': # for some simple binay geological images saved as text files
            self.img = np.loadtxt(full_img_path) #normalized
            self.img = Image.fromarray(self.img)

            self.transform1 = None
        else:
            self.img =Image.open(full_img_path)
            self.img.save("geeks.jpg")
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


        #print(full_img_path)
        #self.img = img
        #print(self.img)
        #exit()

    def __len__(self):
        if self.sampling:
            return self.sampling
        else:
            return 10000

    def __getitem__(self, idx):

        #full_img_path = self.img_path
        #img =Image.open(full_img_path)
        #self.img =Image.open(full_img_path)

        if self.ext == 'txt': 
            if self.trans_crop is not None:
                img = self.trans_crop(self.img)
                img = transforms.ToTensor() (img)
                img = img*1.0
                img[img<=0] = -1
                img[img>0] = 1
                #print(img.shape)
                #print(torch.unique(img))
                #exit()
                #img = torch.from_numpy(img).float()
            else:
                img = transforms.ToTensor() (self.img).float()
                #img = torch.from_numpy(self.img).unsqueeze(0).float()
     
        else:
            if self.trans_crop:
                img = self.trans_crop(self.img)
            img = self.transform1(img)

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

def from_np_array(array_string):
    if array_string[1] == '[':
        array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def read_gcl(filepath):
    file_pointer = open(f"{filepath}", "r")
    data_list = []
    for line in file_pointer:
        line = line.strip().split(' ')

        if line[0] == '--' or line[0] == '' or line[0].find('_') > 0:
            continue
        for data_str_ in line:
            if data_str_ == '':
                continue
            elif data_str_.find('*') == -1:
                try:
                    data_list.append(int(data_str_))
                except:
                    pass # automatically excludes '/'
            else:
                run = data_str_.split('*')
                inflation = [int(run[1])] * int(run[0])
                data_list.extend(inflation)

    file_pointer.close()
    data_np = np.array(data_list)
    # print(data_np.shape)
    data_np = data_np.reshape(100, 100)
    return data_np

d = {2:-1,3:0,5:1} # maps for non-stat


def normalize_by_replace(img):
    for key, value in d.items():
        img[img==np.array(key)] = value
    return img

def image_resize(image, width = None, height = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (w, h) = image.size[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    #resized = cv2.resize(image, dim, interpolation = inter)
    resized = image.resize(dim, Image.Resampling.LANCZOS)

    # return the resized image
    return resized
