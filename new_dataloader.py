import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
from skimage import io
from new_face_align import preprocess_image
import cv2

task_series=['T1','T6','T7','T8']
task_labels=['Happiness','Embarassment','Fear or nervous','Pain']

face_cascade1 = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\envs\pytorches\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
face_landmark1 = cv2.face.createFacemarkLBF()
face_landmark1.loadModel('F:/lbfmodel.yaml.txt')

# https://github.com/omarsayed7/Deep-Emotion/blob/master/data_loaders.py

class image_Loader(Dataset):
    def __init__(self, crop_size, csv_dir, img_dir, transform, target_transform, phase):

        #==============Labels concatenates============
        self.csv_file = pd.read_csv(csv_dir)
        #self.action_unit = action_unit
        self.crop_size = crop_size
        #==============Images===========================
        #self.img_subset = self.csv_file['0']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase
        
    def __len__(self):
        return(len(self.csv_file))
        
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load a row of indicies as label
        df2 = self.csv_file.copy()
        df2.columns = pd.to_numeric(self.csv_file.columns, errors = 'coerce')
        df2 = df2[df2.columns.dropna()]

        y_labels = torch.tensor(df2[[1.0,2.0,4.0,6.0,7.0,10.0,12.0,14.0,15.0,17.0,23.0,24.0]].iloc[0,:].to_numpy())

        #y_labels = torch.tensor(df2.iloc[idx,:].astype('int64').to_numpy())
        #
        # Remove images that have uknown (9) AU labels.

        img_filename = self.img_dir + self.csv_file['Subject'][idx] + "\\" + self.csv_file['Task'][idx] + "\\" + str(self.csv_file['Number'][idx]) + ".jpg"
        counter = 0
        while not os.path.exists(img_filename):
#            print(img_filename)
            counter += 1
            if counter >= 5:
                print("Filename not found for ",img_filename)
                raise FileNotFoundError("We failed to find this file")
                break
            img_filename = '\\'.join(img_filename.split("\\")[:-1]+["0"+img_filename.split("\\")[-1]])

        img = Image.open(img_filename)
        print(img_filename)
        new_img, landmarks, new_biocular_dist = preprocess_image(img, face_landmark=face_landmark1, face_cascade=face_cascade1) #Normalize data

        if self.phase == 'train':
            w = new_img.shape[0]; h = new_img.shape[1]
            offset_w = np.random.randint(0,w-self.crop_size) 
            offset_h = np.random.randint(0,h-self.crop_size)
            flip = np.random.randint(0,1)
            #print(new_img)
            #print(type(new_img))
            if self.transform is not None:        
                img = self.transform(new_img, flip, offset_w, offset_h) # Apply transformation to normalized data
                #img = curr_transform(new_img)
            if self.target_transform is not None:
                new_landmarks = self.target_transform(landmarks,flip,offset_w,offset_h)

        elif self.phase == 'test':
            w,h = new_img.size
            offset_w = (w-self.crop_size) / 2 
            offset_h = (h-self.crop_size) / 2
            if self.transform is not None:
                img = self.transform(new_img, 0, offset_w, offset_h) #Apply transformation to normalized data
                #img = curr_transform(new_img)
            if self.target_transform is not None:
                new_landmarks = self.target_transform(landmarks,0,offset_w,offset_h)

        # Why do I only return the unpreprocessed image?
        #return(img, new_landmarks, new_biocular_dist, y_labels)
    
        return(img,new_landmarks,new_biocular_dist,y_labels)


def eval_data_dataloader(csv_file,img_dir,sample_number,action_unit,transform=None):
    if transform is None:
        transform = transforms.Compose([
                    transforms.Resize(257),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = image_Loader(csv_dir=csv_file,img_dir=img_dir,transform=transform,action_unit="32")
    #print("Transform:",transform)
    label = dataset.__getitem__(sample_number)[1]
    #print(label)
    imgg = dataset.__getitem__(sample_number)[0]
    imgt = (imgg.permute(1,2,0)).numpy()
    #print(imgnp.shape)
    #imgt = imgnp.squeeze()
    plt.imshow(imgt)
    plt.show()

#eval_data_dataloader(csv_file="F:\FaceExprDecode\demo_labels\F001_T1.csv", img_dir="F:/FaceExprDecode/demo_pics/",sample_number=10)
#eval_data_dataloader(train_csv,img_dir="F:\\FaceExprDecode\\F001\\train\\",sample_number=5,action_unit="32")
# eval_data_dataloader(csv_file="F:\\FaceExprDecode\\F001\\train\\train.csv",img_dir="F:\\FaceExprDecode\\F001\\",sample_number=10,action_unit="32")

