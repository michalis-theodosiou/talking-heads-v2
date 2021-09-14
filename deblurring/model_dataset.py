import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import cv2
import random


class DeblurringModule(nn.Module):
    def __init__(self, kernel_size=3):
        assert kernel_size % 2 == 1, 'kernel size must be odd'
        super(DeblurringModule, self).__init__()

        padding = kernel_size//2
        relu = nn.LeakyReLU(inplace=True)
        conv_in = nn.Conv2d(3, 64, kernel_size, stride=1, padding=padding, bias=False)
        conv_out = nn.Conv2d(64, 3, kernel_size, stride=1, padding=padding, bias=False)
        conv_mid = nn.Conv2d(64, 64, kernel_size, stride=1, padding=padding, bias=False)

        layers = []
        layers.append(conv_in)
        layers.append(relu)
        for i in range(10):
            layers.append(conv_mid)
            layers.append(relu)
        layers.append(conv_out)

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        out = self.model(img)
        final_out = torch.add(out, img)
        return final_out


class DeblurringDataset(Dataset):
    def __init__(self, data_dir='/content/drive/MyDrive/Colab Datasets/MEAD_video/M003', batchsize=32):
        self.files = glob.glob(data_dir+'/**/*.mp4', recursive=True)
        self.batchsize = batchsize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        f = self.files[idx]
        vid = cv2.VideoCapture(f)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = random.randint(0, num_frames-self.batchsize-1)
        vid.set(1, start_frame)
        frames_blur = np.empty((self.batchsize, 256, 256, 3))
        frames_tgt = np.empty((self.batchsize, 256, 256, 3))
        for i in range(self.batchsize):
            ret, frame = vid.read()
            frame_tgt = self.crop_and_downsample(frame)
            im = Image.fromarray(frame_tgt)
            im = im.filter(ImageFilter.BoxBlur(1.5))
            frame_blur = np.array(im)
            frames_blur[i] = frame_blur
            frames_tgt[i] = frame_tgt

        frames_blur = (np.swapaxes(frames_blur, 1, 3)/255.).astype('float32')
        frames_tgt = (np.swapaxes(frames_tgt, 1, 3)/255.).astype('float32')

        return torch.tensor(frames_blur).cuda(), torch.tensor(frames_tgt).cuda()

    def crop_and_downsample(self, img, img_dim=256):
        h, w, c = img.shape
        crop_width = (w - h) // 2
        img_crop = img[0:h, crop_width:w-crop_width]
        img_resize = cv2.resize(img_crop, dsize=(img_dim, img_dim))

        return img_resize
