import numpy as np
from torch.utils.data import Dataset
import pickle
import glob
import pandas as pd
import cv2
import random
from src.dataset.image_translation.data_preparation import vis_landmark_on_img
from sklearn.decomposition import PCA


class img_generator_dataset_train(Dataset):

    def __init__(self, lmk_predictor,
                 data_dir='/content/drive/MyDrive/Colab Datasets/MEAD_video/M003',
                 emb_file='/content/drive/MyDrive/Colab Datasets/emotion_centroids.pkl',
                 img_dim=256):

        self.img_dim = img_dim

        # load emotion centroids & reduce dimensionality
        with open(emb_file, 'rb') as f:
            self.emotion_centroids = pickle.load(f)
        self.num_emotions = len(self.emotion_centroids)
        self.emotions = list(self.emotion_centroids.keys())
        p = np.empty((len(self.emotions), 256))
        for i, e in enumerate(self.emotions):
            p[i] = self.emotion_centroids[e]
        pca = PCA()
        tr = pca.fit_transform(p)
        pca_centroids = {}
        for i in range(len(self.emotions)):
            pca_centroids[self.emotions[i]] = tr[i]
        self.pca_centroids = pca_centroids

        # all files
        df_files = pd.DataFrame(data=glob.glob(data_dir+'/**/*.mp4',
                                recursive=True), columns=['filepath'])
        df_files['utterance'] = df_files.filepath.str.split('/').str[-1].str[:3]
        df_files['emotion'] = df_files.filepath.str.split('/').str[-3]
        df_files['speaker'] = df_files.filepath.str.split('/').str[-6]
        self.df_files = df_files

        # neutral files
        self.neutral_files = df_files[df_files.emotion == 'neutral']

        # landmark_model
        self.lmk_predictor = lmk_predictor

    def __len__(self):
        return len(self.neutral_files)*self.num_emotions

    def __getitem__(self, idx):

        neutral_idx = idx // self.num_emotions
        tgt_emotion = self.emotions[idx % self.num_emotions]
        emotion_embedding = self.pca_centroids[tgt_emotion]

        # base image
        base_file = self.neutral_files.iloc[neutral_idx].filepath
        base_img = self.return_random_frame(base_file)

        # target image and landmarks
        tgt_file = self.df_files[self.df_files.emotion == tgt_emotion].filepath.sample(1).values[0]
        tgt_img = self.return_random_frame(tgt_file)
        lmks = self.lmk_predictor.get_landmarks(tgt_img)[0]
        white_bg = np.ones(shape=(self.img_dim, self.img_dim, 3)) * 255.0
        lmk_img = vis_landmark_on_img(white_bg, lmks)

        # vis lmks on target img
        # tgt_img_copy = tgt_img.copy()
        # v = vis_landmark_on_img(tgt_img_copy, lmks)

        # stack lmks with base
        stack_img_lmks = np.concatenate((lmk_img, base_img), axis=2) / 255
        # stack with emotion_embedding
        emotion_embedding_reshape = emotion_embedding.reshape((1, 1, -1))
        emotion_embedding_reshape = np.broadcast_to(
            emotion_embedding_reshape, (self.img_dim, self.img_dim, self.num_emotions))
        full_stack = np.concatenate((emotion_embedding_reshape, stack_img_lmks), axis=2)

        full_stack = np.swapaxes(full_stack, 0, 2)
        tgt_img = np.swapaxes(tgt_img, 0, 2) / 255.

        return (base_img.astype('float32'),
                tgt_img.astype('float32'),
                full_stack.astype('float32'),
                tgt_emotion)

    def crop_and_downsample(self, img, img_dim=256):
        h, w, c = img.shape
        crop_width = (w - h) // 2
        img_crop = img[0:h, crop_width:w-crop_width]
        img_resize = cv2.resize(img_crop, dsize=(img_dim, img_dim))

        return img_resize

    def return_random_frame(self, video_dir):
        vid = cv2.VideoCapture(video_dir)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        framenum = random.randint(5, num_frames-10)
        vid.set(1, framenum)
        ret, frame = vid.read()
        img = self.crop_and_downsample(frame, self.img_dim)
        return img
