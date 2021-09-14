import warnings
import glob
from resemblyzer import preprocess_wav
import random
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np


class audio_data_ge2e(Dataset):
    """
    A class to load a batch of audio files to calculate the ge2e loss.
    Creates returns data in the format

    Input Params
    -----------
    directory : str
      The directory where the audio files are located in the original MEAD directory structure
    intensity: int
      The intensity (from 1 - 3) from which to load audio
    num_utterances: int
      The number of utterances to sample from per emotion and speaker
    memory: ['ram','disk']
      whether to read the files from RAM (pkl) or from disk
    split: ['test', 'train', None]
      what split of the dataset to return

    Methods
    ----------
    __get_item__(idx) : returns dict
      Returns a dictionary of lists for the idx speaker in the format
      {emotion_name:[list of randomly sampled utterances]}
      if split is equal to test, then returns the same data each time

    Todo:
    ----------
    - Add validation/training split functionality

    """

    def __init__(self, memory='ram', num_utterances=16, intensity=3, directory=None, split=None):

        self.memory = memory
        self.num_utterances = num_utterances
        self.intensity = intensity
        self.split = split

        assert memory in ('ram', 'disk'), 'memory parameter must be \'ram\' or \'disk\''
        assert split in (
            'test', 'train', None), 'split parameter must equal "train", "test" or None'

        if memory == 'disk':
            self.intensity_level = 'level_' + str(intensity)
            self.dir = directory
            self.filelist = glob.glob('{}/**/{}/*.m4a'.format(self.dir,
                                                              self.intensity_level), recursive=True)
            self.emotions = sorted(list(set(path.split('/')[3] for path in self.filelist)))
            self.speakers = sorted(list(set(path.split('/')[1] for path in self.filelist)))
            self.utterances = sorted(
                list(set(path.split('/')[5].split('.')[0] for path in self.filelist)))

        if self.memory == 'ram':
            with open(f'/content/drive/MyDrive/Colab Datasets/MEAD_{self.intensity}_total.pkl', 'rb') as f:
                print('loading dataset from drive...')

                if split is None:
                    self.dataset = pickle.load(f)
                else:
                    dataset = pickle.load(f)
                    speakers = list(dataset.keys())
                    random.seed(1)
                    test_speakers = random.sample(speakers, 10)
                    if split == 'train':
                        self.dataset = {spk: dataset[spk]
                                        for spk in speakers if spk not in test_speakers}
                    elif split == 'test':
                        self.dataset = {spk: dataset[spk]
                                        for spk in speakers if spk in test_speakers}

            self.speakers = list(self.dataset.keys())
            self.emotions = list(self.dataset[self.speakers[0]].keys())

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        # selects one speaker and takes 16 random utterances for each emotion
        output_dict = {}
        speaker = self.speakers[idx]

        if self.memory == 'disk':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for emotion in self.emotions:
                    all_files = glob.glob(
                        f'{self.dir}/{speaker}/audio/{emotion}/{self.intensity_level}/*.m4a')
                    chosen_files = random.sample(all_files, self.num_utterances)
                    output_dict[emotion] = []
                    for f in chosen_files:
                        output = preprocess_wav(f)
                        output_dict[emotion].append(output)

        else:
            for emotion in self.emotions:
                if self.split == 'test':
                    random.seed(0)
                output_dict[emotion] = random.sample(
                    self.dataset[speaker][emotion], self.num_utterances)

        return output_dict


class audio_data_triplet(Dataset):

    def __init__(self, split=None):

        # self.split = split

        assert split in (
            'test', 'train', None), 'split parameter must equal "train", "test" or None'

        with open('/content/drive/MyDrive/Colab Datasets/MEAD_3_total.pkl', 'rb') as f:
            print('loading dataset from drive...')
            dataset = pickle.load(f)

        speakers = list(dataset.keys())
        self.emotions = list(dataset[speakers[0]].keys())

        random.seed(1)
        test_speakers = random.sample(speakers, 10)

        if split == 'train':
            self.dataset = {spk: dataset[spk]
                            for spk in speakers if spk not in test_speakers}
            self.speakers = [speaker for speaker in speakers if speaker not in test_speakers]
        elif split == 'test':
            self.dataset = {spk: dataset[spk]
                            for spk in speakers if spk in test_speakers}
            self.speakers = test_speakers
        elif split is None:
            self.dataset = dataset
            self.speakers = speakers

        df_utterances = pd.DataFrame(
            data=np.zeros((len(self.emotions), len(self.speakers))),
            columns=self.speakers,
            index=self.emotions)
        dataset_indices = []

        for s in self.speakers:
            for e in self.emotions:
                df_utterances.at[e, s] = len(self.dataset[s][e])
                for i in range(len(self.dataset[s][e])):
                    dataset_indices.append((s, e, i))
        self.indices = dataset_indices
        self.df_utterances = df_utterances.astype(int)

        # if split is not None:
        #     indices = list(range(len(self.indices)))
        #     test_split = 0.2
        #     batch_size = 64
        #     num_test = int(((len(indices)*test_split) // batch_size)*batch_size)
        #     random.seed(0)
        #     test_indices = random.sample(indices, num_test)
        #     if split == 'test':
        #         self.indices = [self.indices[idx] for idx in test_indices]
        #     elif split == 'train':
        #         train_indices = [i for i in indices if i not in test_indices]
        #         self.indices = [self.indices[idx] for idx in train_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        a_speaker, a_emotion, a_utterance = self.indices[idx]
        anchor_audio = self.dataset[a_speaker][a_emotion][a_utterance]

        p_emotion = a_emotion
        p_speaker = random.choice([s for s in self.speakers if s != a_speaker])
        p_utterance = random.choice([i for i in self.indices if i[0] ==
                                    p_speaker and i[1] == p_emotion])[2]
        positive_audio = self.dataset[p_speaker][p_emotion][p_utterance]

        n_emotion = random.choice([e for e in self.emotions if e != a_emotion])
        n_speaker = a_speaker
        n_utterance = random.choice([i for i in self.indices if i[0] ==
                                    n_speaker and i[1] == n_emotion])[2]
        negative_audio = self.dataset[n_speaker][n_emotion][n_utterance]

        return (anchor_audio, positive_audio, negative_audio), a_emotion

    def get_single(self, idx):
        speaker, emotion, utterance = self.indices[idx]
        audio = self.dataset[speaker][emotion][utterance]
        return audio, emotion

    def get_batch(self, indices):
        audio_batch, emotion_batch = [], []
        for idx in indices:
            audio, emotion = self.get_single(idx)
            audio_batch.append(audio)
            emotion_batch.append(emotion)

        return audio_batch, emotion_batch

    def get_triplet_batch(self, indices):
        audio_batch, emotion_batch = [], []
        for idx in indices:
            audio, emotion = self[idx]
            audio_batch.append(audio)
            emotion_batch.append(emotion)

        return audio_batch, emotion_batch


def return_pca_centroid(embedding, centroids, pca_centroids):
    max_sim = -2
    for emotion in centroids.keys():
        sim = cosine_similarity(embedding, centroids[emotion])
        if sim > max_sim:
            max_emotion = emotion
            max_sim = sim
    print(f'inferred "{max_emotion}"')
    return pca_centroids[max_emotion], max_emotion


def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
