import warnings
import glob
from resemblyzer import preprocess_wav
import random
from torch.utils.data import Dataset
import pickle
import random


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
        assert split in ('test', 'train', None), 'split parameter must equal "train", "test" or None')

        if memory == 'disk':
            self.intensity_level='level_' + str(intensity)
            self.dir=directory
            self.filelist=glob.glob('{}/**/{}/*.m4a'.format(self.dir,
                                      self.intensity_level), recursive = True)
            self.emotions=sorted(list(set(path.split('/')[3] for path in self.filelist)))
            self.speakers=sorted(list(set(path.split('/')[1] for path in self.filelist)))
            self.utterances=sorted(
                list(set(path.split('/')[5].split('.')[0] for path in self.filelist)))

        if self.memory == 'ram':
            with open(f'/content/drive/MyDrive/Colab Datasets/MEAD_{self.intensity}_total.pkl', 'rb') as f:
                print('loading dataset from drive...')

                if split is None:
                    self.dataset=pickle.load(f)
                else:
                    dataset=pickle.load(f)
                    speakers=list(dataset.keys())
                    random.seed(1)
                    test_speakers=random.sample(speakers, 10)
                    if split == 'train':
                        self.dataset={spk: dataset[spk]
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
