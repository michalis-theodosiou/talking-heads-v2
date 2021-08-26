from resemblyzer import VoiceEncoder, audio
import torch
import numpy as np


class VoiceEncoder_train(VoiceEncoder):

    """parent class of resemblyzer voice encoder to add embedding function with gradient and
    ge2e forward pass
    """

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        self.relu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Sequential(
            torch.nn.Linear(256, 8),
            torch.nn.LogSoftmax(0)
        )
        self.to(device)

    def embed_utterance_train(self, wav, rate=1.3, min_coverage=0.75):
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        mels = torch.from_numpy(mels).to(self.device)
        # forward through the network
        partial_embeds = self(mels)

        # Compute the utterance embedding from the partial embeddings
        raw_embed = partial_embeds.mean(axis=0)
        embed = torch.nn.functional.normalize(raw_embed, p=2, dim=0)

        return embed

    def ge2e_forward(self, data):

        N = len(data.keys())
        M = len(data[list(data.keys())[0]])
        D = 256
        output = torch.empty([N, M, D])

        for n, emotion in enumerate(list(data.keys())):
            for m, au in enumerate(data[emotion]):
                output[n, m, :] = self.embed_utterance_train(au)

        return output

    def embed_dataset(self, dataset):
        # get count of utterances by emotion
        self.eval()
        with torch.no_grad():
            emotion_counts = {}
            for emotion in dataset.emotions:
                emotion_counts[emotion] = 0
                for speaker in dataset.speakers:
                    emotion_counts[emotion] += len(dataset.dataset[speaker][emotion])
            output = {}
            # create empty output dictionary and populate with embeddings
            for emotion in dataset.emotions:
                output[emotion] = torch.zeros((emotion_counts[emotion], 256))
                c = 0
                for speaker in dataset.speakers:
                    for utterance in dataset.dataset[speaker][emotion]:
                        output[emotion][c] = self.embed_utterance_train(utterance)
                        c += 1

        return output

    def softmax_forward_single(self, data):
        emb = self.embed_utterance_train(data)
        softmax = self.softmax(emb)

        return softmax

    def softmax_forward_batch(self, batch):
        output_list = []
        for data in batch:
            output = self.softmax_forward_single(data)
            output_list.append(output)

        return torch.stack(output_list)
