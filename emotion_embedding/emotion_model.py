from resemblyzer import VoiceEncoder, audio
import torch
import numpy as np
from resemblyzer import preprocess_wav


class VoiceEncoder_train(VoiceEncoder):

    """parent class of resemblyzer voice encoder to add embedding function with gradient and
    ge2e forward pass
    """

    def __init__(self, softmax=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        self.relu = torch.nn.LeakyReLU()
        if softmax:
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

    def triplet_forward_single(self, data, softmax=False):
        # data in the format (anchor, positive, negative)
        # output (emb_anchor, emb_positive, emb_negative), ()
        anchor, positive, negative = data
        emb_anchor = self.embed_utterance_train(anchor)
        emb_positive = self.embed_utterance_train(positive)
        emb_negative = self.embed_utterance_train(negative)

        if softmax is False:
            return (emb_anchor, emb_positive, emb_negative)
        elif softmax is True:
            softmax_anchor = self.softmax(emb_anchor)
            return (emb_anchor, emb_positive, emb_negative), softmax_anchor

    def triplet_forward_batch(self, batch, softmax=False):
        anchor_list = []
        positive_list = []
        negative_list = []
        softmax_list = []

        for triplet in batch:
            if softmax is False:
                emb_anchor, emb_positive, emb_negative = self.triplet_forward_single(
                    triplet, softmax)
                anchor_list.append(emb_anchor)
                positive_list.append(emb_positive)
                negative_list.append(emb_negative)
            elif softmax is True:
                (emb_anchor, emb_positive, emb_negative), softmax_anchor = self.triplet_forward_single(
                    triplet, softmax)
                anchor_list.append(emb_anchor)
                positive_list.append(emb_positive)
                negative_list.append(emb_negative)
                softmax_list.append(softmax_anchor)

        if softmax is False:
            return torch.stack(anchor_list), torch.stack(positive_list), torch.stack(negative_list)
        elif softmax is True:
            return (torch.stack(anchor_list),
                    torch.stack(positive_list),
                    torch.stack(negative_list),
                    torch.stack(softmax_list)
                    )

    def embedding_batch(self, batch):
        output_list = []
        for data in batch:
            emb = self.embed_utterance_train(data)
            output_list.append(emb)

        return torch.stack(output_list)

    def forward_emb_softmax(self, data):
        emb = self.embed_utterance_train(data)
        softmax = self.softmax(emb)

        return emb, softmax

    def process_ravdess(self, files):
        from resemblyzer import preprocess_wav
        embeddings = []
        softmax = []
        with torch.no_grad():
            for f in files:
                proc = preprocess_wav(f)
                emb, smax = self.forward_emb_softmax(proc)
                embeddings.append(emb)
                softmax.append(smax)

        return torch.stack(embeddings), torch.stack(softmax)

    def eval_wav(self, wav):
        self.eval()
        with torch.no_grad():
            audio = preprocess_wav(wav)
            return self.embed_utterance_train(audio)
