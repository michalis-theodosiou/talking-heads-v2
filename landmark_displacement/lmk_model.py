import os
import torch.nn.parallel
import torch.utils.data
from src.dataset.audio2landmark.audio2landmark_dataset import Audio2landmark_Dataset
from src.models.model_audio2landmark import *
from util.utils import get_n_params
import numpy as np
import pickle


class Audio2landmark_model_talkingheads():

    def __init__(self, jpg_shape, ckpt, output_folder):
        '''
        Init model with opt_parser
        '''

        # Step 1 : load opt_parser
        self.std_face_id = jpg_shape
        self.std_face_id = self.std_face_id.reshape(1, 204)
        self.std_face_id = torch.tensor(
            self.std_face_id, requires_grad=False, dtype=torch.float).to(device)
        self.output_folder = output_folder
        ########################################################################
        self.eval_data = Audio2landmark_Dataset(dump_dir='/content/talking-heads-v2/examples/dump',
                                                dump_name='random',
                                                status='val',
                                                num_window_frames=18,
                                                num_window_step=1)
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_data, batch_size=1,
                                                           shuffle=False, num_workers=0,
                                                           collate_fn=self.eval_data.my_collate_in_segments)

        ''' baseline model '''
        self.C = Audio2landmark_content(num_window_frames=18,
                                        in_size=80, use_prior_net=True,
                                        bidirectional=False, drop_out=0.5)

        ckpt = torch.load(ckpt)
        self.C.load_state_dict(ckpt['model_g_face_id'])

        self.C.to('cuda')

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        self.anchor_t_shape = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt')
        self.anchor_t_shape = self.anchor_t_shape[self.t_shape_idx, :]

    def __train_face_and_pos__(self, fls, aus, face_id, smooth_win=31, close_mouth_ratio=.99):

        fls_without_traj = fls[:, 0, :].detach().clone().requires_grad_(False)

        if (face_id.shape[0] == 1):
            face_id = face_id.repeat(aus.shape[0], 1)
        face_id = face_id.requires_grad_(False)
        baseline_face_id = face_id.detach()
        residual_face_id = baseline_face_id

        # ''' CALIBRATION '''
        baseline_pred_fls, _ = self.C(aus[:, 0:18, :], residual_face_id)
        baseline_pred_fls = self.__calib_baseline_pred_fls__(baseline_pred_fls)
        fl_dis_pred = baseline_pred_fls

        return fl_dis_pred, face_id[0:1, :]

    def __calib_baseline_pred_fls__(self, baseline_pred_fls, ratio=0.5):
        np_fl_dis_pred = baseline_pred_fls.detach().cpu().numpy()
        K = int(np_fl_dis_pred.shape[0] * ratio)
        for calib_i in range(204):
            min_k_idx = np.argpartition(np_fl_dis_pred[:, calib_i], K)
            m = np.mean(np_fl_dis_pred[min_k_idx[:K], calib_i])
            np_fl_dis_pred[:, calib_i] = np_fl_dis_pred[:, calib_i] - m
        baseline_pred_fls = torch.tensor(np_fl_dis_pred, requires_grad=False).to(device)
        # baseline_pred_fls[:, 48 * 3::3] *= self.opt_parser.amp_lip_x  # mouth x
        # baseline_pred_fls[:, 48 * 3 + 1::3] *= self.opt_parser.amp_lip_y  # mouth y
        return baseline_pred_fls

    def __train_pass__(self, au_emb=None, vis_fls=False):

        # Step 1: init setup
        self.C.eval()
        data = self.eval_data
        dataloader = self.eval_dataloader

        # Step 2: train for each batch
        for i, batch in enumerate(dataloader):

            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]

            # Step 2.1: load batch data from dataloader (in segments)
            inputs_fl, inputs_au, _ = batch

            keys = ['audio_embed']
            for key in keys:
                # load saved emb
                emb_val = au_emb[i]

                inputs_fl, inputs_au = inputs_fl.to(device), inputs_au.to(device)

                std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
                seg_bs = 512

                for j in range(0, inputs_fl.shape[0], seg_bs):

                    # Step 3.1: load segments
                    inputs_fl_segments = inputs_fl[j: j + seg_bs]
                    inputs_au_segments = inputs_au[j: j + seg_bs]

                    if(inputs_fl_segments.shape[0] < 10):
                        continue

                    input_face_id = self.std_face_id

                    fl_dis_pred_pos, input_face_id = \
                        self.__train_face_and_pos__(inputs_fl_segments, inputs_au_segments,
                                                    input_face_id)

                    fl_dis_pred_pos = (fl_dis_pred_pos + input_face_id).data.cpu().numpy()
                    ''' solve inverse lip '''
                    fl_dis_pred_pos = self.__solve_inverse_lip2__(fl_dis_pred_pos)
                    fls_pred_pos_list += [fl_dis_pred_pos]

                fake_fls_np = np.concatenate(fls_pred_pos_list)

                # revise nose top point
                fake_fls_np[:, 27 * 3:28 * 3] = fake_fls_np[:, 28 *
                                                            3:29 * 3] * 2 - fake_fls_np[:, 29 * 3:30 * 3]

                # smooth
                from scipy.signal import savgol_filter
                fake_fls_np = savgol_filter(fake_fls_np, 5, 3, axis=0)

                ############################################################################################
                filename = 'pred_fls_{}_{}.txt'.format(
                    video_name.split('\\')[-1].split('/')[-1], key)
                np.savetxt(os.path.join(self.output_folder, filename), fake_fls_np, fmt='%.6f')

                # ''' Visualize result in landmarks '''
                if(vis_fls):
                    from util.vis import Vis
                    Vis(fls=fake_fls_np, filename='tmp_lmk', fps=62.5,
                        audio_filenam='/content/talking-heads-v2/examples/dump/tmpaudio.wav')
                    print('filename = ', video_name.split('\\')[-1].split('/')[-1])

    def test(self, au_emb=None):
        with torch.no_grad():
            self.__train_pass__(au_emb, vis_fls=True)

    def extract_lmks(self, au_emb):
        with torch.no_grad():
            self.__train_pass__(au_emb, vis_fls=False)

    def __solve_inverse_lip2__(self, fl_dis_pred_pos_numpy):
        for j in range(fl_dis_pred_pos_numpy.shape[0]):
            init_face = self.std_face_id.detach().cpu().numpy()
            from util.geo_math import area_of_signed_polygon
            fls = fl_dis_pred_pos_numpy[j].reshape(68, 3)
            area_of_mouth = area_of_signed_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < 0):
                fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3] = 0.5 * \
                    (fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] +
                     fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3])
                fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] = fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3]
                fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3] = 0.5 * \
                    (fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] +
                     fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3])
                fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] = fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3]
                fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3] = 0.5 * \
                    (fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] +
                     fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3])
                fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] = fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3]
                p = max([j-1, 0])
                fl_dis_pred_pos_numpy[j, 55 * 3+1:59 * 3+1:3] = fl_dis_pred_pos_numpy[j, 64 * 3+1:68 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 55 * 3+1:59 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 64 * 3+1:68 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 59 * 3+1:60 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:61 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 59 * 3+1:60 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 60 * 3+1:61 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 49 * 3+1:54 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:65 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 49 * 3+1:54 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 60 * 3+1:65 * 3+1:3]
        return fl_dis_pred_pos_numpy


def lmk_emotion_adjustment(lmks, EMOTION):
    if EMOTION in ('surprised', 'happy', 'fear'):
        lmks[[37, 38, 43, 44], 1] -= 1  # larger eyes
        lmks[[40, 41, 46, 47], 1] += 1  # larger eyes
    if EMOTION == 'surprised':
        lmks[[17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] -= 4  # raise eyebrows
    if EMOTION == 'fear':
        lmks[[17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] -= 2  # raise eyebrows
    if EMOTION == 'happy':
        lmks[48:, 0] = (lmks[48:, 0] - np.mean(lmks[48:, 0])) * \
            1.1 + np.mean(lmks[48:, 0])  # wider lips
    return lmks
