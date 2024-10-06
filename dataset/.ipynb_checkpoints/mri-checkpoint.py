import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from glob import glob
import scipy.io

import dataset.vocoder_LSP_sptk as vocoder_LSP_sptk
import moviepy.editor as mp

from tqdm import tqdm

class MRI(Dataset):
    def __init__(self,
                 args,
                 ):

        self.video_list = list()
        dir_list = sorted(glob(os.path.join(args.dataset_dir, 'sub*')))

        # vocoder params

        # how can we get frameshift?
        #
        # video framerate 83.28fps, video frames 2963
        # total audio samples: (if samplingFreq 20k) --> 20k * 2963 / 83.28 = 711400 samples
        # to get a frames Shift,
        # (711400 frames - 1024 frame shift) / (2963 total frames - 1) ~= 240
        
        self.samplingFrequency = 20000
        self.frameLength = 1024
        self.frameShift = 480
        self.order = 24
        self.alpha = 0.42
        self.stage = 3
        self.n_mgc = self.order + 1
        
        for dir_item in tqdm(dir_list):
            video_fnames = sorted(glob(os.path.join(dir_item, '2drt', 'video', '*.mp4')))

            for video_fname in video_fnames:
                if 'picture' in video_fname:
                    continue
                elif 'topic' in video_fname:
                    continue
                elif 'postures' in video_fname:
                    continue
                else:
                    self.video_list.append(video_fname)

                    audio_fname = video_fname.replace('.mp4', '.wav')

                    if not os.path.isfile(audio_fname):      
                        self.extract_audio_from_video(video_fname, audio_fname)

                    add_aud_fname = audio_fname.replace('.wav', '.mgclsp')

                    if not os.path.isfile(add_aud_fname):
                        mgc_lsp_coeff, lf0 = self.get_mgc_lsp_coeff(audio_fname[:-4])

    def __len__(self):
        return len(self.video_list)
    
    # load vocoder features,
    # or calculate, if they are not available
    def get_mgc_lsp_coeff(self, audio_fname):
        return vocoder_LSP_sptk.encode(
            audio_fname,
            self.samplingFrequency,
            self.frameLength,
            self.frameShift,
            self.order,
            self.alpha,
            self.stage)

    def extract_audio_from_video(self, video_path, audio_output_path):
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path)
    
    # from LipReading with slight modifications
    # https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
    ################## VIDEO INPUT ##################
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        fps = cap.get(cv2.CAP_PROP_FPS)

        #framesPerSec = 23.18
        
        # make sure that all the videos are the same FPS
        #if (np.abs(fps - framesPerSec) > 0.01):
        #    print('fps:', fps, '(' + video_path + ')')
        #    raise
        
        buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype('float32')
            # min-max scaling to [0-1]
            frame = frame-np.amin(frame)
            # make sure not to divide by zero
            if np.amax(frame) != 0:
                frame = frame/np.amax(frame)
            buf[:,:,fc]=frame
            fc += 1
        cap.release()

        return buf

    def load_mgc(self, fname):
        return np.fromfile(fname, dtype=np.float32).reshape(-1, self.order + 1)

    def load_lf0(self, fname):
        return np.fromfile(fname, dtype=np.float32)
        
    def __getitem__(self, idx):

        # load video data here

        video_fname = self.video_list[idx]
        mgc_fname = video_fname.replace('.mp4', '.mgclsp')
        lf0_fname = video_fname.replace('.mp4', '.lf0')
        
        video = self.load_video(video_fname)
        mgc_lsp_coeff = self.load_mgc(mgc_fname)
        lf0 = self.load_lf0(lf0_fname)
        import pdb; pdb.set_trace()
        
        # check validation of data
        
        return text_feat, data
