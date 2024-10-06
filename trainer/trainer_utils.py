import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

#from models.speech_to_2d_mri import Speech2MRI2D
from models.mri_to_speech import MRI2Speech

def build_optimizer_model(args, logger, dataset, device):

    if args.data.feature_mode == 'raw':
        model_in_feat = args.data.frameShift
    else:
        model_in_feat = args.model.in_feat
    
    return load_model(args,
                      model_in_feat,
                      dataset.frameHeight,
                      dataset.frameWidth,
                      dataset.fps,
                      device)
    
def load_model(args, n_input_feats, H, W, fps, device):

    model = MRI2Speech(args=args,
                       n_mgc=n_input_feats,
                       n_width=W,
                       n_height=H,
                       dropout_rate=0.5,
                       fps=fps)

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    
    log_file_list = os.listdir(args.log_dir)

    ckpt_list = list()

    for fname in log_file_list:
        if 'ckpt' in fname:
            ckpt_list.append(fname)

    if len(ckpt_list) > 0:
        if args.select_ckpt_idx == 0:
            last_ckpt_fname = sorted(ckpt_list)[-1]
        else:
            last_ckpt_fname = sorted(ckpt_list)[args.select_ckpt_idx]

        ckpt_path = os.path.join(
            args.log_dir,
            last_ckpt_fname
            )

        state_dict = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        start_iter = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])

        print(f'load: {last_ckpt_fname} of {args.log_dir}')
        
        # Move optimizer state to the GPU
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        mgc_mean = state_dict['mgc_mean']
        mgc_std = state_dict['mgc_std']

        print(f'=======================================')
        print(f'=======================================')
        print(f'loaded model for epoch: {start_iter}')
        print(f'=======================================')
        print(f'=======================================')
        
    else:
        start_iter = 1
        mgc_mean = 0
        mgc_std = 0
        
    return optimizer, scheduler, model, start_iter, (mgc_mean, mgc_std)

def data_batchify(voice, video=None, lookback=10, fps_control_ratio=1):
    # video.shape: [B, L_vid, H, W]
    # voice.shape: [B, L_voi, C]

    # return batchfied video, batchfied audio, initial video

    # for general case, the B is 1, since there are video data

    # L_vid and L_voice can be different if the fps_control_ratio is not 1
    # for this case, the length of video and voice is not compatible,
    # we have to change this value properly

    if video is not None:
        _, L, H, W = video.shape
        video = video.view(L, H, W)
        # for this case, we have to change this lookback value
        # since we do not predict image, but audio, so the adaptive lookback value is diff.
        new_video = video[lookback:]     # new_video.shape: [B, L_vid - lookback, H, W]
    else:
        new_video = None
        
    _, L, C = voice.shape
    voice = voice.permute(1,0,2)

    idx1 = torch.arange(int(lookback*fps_control_ratio)).unsqueeze(0)
    idx2 = torch.arange(int((L-lookback*fps_control_ratio+1) // fps_control_ratio) - 1).unsqueeze(1) * int(fps_control_ratio)
    idx = idx1 + idx2
    new_idx = idx.unsqueeze(-1).repeat(1, 1, C)
    
    new_voice = voice.squeeze().unsqueeze(0).expand(int(L-lookback*fps_control_ratio+1), -1, -1)
    new_audio = torch.gather(new_voice, 1, new_idx.cuda())

    # cut audio and video here again for shortest length
    # if there is video
    
    if video is not None:
        vd_length = min(video.shape[0], new_audio.shape[0])

        new_video = new_video[:vd_length]
        new_audio = new_audio[:vd_length]

    # build audio for supervision
    idx3 = torch.arange(int(fps_control_ratio)).unsqueeze(0)
    idx4 = torch.arange(lookback, int((L) // fps_control_ratio) -1).unsqueeze(1) * int(fps_control_ratio)

    idx2 = idx3 + idx4
    new_idx2 = idx2.unsqueeze(-1).repeat(1, 1, C)
    
    sup_voice = voice.squeeze().unsqueeze(0).expand(int(L-lookback*fps_control_ratio +1), -1, -1)
    sup_audio = torch.gather(new_voice, 1, new_idx2.cuda())

    #sup_audio = voice[int(lookback * fps_control_ratio)::int(fps_control_ratio)]

    # cut a sup audio to make a blance
    ll_length = min(sup_audio.shape[0], new_audio.shape[0])

    sup_audio = sup_audio[:ll_length]
    new_audio = new_audio[:ll_length]
    new_video = new_video[:ll_length]
        
    # debug:
    # new_voice[0] - voice.squeeze()[:10] ---> all 0
    return new_video, new_audio, sup_audio.squeeze()
