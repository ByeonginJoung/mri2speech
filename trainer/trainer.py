import os
import torch
import torch.nn.functional as F
import torchaudio.functional as TF
import torchaudio.transforms as T

import soundfile as sf
from tqdm import tqdm
import subprocess
import cv2
import numpy as np

from dataset.mri import MRI
from trainer.trainer_utils import build_optimizer_model, data_batchify

from utils.seed import set_seed
from utils.viz_utils import visualization

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def test_eval(args, model, epoch_idx, val_loader, logger, res, data_stats, device):
    model.eval()

    mse_loss_list = list()
    
    with torch.no_grad():
        for _, items in enumerate(tqdm(val_loader)):
            video = items[0].to(device)
            audio = items[1].to(device)

            H, W = video.shape[-2], video.shape[-1]
            
            new_video, new_audio, _ = data_batchify(audio, video, args.data.lookback, args.data.fps_control_ratio)

            if args.model.use_deform:
                pred = model(new_audio.float(), video.view(-1, video.shape[-2], video.shape[-1])[0])
                pred = pred.view(-1, res[0], res[1])
            elif args.model.use_prev_frame:
                init_img = video.squeeze()[0]

                temp_vid_list = list()

                for proc_idx, temp_audio in enumerate(tqdm(new_audio)):
                    with torch.no_grad():
                        in_aud = temp_audio.unsqueeze(0)
                        in_vid = new_video[proc_idx]

                        temp_pred = model(in_aud, in_vid.unsqueeze(0)).cpu().detach().squeeze()
                        temp_vid_list.append(temp_pred)

                # to make a full video with given image and predicted audio,
                # use origin video

                # new_video.shape and new_audio is compatible.                        
                pred = torch.stack(temp_vid_list).view(-1, args.model.in_feat)
            else:
                pred = model(audio.float())
                pred = pred.view(-1, res[0], res[1])

    # save file here
    if True:
        # First, save video file.
        video_fname = items[-1][0]
        #prediction = (pred.numpy() * 255).astype(np.uint8)
        process_list = ['cp', '-r', video_fname, 'demo_items/debug_eval_origin.avi']
        subprocess.run(process_list)

        video_output_file = f'demo_items/debug_eval_pred_temp.avi'

        #total_length = pred.view(-1).shape[0] / (44100 / args.data.fps_control_ratio)
        
        if args.dataset_type == 'timit':
            fps = 23.18 / args.data.fps_control_ratio
        elif args.dataset_type == '75-speaker':
            fps = 83.28 / args.data.fps_control_ratio
            
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec (you can use other codecs like 'XVID', 'MJPG', etc.)
        frame_size = (res[1], res[0])  # Frame size
        n_frames = new_video.shape[0]
        
        # Create VideoWriter object
        out = cv2.VideoWriter(video_output_file, fourcc, fps, frame_size)

        # Convert and write frames to video
        for i in range(n_frames):
            # Convert single-channel grayscale to 3-channel BGR
            bgr_frame = cv2.cvtColor((new_video[i].cpu().detach().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            out.write(bgr_frame)

        # Release the VideoWriter object
        out.release()

        # second, save audio file
        audio_output_file = f'demo_items/predicted_audio.wav'

        if args.data.feature_mode == 'raw':
            sf.write(audio_output_file, (pred.view(-1).cpu().detach().numpy() - 0.5) * 2., 44100, format='WAV', subtype='PCM_16')
        elif args.data.feature_mode == 'melspectrogram':
            # get a linear value from db scale
            db_min = -120
            db_max = 50

            pred = (pred * (db_max - db_min)) + db_min   # reverse normalizing as refered in dataloader

            mel_spec = TF.DB_to_amplitude(pred.T, ref=1.0, power=1.0).cuda()

            # convert mel spec to linear
            mel_to_linear_transform = T.InverseMelScale(
                n_stft=args.data.frameLength // 2 + 1,
                n_mels=args.model.in_feat,
                sample_rate=args.data.samplingFrequency,
                f_min=0,
                f_max=8000
            ).cuda()

            # idk why, anyway, mel-spec should be in variable
            
            linear_spectrogram = mel_to_linear_transform(mel_spec).cpu().detach()

            # Apply Griffin-Lim to reconstruct the waveform from the spectrogram
            # this part consumes a plenty of the time
            griffin_lim_transform = T.GriffinLim(n_fft=args.data.frameLength, hop_length=args.data.frameShift)
            reconstructed_waveform = griffin_lim_transform(linear_spectrogram)
            
            # Step 5: Save the reconstructed audio
            sf.write(audio_output_file, reconstructed_waveform.numpy(), args.data.samplingFrequency)

            """
            process = ['sox', f'{audio_output_file}', '-n', 'noiseprof', 'demo_items/noise.prof']
            subprocess.run(process)

            audio_output_file_final = f'demo_items/predicted_audio_final.wav'
            
            process = ['sox', f'{audio_output_file}', f'{audio_output_file_final}',
                       'noisered', 'demo_items/noise.prof', '0.21',
                       'equalizer', '100', '0.5q', '-10', 'equalizer', '1000', '0.7q', '+5', 'equalizer', '5000', '0.6q', '+2',
                       'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2',
                       'reverb', '10', '50', '100',
                       'pitch', '50']
            subprocess.run(process)
            """
            
        # third, load both and concat video and audio
        output_file = f'demo_items/debug_eval_pred_{epoch_idx}.avi'
        command = [
            'ffmpeg', '-y', '-stream_loop', '-1', '-i', audio_output_file, '-i', video_output_file,
            '-shortest', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_file
            ]

        subprocess.run(command)
    
    #print_info = f'total validation length: {len(mse_loss_list)}'
    #print_info2 = f'mse loss for validation set: {mse_loss}'
    #logger.info(print_info)
    #logger.info(print_info2)

def run_trainer(args, logger):

    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = MRI(args)
    val_dataset = MRI(args, val=True)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             shuffle=True,
                                             drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=1,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 drop_last=False)
    
    # set optimizer
    optimizer, scheduler, model, start_epoch, _ = build_optimizer_model(args,
                                                                        logger,
                                                                        dataset,
                                                                        device)
    model = model.to(device)
    
    data_stats = (dataset.mgc_mean, dataset.mgc_std)
    
    frame_H = dataset.frameHeight
    frame_W = dataset.frameWidth

    # TODO list to enhance the performance...
    # add perceptual loss such as SSIM
    # construct more robust AI model

    print(f'start_epoch: {start_epoch}')
    
    for epoch_idx in tqdm(range(start_epoch, args.train_epoch+1)):
        model.train()

        for iteration, items in enumerate(dataloader):
            optimizer.zero_grad()

            video = items[0].to(device)
            audio = items[1].to(device)

            _, B, H, W = video.shape
            
            sup_video, voice, sup_voice = data_batchify(audio, video, args.data.lookback, args.data.fps_control_ratio)
            
            # run model
            # pred first image in initial
            if args.model.use_deform:
                pred = model(voice, video.view(B, H, W)[0])
            elif args.model.use_prev_frame:
                # since the voice and video were cut at data_batchify function,
                # the length of video was changed.
                # therefore change the code a little bit
                pred = model(voice, sup_video.view(-1, H, W))
            else:
                pred = model(voice)

            if args.data.feature_mode == 'melspectrogram':
                pred = pred.view(voice.shape[0], int(args.data.fps_control_ratio), args.model.in_feat)
                
            # pred next image with given initial images
            loss_dict = dict()
            loss_dict['mse_loss'] = torch.nn.functional.mse_loss(pred, sup_voice) * args.mseloss_weight

            if args.use_mrsloss:
                mrs_pred = pred
                mrs_gt = sup_voice

                mrs_loss = 0.
                
                for res, hop, win in zip(args.mrsloss_res, args.mrsloss_hop, args.mrsloss_win):
                    mrs_pred_temp = torch.stft((mrs_pred-0.5) * 2., n_fft=res, hop_length=hop, win_length=win, return_complex=True)
                    mrs_gt_temp = torch.stft((mrs_gt-0.5) * 2., n_fft=res, hop_length=hop, win_length=win, return_complex=True)
                
                    mag_pred = torch.abs(mrs_pred_temp)
                    mag_gt = torch.abs(mrs_gt_temp)

                    mrs_loss += torch.mean((mag_pred - mag_gt)**2) * args.mrsloss_weight

                loss_dict['mrs_loss'] = mrs_loss
                
            if args.use_temporal_consistency:
                temp_cons = pred[:,1:] - pred[:,:-1]
                loss_dict['temp_cons_loss'] = torch.mean(temp_cons ** 2) * args.temporal_cons_weight

            loss = 0
            for key, value in loss_dict.items():
                loss = loss + value
            
            loss.backward()
            optimizer.step()
            
        if epoch_idx % args.epoch_print == 0 or epoch_idx == 1:
            init_viz = '\n Train {} | Epoch: {} | Iter: {} | '.format(args.exp_name, epoch_idx, iteration)
            loss_viz = 'loss: {:4f}'.format(loss.item())

            print_item = init_viz + loss_viz

            for key, value in loss_dict.items():
                temp_print_item = f' | {key}: {value.item():4f}'
                print_item += temp_print_item
            
            logger.info(print_item)

        if epoch_idx % args.epoch_eval == 0 and epoch_idx > 0:
            test_eval(args, model, epoch_idx, val_dataloader, logger, (frame_H, frame_W), data_stats, device)

        # disable for this, since there is no visualization factors
        #if epoch_idx % args.epoch_viz == 0 and epoch_idx > 0:
        #    visualization(args, pred, sup_video, epoch_idx, iteration)

        # Save checkpoint
        if epoch_idx % args.epoch_save == 0 and epoch_idx > 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch_idx,
                     'mgc_mean': data_stats[0] if args.data.feature_mode == 'mgclsp' else 0,
                     'mgc_std': data_stats[1] if args.data.feature_mode == 'mgclsp' else 0,
            }
            torch.save(state, os.path.join(args.log_dir, "ckpt_{:05d}".format(epoch_idx)))
