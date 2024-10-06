import os
import cv2
import argparse
import torch
import random
import logging
import math
import numpy as np
import subprocess
import torchaudio
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt

from trainer.trainer import run_trainer
from trainer.trainer_utils import load_model

from utils.parser import arg_parser
from utils.logger import set_logger
from utils.seed import set_seed

import dataset.vocoder_LSP_sptk as vocoder_LSP_sptk

from config.utils import load_config

from utils.voice_converter import make_tts_like, make_tts_like_ver2
from trainer.trainer_utils import data_batchify

from tqdm import tqdm
from moviepy.editor import VideoFileClip, clips_array, ImageSequenceClip

def load_video(video_path):
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

def main(args, logger):

    for key in args.keys():
        logger.info(f'{key}: {args[key]}')

    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # for the given pretrained model, img shape: [84, 84]
    if args.dataset_type == '75-speaker':
        frame_H = 84
        frame_W = 84
    elif args.dataset_type == 'timit':
        frame_H = 68
        frame_W = 68
    else:
        raise NotImplementedError
    
    if args.input_fname == 'None' and args.audio_fname == 'None':
        print(f'There is no input item. Please select just one.')
        raise NotImplementedError
    
    if args.exist_input_vid:
        if args.audio_fname != 'None':
            print(f'There is a video input, but the audio input was also. please select only one.')
            raise NotImplementedError
        else:
            # extract audio from video
            if args.cut_vid_init == None:
                args.cut_vid_init = 0
            if args.cut_vid_end == None:
                args.cut_vid_end = 10

            temp_out_fname = 'demo_items/output_cut_vid.mp4'
            args.audio_fname = 'demo_items/output_audio.wav'
                
            procs1 = ['ffmpeg', '-i', args.input_fname, '-ss', f'{args.cut_vid_init}', '-t', f'{args.cut_vid_end}', '-c', 'copy', temp_out_fname, '-y']
            procs2 = ['ffmpeg', '-i', temp_out_fname, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', args.audio_fname, '-y']

            print(f'process1 for video cutting: {procs1}')
            print(f'process2 for audio parsing: {procs2}')
            
            subprocess.run(procs1)
            subprocess.run(procs2)
    else:
        # if the video does not exist, set the input name as a input audio name.
        if args.audio_fname == 'None':
            print(f'The video and audio does not selected. Please use arguments for that.')
            raise NotImplementedError
            
    # prepare input audio

    # For the galaxy S22, the sampling frequency is 44.1kHz
    samplingFrequency = args.data.samplingFrequency #16000
    frameLength = args.data.frameLength
    #frameShift = int(args.data.frameShift * args.data.fps_control_ratio) #512 # for 83.28 frames, 481 is 
    frameShift = args.data.frameShift #512 # for 83.28 frames, 481 is 
    order = args.data.order
    alpha = args.data.alpha
    stage = args.data.stage
    n_mgc = order + 1

    """ 
    sox input.wav output_final.wav \
    noisered noise.prof 0.21 \
    equalizer 100 0.5q -10 equalizer 1000 0.7q +5 equalizer 5000 0.6q +2 \
    compand 0.3,1 6:-70,-60,-20 -5 -90 0.2 \
    reverb 10 50 100 \
    pitch 50
    """

    if args.data.feature_mode == 'mgclsp':
        raise NotImplementedError
    elif args.data.feature_mode == 'raw':
        raise NotImplementedError
    elif args.data.feature_mode == 'melspectogram':
        # mri image to audio....
        # how is it possible?
        output_wav = args.audio_fname.replace('.wav', '_convert.wav')
        make_tts_like_ver2(args.audio_fname, output_wav)

        audio, sr = torchaudio.load(output_wav)

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=samplingFrequency,  # Sampling rate of the audio
            n_fft=frameLength,         # Frame length
            hop_length=frameShift,     # Frame shift
            n_mels=args.model.in_feat           # Number of Mel bands
        )
        to_db = T.AmplitudeToDB()
        
        mel_spec = mel_spectrogram(audio)
        mel_spec_db = to_db(mel_spec).squeeze()

        audio_min = -80
        audio_max = 20

        audio = (mel_spec_db - audio_min) / (audio_max - audio_min)
        
        # melspectrogram visualization

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(audio.cpu().detach().numpy(), sr=int(sr//2), x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig('demo_items/viz_melspectro.png')
    else:
        raise NotImplementedError
    
    # load model
    _, _, model, _, _ = load_model(args, args.model.in_feat, frame_H, frame_W, device)
    
    model.eval()
    model = model.to(device)
    
    _, audio = data_batchify(audio.T.unsqueeze(0).to(device), lookback=args.data.lookback, fps_control_ratio=args.data.fps_control_ratio)
    
    n_frames = audio.shape[0]
    frame_length = frameShift / samplingFrequency # idk why, the sampling frequency should be doubled
    fps = 1 / frame_length / args.data.fps_control_ratio

    print(f'total_nframes: {n_frames}')
    print(f'total fps: {fps:.4f}')
    print(f'total video length: {n_frames / fps:.4f}')

    if args.model.use_prev_frame:
        # if you use option for opt.use_prev_frame, u need to use initial frame to inference
        if args.dataset_type == '75-speaker':
            random_video = load_video('demo_items/sub051_2drt_07_grandfather1_r1_video.avi')
        elif args.dataset_type == 'timit':
            random_video = load_video('demo_items/usctimit_mri_m1_011_015_withaudio.avi')
        else:
            raise NotImplementedError
        
        init_vid = random_video.transpose(-1, 0, 1)[10]
        init_vid = torch.from_numpy(init_vid).cuda()
        
        temp_vid_list = list()
        
        for proc_idx, temp_audio in enumerate(tqdm(audio)):
            with torch.no_grad():
                if proc_idx == 0:
                    temp_pred = model(temp_audio.unsqueeze(0), init_vid.unsqueeze(0)).view(frame_W, frame_H).cpu().detach()
                else:
                    prev_vid = temp_vid_list[-1].unsqueeze(0).cuda()
                    temp_pred = model(temp_audio.unsqueeze(0), prev_vid).view(frame_W, frame_H).cpu().detach()
                temp_vid_list.append(temp_pred)

        pred = torch.stack(temp_vid_list).numpy() * 255
    else:
        # put into the model and process the results
        with torch.no_grad():
            pred = model(audio.squeeze()).view(n_frames, frame_W, frame_H).cpu().detach().numpy() * 255
        
    pred = pred.astype(np.uint8)
   
    print(f'Prediction complete')
    # save video
    # Define video parameters
    output_file = args.audio_fname.replace('.wav', '.avi')  # Output video file name
    #fps = 30  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec (you can use other codecs like 'XVID', 'MJPG', etc.)
    frame_size = (frame_W, frame_H)  # Frame size

    # Create VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # Convert and write frames to video
    for i in range(n_frames):
        # Convert single-channel grayscale to 3-channel BGR
        bgr_frame = cv2.cvtColor(pred[i], cv2.COLOR_GRAY2BGR)
        out.write(bgr_frame)

    # Release the VideoWriter object
    out.release()

    if args.concat_vid:
        print(f'Start concatenation of given video and 2drt MRI saggital views')

        # Load the two videos
        video1 = VideoFileClip(temp_out_fname)
        video2 = VideoFileClip(output_file)

        # cut the first lookback * fps_control_ratio
        # Extract frames as a list
        frames = list(video1.iter_frames())

        # Select frames (e.g., from frame 10 to the end)
        selected_frames = frames[int(args.data.lookback * args.data.fps_control_ratio):]  # Adjust this slice as needed

        # Create a new video clip from the selected frames
        video1_temp = ImageSequenceClip(selected_frames, fps=video1.fps)

        # Calculate the duration of the new video clip in seconds
        frame_count = len(selected_frames)
        duration = frame_count / video1.fps

        # Extract the audio segment matching the selected frames
        # Calculate start and end time in seconds
        start_time = 10 / video1.fps  # The time corresponding to the 10th frame
        end_time = start_time + duration
        audio_segment = video1.audio.subclip(start_time, end_time)

        # Set the audio to the new video clip
        video1_temp = video1_temp.set_audio(audio_segment)
        video1 = video1_temp
        
        # Resize videos to have the same height (necessary for horizontal stacking)
        video2 = video2.resize(height=video1.h)

        # Stack videos horizontally
        final_video = clips_array([[video1, video2]])

        # Save the output
        final_video.write_videofile("demo_items/output_concat.mp4")
    else:
        print(f"Embedding audio to video with ffmpeg")

        # Paths to your input files
        input_video = output_file
        input_audio = args.audio_fname
        end_comp = input_audio.split('/')[-1]
        new_end_comp = 'trimmed_' + end_comp
        trimmed_input_audio = input_audio.replace(end_comp, new_end_comp)
        output_video = input_video.replace('.avi', f'_embed_{args.exp_name}.avi')

        # Get the duration of the video
        #result = subprocess.run(['ffmpeg', '-i', input_video], stderr=subprocess.PIPE, universal_newlines=True)
        #duration_line = [x for x in result.stderr.split('\n') if "Duration" in x][0]
        #duration = duration_line.split(',')[0].split('Duration:')[1].strip()
        # initial frames: 83.28 fps
        # the total cutted frames: 10 * args.data.fps_control_ratio / 83.28

        # the total number of trimmed frames: 10 * args.data.fps_control_ratio
        trimmed_seconds = 10 * args.data.fps_control_ratio / 83.28
        command0 = [
            'ffmpeg', '-y', '-i', input_audio, '-ss', f'{trimmed_seconds}', '-c', 'copy', trimmed_input_audio
        ]

        subprocess.run(command0)
        
        # Embed the audio, looping if necessary
        command = [
            'ffmpeg', '-y', '-stream_loop', '-1', '-i', trimmed_input_audio, '-i', input_video,
            '-shortest', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_video
        ]

        subprocess.run(command)

        print(f"Video saved as {output_video}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mri')
    parser.add_argument('--dataset_type', type=str, default='timit')
    parser.add_argument('--exp_name', type=str, default='exp0000')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--audio_fname', type=str, default='None')
    parser.add_argument('--input_fname', type=str, default='None')
    parser.add_argument('--reset_mgclsp', action='store_true')
    parser.add_argument('--sub_name', type=int, default=0)
    parser.add_argument('--config_name', type=str, default='mri_base_aug_erase')
    parser.add_argument('--select_ckpt_idx', type=int, default=0)
    parser.add_argument('--cut_vid_init', type=int, default=None)
    parser.add_argument('--cut_vid_end', type=int, default=None)
    parser.add_argument('--concat_vid', action='store_true')
    parser.add_argument('--exist_input_vid', action='store_true')
    
    new_args = load_config(parser.parse_args())
    
    logger = set_logger(new_args)
    main(new_args, logger)
