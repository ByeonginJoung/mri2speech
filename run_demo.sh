export CUDA_VISIBLE_DEVICES=0

#NORM_AUDIO_FNAME=demo_items/test0000.wav
#NORM_AUDIO_FNAME=demo_items/sub051_2drt_07_grandfather1_r1_video.wav
#NORM_AUDIO_FNAME=/hdd4tb_00/dataset/mri_data/sub051/2drt/video/sub051_2drt_07_grandfather1_r1_video.wav
#NORM_AUDIO_FNAME=demo_items/sub053_2drt_10_northwind2_r1_video.wav
#NORM_AUDIO_FNAME=demo_items/sub052_2drt_07_grandfather1_r1_video.wav

# for the song, instruments such as drum, guitar
# are considered, the generated audio was not proper

#NORM_AUDIO_FNAME=demo_items/F_mat_song.wav
#NORM_AUDIO_FNAME=demo_items/steve_jobs_iphone.wav
#NORM_AUDIO_FNAME=demo_items/cnn_news00.wav
INPUT_FNAME=demo_items/cnn_news00.mp4

PRINT_0='========================================='

SUB_N=M2
CONF_NAME=mri_melspectogram_baseline_ver0004_$SUB_N
EXP_NAME=lstm_msessim_256_$SUB_N_$CONF_NAME

echo $PRINT_0
echo $PRINT_0
echo $EXP_NAME
echo $PRINT_0
echo $PRINT_0

python demo.py --dataset mri --exp_name $EXP_NAME --input_fname $INPUT_FNAME --config_name $CONF_NAME \
       --select_ckpt_idx 6 --exist_input_vid --cut_vid_init 0 --cut_vid_end 30 --concat_vid

# for single input audio

AUDIO_FNAME=demo_items/test0000.wav

python demo.py --dataset mri --exp_name $EXP_NAME --audio_fname $AUDIO_FNAME --config_name $CONF_NAME \
       --select_ckpt_idx 6
