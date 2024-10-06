export CUDA_VISIBLE_DEVICES=0

#SCENE_NAME=("004" "006" "008" "009" "013" "015" "019" "020" "022" "023" "028" "030" "032" "033" "034" "035" "037" "038" "040" "043" "052" "057" "061" "063" "069" "073")

# for 75-speaker dataset
#SCENE_NAME=("013" "022" "030" "051" "052")
#SCENE_NAME=("051")

# for timit dataset
SCENE_NAME=("053")

for SUB_N in "${SCENE_NAME[@]}"
do
    echo $SUB_N
    # for 75-speaker dataset
    CONF_NAME=mri_melspectogram_ver0004_scene$SUB_N
    EXP_NAME=$SUB_N_$CONF_NAME

    python train.py --dataset mri --exp_name $EXP_NAME --sub_name $SUB_N --config_name $CONF_NAME --dataset_type '75-speaker'
    
    # for 75-speaker dataset
    CONF_NAME=mri_melspectogram_ver0005_scene$SUB_N
    EXP_NAME=$SUB_N_$CONF_NAME

    python train.py --dataset mri --exp_name $EXP_NAME --sub_name $SUB_N --config_name $CONF_NAME --dataset_type '75-speaker'
    
    # for timit dataset
    #CONF_NAME=mri_melspectogram_ver0000_$SUB_N
    #EXP_NAME=$SUB_N_$CONF_NAME

    #python train.py --dataset mri --exp_name $EXP_NAME --sub_name $SUB_N --config_name $CONF_NAME --dataset_type 'timit'
done
