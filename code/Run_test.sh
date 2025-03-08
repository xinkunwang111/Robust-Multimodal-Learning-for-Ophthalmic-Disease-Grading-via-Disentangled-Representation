CUDA_VISIBLE_DEVICES=6 python fusion_test.py \
                         --folder "folder0"\
                         --mode "train&test"\
                         --model_name "IMDR"\
                         --model_base "transformer"\
                         --dataset "dr2"\
                         --condition "noise"  --name "10008_enhanced_rate_04"  --Condition_G_Variance 0\
                         --lr 1e-5\
                         --checkpoint 
