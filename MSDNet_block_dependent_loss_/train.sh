# 7 Exit

# Conventonal
# python3 main_block7.py --data-root dataset/CIFAR100 --data cifar100 --save result_conventional_7 --arch msdnet --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 5 --seed 0


# overlap 3 without final loss
# python3 main_block7.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withoutfinal_7_overlap3 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 2 --method overlap --include_final False --overlap 3 --seed 2022

# overlap 3 with final loss
# python3 main_block7.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withfinal_7_overlap3 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 2 --method overlap --include_final True --overlap 3 --seed 2024

#local error
# python3 main_block7.py --data-root dataset/CIFAR100 --data cifar100 --save result_local_error_7 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 3 --method local_error --seed 2024


#No overlap
# python3 main_block7.py --data-root dataset/CIFAR100 --data cifar100 --save result_no_overlap_7 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 2 --method no_overlap --seed 2022

###################################################################################################################################################################################################################################################################################

# 7 Exits with self distillation

# Conventional
# python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_conventional_7_self_distil_T25 --arch msdnet --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 0 --seed 2021 --self_distillation True --distill_co 0.35 --T 2.5


# Overlap 3 without final loss
# python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withoutfinal_7_overlap3_self_distil --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 1 --method overlap --include_final False --overlap 3 --seed 2022 --self_distillation True --distill_co 0.25



# Overlap 3 with final loss
# python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withfinal_7_overlap3_self_distil_a03 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 4 --method overlap --include_final True --overlap 3 --seed 2021 --self_distillation True --distill_co 0.3


#local error
# python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_local_error_7_self_distil --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 1 --method local_error --seed 2022 --self_distillation True --distill_co 0.25


#No overlap
# python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_no_overlap_7_self_distil --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 4 --method no_overlap --seed 2021 --self_distillation True --distill_co 0.25


###################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################

# 7 Exits with pretrained distillation

# Conventional
#python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_conventional_7_pretrained_T25 --arch msdnet --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 2 --seed 2021 --self_distillation False --distill_co 0.35 --T 2.5


# Overlap 3 without final loss
#python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withoutfinal_7_overlap3_pretrained --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 7 --method overlap --include_final False --overlap 3 --seed 2022 --self_distillation False --distill_co 0.25

# Overlap 3 with final loss
#python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_overlap_withfinal_7_overlap3_pretrained_T25 --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 5 --method overlap --include_final True --overlap 3 --seed 2021 --self_distillation False --distill_co 0.35 --T 2.5


# Local error
#python3 main_block7_distil.py --data-root dataset/CIFAR100 --data cifar100 --save result_local_error_7_pretrained --arch msdnet_overlap --batch-size 64 --epoch 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 16 --gpu 2 --method local_error --seed 2021 --self_distillation False --distill_co 0.25

#############################################################################################################################################################################################################################################################################
