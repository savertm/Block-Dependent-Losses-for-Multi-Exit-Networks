ulimit -n 10000

#conventional
#python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_conventional_5 --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 0,1 -j 16 --num_data 500 --seed 2023

#overlap 2 with final
# python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap2_withfinal_5 --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 4,5,6,7 -j 16 --method overlap --include_final True --overlap 2 --seed 2024 --num_data 1200

#overlap 2 without final
python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap2_withoutfinal_5 --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 2,3 -j 16 --method overlap --include_final False --overlap 2 --seed 2022 --num_data 300

#######################################################################

#Self-distillation

#conventional
#python3 main_block5_imagente_distil.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_conventional_self_distil_5 --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 0,1 -j 16 --num_data 1200 --seed 2021 --self_distillation True --distill_co 0.35 --T 1.5


#overlap 2 with final
#python3 main_block5_imagente_distil.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap2_withfinal_self_distil_5 --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 4,5,6,7 -j 16 --method overlap --include_final True --overlap 2 --seed 2021 --num_data 1200 --self_distillation True --distill_co 0.35 --T 1.5

#overlap 2 without final
#python3 main_block5_imagente_distil.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap2_withoutfinal_self_distil_5 --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --use-valid --gpu 4,5,6,7 -j 16 --method overlap --include_final False --overlap 2 --seed 2021 --num_data 1200 --self_distillation True --distill_co 0.35 --T 1.5
















###################################################################################
#No use of validation

#python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_conventional_5_noval --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --gpu 2,3 -j 16 --num_data 200 --seed 2021

#python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap3_withfinal_5_noval --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --gpu 0,1 -j 16 --method overlap --include_final True --overlap 3 --seed 2021 --num_data 200

#python3 main_block5_imagenet.py --data-root /drive2/data/imagenet --data ImageNet --save result_imagenet_overlap2_withfinal_5_noval --arch msdnet_overlap_imagenet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --gpu 4,5,6,7 -j 16 --method overlap --include_final True --overlap 2 --seed 2022 --num_data 200




