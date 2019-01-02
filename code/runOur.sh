CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=1 --featLearn
CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=2 --featLearn
CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=4 --featLearn
CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=8 --featLearn
CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=12 --featLearn
CUDA_VISIBLE_DEVICES=1 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dvmoe --classification --n_experts=16 --featLearn
