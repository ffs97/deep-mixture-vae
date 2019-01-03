CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=2  --n_epochs=300
CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=4  --n_epochs=300
CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=8  --n_epochs=300
CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=12 --n_epochs=300
CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=15 --n_epochs=300
CUDA_VISIBLE_DEVICES=2 python -B train.py --pretrain_epochs_vae 0 --pretrain_epochs_gmm 0 --model dmoe --classification --n_experts=20 --n_epochs=300
