#!/bin/bash  

# python train_pg_pytorch.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na
# python train_pg_pytorch.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --nn_baseline --exp_name sb_rtg_dna_nb

# python train_pg_pytorch.py InvertedPendulum-v1 -n 100 -b 1000 -e 1 -rtg --n_layers=2 --size=64 --exp_name sb_rtg_na_nb
python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 -lr 0.01 -n 100 -b 30000 -e 1 -rtg --n_layers=1 --size=128 --exp_name lb_rtg_na
