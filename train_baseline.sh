#!/bin/bash -e

DIR=~/dev/Attentive_reader/codes/
SDIR=$DIR/att_reader/
MDIR=/data/lisatmp3/cooijmat/run/batchnorm/attr/
export PYTHONPATH=$DIR:$PYTHONPATH

tg 4 ${SDIR}/train_attentive_reader.py \
    --use_dq_sims 1 --use_desc_skip_c_g 0 --dim 240 --learn_h0 1 --lr 8e-5 --truncate -1 \
    --model "default.npz" --batch_size 64 \
    --optimizer "adam" --validFreq 1000 --model_dir $MDIR --use_desc_skip_c_g 1
