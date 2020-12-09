#!/usr/bin/bash

python benchmark.py --device cpu --n_threads 0 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 32 --rate_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 32 --rate_coding

python benchmark.py --device cpu --n_threads 0 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 64 --rate_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 64 --rate_coding

python benchmark.py --device cpu --n_threads 0 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 128 --rate_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 128 --rate_coding

python benchmark.py --device cpu --n_threads 0 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 32 --temporal_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 32 --temporal_coding

python benchmark.py --device cpu --n_threads 0 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 64 --temporal_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 64 --temporal_coding

python benchmark.py --device cpu --n_threads 0 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 4 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 8 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 12 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 16 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 20 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 24 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 28 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 32 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 36 --batch_size 128 --temporal_coding
python benchmark.py --device cpu --n_threads 40 --batch_size 128 --temporal_coding
