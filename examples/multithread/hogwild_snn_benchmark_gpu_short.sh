#!/usr/bin/bash

python benchmark.py --device gpu --n_threads 0 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 1 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 2 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 3 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 4 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 5 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 6 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 7 --batch_size 128 --rate_coding
python benchmark.py --device gpu --n_threads 8 --batch_size 128 --rate_coding

python benchmark.py --device gpu --n_threads 0 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 1 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 2 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 3 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 4 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 5 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 6 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 7 --batch_size 128 --temporal_coding
python benchmark.py --device gpu --n_threads 8 --batch_size 128 --temporal_coding
