#!/usr/bin/bash

python benchmark.py --device cpu --n_threads 0 --n_layers 100 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 0 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 0 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 0 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 0 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 0 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device cpu --n_threads 0 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 0 --n_layers 1000 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 0 --n_layers 100 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 0 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 0 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 0 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 0 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 0 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 0 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 0 --n_layers 1000 --n_neurons_per 1000 --recurrent

python benchmark.py --device cpu --n_threads 4 --n_layers 100 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 4 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 4 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 4 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 4 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 4 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device cpu --n_threads 4 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 4 --n_layers 1000 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 4 --n_layers 100 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 4 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 4 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 4 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 4 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 4 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 4 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 4 --n_layers 1000 --n_neurons_per 1000 --recurrent

python benchmark.py --device cpu --n_threads 8 --n_layers 100 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 8 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 8 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 8 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 8 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 8 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device cpu --n_threads 8 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 8 --n_layers 1000 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 8 --n_layers 100 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 8 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 8 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 8 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 8 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 8 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 8 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 8 --n_layers 1000 --n_neurons_per 1000 --recurrent

python benchmark.py --device cpu --n_threads 16 --n_layers 100 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 16 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 16 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device cpu --n_threads 16 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device cpu --n_threads 16 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 16 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device cpu --n_threads 16 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device cpu --n_threads 16 --n_layers 1000 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 16 --n_layers 100 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 16 --n_layers 100 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 16 --n_layers 1000 --n_neurons_per 100
python benchmark.py --device gpu --n_threads 16 --n_layers 1000 --n_neurons_per 1000
python benchmark.py --device gpu --n_threads 16 --n_layers 100 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 16 --n_layers 100 --n_neurons_per 1000 --recurrent
python benchmark.py --device gpu --n_threads 16 --n_layers 1000 --n_neurons_per 100 --recurrent
python benchmark.py --device gpu --n_threads 16 --n_layers 1000 --n_neurons_per 1000 --recurrent
