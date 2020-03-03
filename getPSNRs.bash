#!/bin/bash

for f in QP*
do
	echo "Running test for $f"
	
	rm SR/*
	python test.py --input_noisy ./$f/test/ --input_noNoise ./RAW/test/ --model_dir ./model/$f/model_best_weights.h5

	python PSNR_gen.py ${f}_enhanced SR ./PSNRs ./RAW/test
	python PSNR_gen.py ${f}_raw ./$f/test ./PSNRs ./RAW/test

done

