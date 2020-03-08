#!/bin/bash


for qp in 22 18 14 10 6 2
do
	echo "Running all for QP=$qp"
	
	echo "Generating dataset..."
	bash GenDataset.bash /opt/a/shared/movies_540/ $qp /data/shared/VideoEnhancement/Dataset/images/ temp_dir/ &>>log.txt
	echo "Generating dataset done."

	mkdir Experiments/Models/QP${qp}/

	echo "Running training..."
	python train.py --LR_dir /data/shared/VideoEnhancement/Dataset/images/RAW540/train/ --NR_dir /data/shared/VideoEnhancement/Dataset/images/QP${qp}/train/ --model_save_dir Experiments/Models/QP${qp}/ &>>log.txt
	echo "Training is done."

	rm SR/*
	echo "Running testing..."
	python test.py -inr /data/shared/VideoEnhancement/Dataset/images/QP${qp}/test/ -ilr /data/shared/VideoEnhancement/Dataset/images/RAW540/test/ -o ./SR/ -m Experiments/Models/QP${qp}/model_best_weights.h5 &>>log.txt
	echo "Testing is done."

	echo "Finding PSNR..."
	python PSNR_gen.py QP${qp}_enhanced SR/ PSNRs/ /data/shared/VideoEnhancement/Dataset/images/RAW540/test/ &>>log.txt
	python PSNR_gen.py QP${qp}_raw /data/shared/VideoEnhancement/Dataset/images/QP${qp}/test/ PSNRs/ /data/shared/VideoEnhancement/Dataset/images/RAW540/test/ &>>log.txt
	echo "Finding PSNR is done."

	echo "All tasks for QP=${qp} is done."

done

#bash GenDataset.bash /opt/a/shared/movies_540/ 26 /data/shared/VideoEnhancement/Dataset/images/ temp_dir/


