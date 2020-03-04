#!/bin/bash

Dir_540=$1
QP=$2
Dir_dest=$3
Dir_temp=$4 #Make sure this directory exists


if [[ $QP -ne 0 ]]
then
mkdir ${Dir_dest}/QP${QP}
mkdir ${Dir_dest}/QP${QP}/train
mkdir ${Dir_dest}/QP${QP}/test

else
mkdir ${Dir_dest}/RAW540
mkdir ${Dir_dest}/RAW540/train
mkdir ${Dir_dest}/RAW540/test
fi

for vid in $Dir_540/train/*.mov
do
    vid_name_ext=$(basename -- "$vid")
    vid_ext="${vid_name_ext##*.}"
	  vid_name="${vid_name_ext%.*}"

    if [[ $QP -eq 0 ]]
    then
        echo "Extracting RAW frames from $vid_name..."
        ffmpeg -i $vid ${Dir_dest}/RAW540/train/${vid_name}_%04d.png #2>&-
    else
        dest_vid=${Dir_temp}/${vid_name}.mov
        echo "converting $vid_name_ext with QP=$QP..."
        ffmpeg -i $vid -c:v libx264 -qp $QP -pix_fmt yuv420p -y -strict -2 $dest_vid #2>&-
        echo "Extracting frames from $vid_name with QP=$QP..."
        ffmpeg -i $dest_vid ${Dir_dest}/QP${QP}/train/${vid_name}_%04d.png #2>&-
        rm $dest_vid
    fi
done

for vid in $Dir_540/test/*.mov
do
    vid_name_ext=$(basename -- "$vid")
    vid_ext="${vid_name_ext##*.}"
	  vid_name="${vid_name_ext%.*}"

    if [[ $QP -eq 0 ]]
    then
        echo "Extracting RAW frames from $vid_name..."
        ffmpeg -i $vid ${Dir_dest}/RAW540/test/${vid_name}_%04d.png #2>&-
    else
        dest_vid=${Dir_temp}/${vid_name}.mov
        echo "converting $vid with QP=$QP..."
        ffmpeg -i $vid -c:v libx264 -qp $QP -pix_fmt yuv420p -y -strict -2 $dest_vid #2>&-
        echo "Extracting frames from $vid_name with QP=$QP..."
        ffmpeg -i $dest_vid ${Dir_dest}/QP${QP}/test/${vid_name}_%04d.png #2>&-
        rm $dest_vid
    fi
done


