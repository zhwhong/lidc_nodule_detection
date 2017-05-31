#!/usr/bin/env bash

dcm_dir="/baina/sda1/data/lidc/DOI/"
out="/baina/sda1/data/lidc_matrix/TXT/*.pkl"
for s in `ls $dcm_dir`; do
	if [ -d $dcm_dir$s ];
	then
		for a in `ls $dcm_dir$s/`; do
			for b in `ls $dcm_dir$s/$a/`;do
				# echo $dcm_dir$s/$a/$b/
				python nodule2.py -s $dcm_dir$s/$a/$b/
				# rm ./output/*.pkl
				rm $out
			done
		done
	fi
done
