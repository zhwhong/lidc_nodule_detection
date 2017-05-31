#!/usr/bin/env bash

dcm_dir="/baina/sda1/data/lidc/DOI/"
for s in `ls $dcm_dir`; do
	if [ -d $dcm_dir$s ];
	then
		for a in `ls $dcm_dir$s/`; do
			for b in `ls $dcm_dir$s/$a/`;do
				# echo $dcm_dir$s/$a/$b/
				python write_bin_file.py -s $dcm_dir$s/$a/$b/ -d /baina/sda1/data/lidc_matrix/tmp/
			done
		done
	fi
done
