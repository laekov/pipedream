#!/bin/bash

config_name=$1

if [ .$config_name = . ] || ! [ -f "../runtime/image_classification/models/vgg16/gpus=4/$config_name.json" ]
then
	echo """No config specified. Available configs are:
hybrid_2p2s_pipefirst """
	exit 1
fi

export ngpus=8
export MODEL="gpus=4"
export CONF=$config_name
export LOG_NAME=logs/$(date +%m%d-%H%M%S)-$config_name.log

TASK=normal ./run.sh | tee $LOG_NAME
