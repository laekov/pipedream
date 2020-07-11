export IMAGENET_DIR=/home/laekov/dataset/imagenet
if [ .$ngpus = . ]
then
	export ngpus=4
fi

export ncpus=4

if [ .$lr = . ]
then
	lr=0.01
fi

if [ .$OFFSET = . ]
then
	OFFSET=0
fi

export CUDA_VISIBLE_DEVICES=$(seq -s , $OFFSET $(expr $OFFSET + $ngpus))
export TASK_MASTER_PORT=$(expr 10210 + $OFFSET)

if [ .$MODEL = . ] || [ .$CONF = . ]
then
	echo 'No model or conf specified'
	exit
fi

if [ .$CONF = '.dp_conf' ]
then
	BACKEND=nccl
else
	BACKEND=gloo
fi

train() {
	myrank=$1

	firstcpu=$(expr \( $OFFSET + $myrank \) \* $ncpus)
	lastcpu=$(expr $firstcpu + $ncpus - 1)
	cd ../runtime/image_classification

	case $TASK in
		normal)
			numactl -C $firstcpu-$lastcpu \
				python main_with_runtime.py \
				--module models.vgg16.$MODEL -b 64 \
				--data_dir $IMAGENET_DIR \
				--lr $lr \
				--rank $myrank --local_rank $myrank \
				--no_input_pipelining \
				--master_addr 127.0.0.1 \
				--config_path models/vgg16/$MODEL/$CONF.json \
				--distributed_backend $BACKEND

			;;

		dp)
			export CUDA_VISIBLE_DEVICES=$(expr $myrank + $OFFSET)
			# export RANK=$myrank
			numactl -C $firstcpu-$lastcpu \
				python main.py \
				--module models.vgg16.$MODEL -b 64 \
				--data_dir $IMAGENET_DIR \
				--lr $lr \
				--dist-url tcp://127.0.0.1:13333 \
				--rank $myrank \
				--world-size $ngpus \
				--dist-backend gloo

			;;

		timeline)
			nmb=63
			if [ $myrank -lt 3 ] && [ $MODEL = "gpus=$ngpus" ]
			then
				nmb=$(expr $nmb / 3)
			fi
			numactl -C $firstcpu-$lastcpu \
				python main_with_runtime.py \
				--get-timeline \
				--module models.vgg16.$MODEL -b 64 \
				--num_minibatches=$nmb \
				--epochs=1 \
				--data_dir $IMAGENET_DIR \
				--lr $lr \
				--rank $myrank --local_rank $myrank \
				--master_addr 127.0.0.1 \
				--config_path models/vgg16/$MODEL/$CONF.json \
				--distributed_backend $BACKEND

			;;

		*)
			if [ $myrank = 0 ]
			then
				echo "No TASK is specified"
			fi
			;;
	esac
}

pids=""

endrun() {
	echo Ending run $pids
	for pid in $pids
	do
		pkill -9 -P $pid
	done
}

echo >pids

trap endrun SIGINT

echo Using distributed backend $BACKEND

for ((i=0;i<$ngpus;++i))
do
	train $i & pids="$pids $!"
done

wait

