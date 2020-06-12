export IMAGENET_DIR=/home/laekov/dataset/imagenet
export ngpus=4
export ncpus=4

if [ .$lr = . ]
then
	lr=0.1
fi

if [ .$offset = . ]
then
	offset=0
fi

export CUDA_VISIBLE_DEVICES=$(seq -s , $offset $(expr $offset + 3))
export TASK_MASTER_PORT=$(expr 10210 + $offset)

train() {
	myrank=$1

	firstcpu=$(expr \( $offset + $myrank \) \* $ncpus)
	lastcpu=$(expr $firstcpu + $ncpus - 1)
	cd ../runtime/image_classification

	numactl -C $firstcpu-$lastcpu \
		python main_with_runtime.py \
		--module models.vgg16.gpus=$ngpus -b 64 \
		--data_dir $IMAGENET_DIR \
		--lr $lr \
		--rank $myrank --local_rank $myrank \
		--master_addr 127.0.0.1 \
		--config_path models/vgg16/gpus=$ngpus/hybrid_conf.json \
		--distributed_backend gloo
}

pids=""

endrun() {
	echo Ending run $pids
	for pid in $pids
	do
		pkill -9 -P $pid
	done
}

n=4

echo >pids

trap endrun SIGINT

for ((i=0;i<$n;++i))
do
	train $i & pids="$pids $!"
done

wait

