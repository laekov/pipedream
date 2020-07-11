ngpus=4
cd ../optimizer

case $TASK in
	vgg31)
		partition_name=vgg16_partitioned_31

		python optimizer_graph_hierarchical.py \
			-f ../profiler/image_classification/profiles/vgg16/graph.txt \
			-n $ngpus \
			--activation_compression_ratio 1 -o $partition_name

		python convert_graph_to_model.py \
			-f $partition_name/gpus=$ngpus.txt \
			-n VGG16Partitioned -a vgg16 \
			-o ../runtime/image_classification/models/vgg16/gpus$ngpus \
			--stage_to_num_ranks 0:3,1:1

		;;

	vgg71)
		partition_name=vgg16_partitioned_71
		ngpus=8

		python optimizer_graph_hierarchical.py \
			-f ../profiler/image_classification/profiles/vgg16/graph.txt \
			-n $ngpus \
			--activation_compression_ratio 1 -o $partition_name

		python convert_graph_to_model.py \
			-f $partition_name/gpus=$ngpus.txt \
			-n VGG16Partitioned -a vgg16 \
			-o ../runtime/image_classification/models/vgg16/gpus$ngpus \
			--stage_to_num_ranks 0:6,1:1,2:1

		;;


	vgg62)
		partition_name=vgg16_partitioned_62
		ngpus=8

		python optimizer_graph_hierarchical.py \
			-f ../profiler/image_classification/profiles/vgg16/graph.txt \
			-n $ngpus \
			--activation_compression_ratio 1 -o $partition_name

		python convert_graph_to_model.py \
			-f $partition_name/gpus=$ngpus.txt \
			-n VGG16Partitioned -a vgg16 \
			-o ../runtime/image_classification/models/vgg16/gpus$ngpus 

		;;


	vggstraight)
		partition_name=vgg16_partitioned_straight

		python optimizer_graph_hierarchical.py \
			-f ../profiler/image_classification/profiles/vgg16/graph.txt \
			-n $ngpus \
			--straight_pipeline \
			--activation_compression_ratio 1 -o $partition_name

		python convert_graph_to_model.py \
			-f $partition_name/gpus=$ngpus.txt \
			-n VGG16PartitionedStraight -a vgg16 \
			-o ../runtime/image_classification/models/vgg16/gpus${ngpus}_straight

		;;

esac


