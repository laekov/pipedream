ngpus=4
cd ../optimizer
python optimizer_graph_hierarchical.py -f ../profiler/image_classification/profiles/vgg16/graph.txt -n $ngpus --activation_compression_ratio 1 -o vgg16_partitioned
python convert_graph_to_model.py -f vgg16_partitioned/gpus=$ngpus.txt -n VGG16Partitioned -a vgg16 -o ../runtime/image_classification/models/vgg16/gpus=$ngpus --stage_to_num_ranks 0:3,1:1


