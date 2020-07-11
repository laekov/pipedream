IMAGENET_DIR=/home/laekov/dataset/imagenet

cd ../profiler/image_classification
# CUDA_VISIBLE_DEVICES=0 python main.py -a vgg16 -b 64 --data_dir $IMAGENET_DIR
CUDA_VISIBLE_DEVICES=1 python main.py -a resnet50 -b 128  --data_dir $IMAGENET_DIR

