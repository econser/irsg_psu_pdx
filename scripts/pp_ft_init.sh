cd /home/econser/School/Thesis/external/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
     --imdb dogwalking_train \
     --cfg /home/econser/research/models/model_definitions/dog_walking_2/config.yml \
     --solver /home/econser/research/models/model_definitions/dog_walking_2/faster_rcnn_end2end/solver\
.prototxt \
     --iter 0
