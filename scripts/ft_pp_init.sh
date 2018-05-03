cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
     --imdb pingpong_train \
     --cfg /home/econser/research/models/model_definitions/pingpong/config.yml \
     --solver /home/econser/research/models/model_definitions/pingpong/faster_rcnn_end2end/solver.prototxt \
     --iter 0
