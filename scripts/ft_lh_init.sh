cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 1 \
     --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
     --imdb leadinghorse_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/leadinghorse/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/leadinghorse/faster_rcnn_end2end/solver.prototxt \
     --iter 0
