cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
     --imdb mini_vg_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/solver_init.prototxt \
     --iter 0
