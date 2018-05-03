cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights /home/econser/usr/py-faster-rcnn/output/default/train/vgg_cnn_m_1024_faster_rcnn_iter_0.caffemodel \
     --imdb pingpong_train \
     --cfg /home/econser/research/models/model_definitions/pingpong/config.yml \
     --solver /home/econser/research/models/model_definitions/pingpong/faster_rcnn_end2end/solver.prototxt \
     --iter 50000
