cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights /home/econser/usr/py-faster-rcnn/output/default/train/vgg_cnn_m_1024_faster_rcnn_iter_0.caffemodel \
     --imdb dogwalking_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/dog_walking_2/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/dog_walking_2/faster_rcnn_end2end/solver.prototxt \
     --iter 50000
