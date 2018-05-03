cd /home/econser/usr/py-faster-rcnn
time ./tools/test_net.py --gpu 0 \
     --net /home/econser/usr/py-faster-rcnn/output/default/train/vgg_cnn_m_1024_faster_rcnn_iter_50000.caffemodel \
     --imdb dogwalking_test \
     --cfg /home/econser/research/models/model_definitions/dog_walking_2/config.yml \
     --def /home/econser/research/models/model_definitions/dog_walking_2/faster_rcnn_end2end/test.prototxt
