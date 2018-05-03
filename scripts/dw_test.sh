cd /home/econser/School/Thesis/external/py-faster-rcnn/
time ./tools/test_net.py \
     --net /home/econser/School/research/models/model_weights/vgg_cnn_m_1024_faster_rcnn_iter_10000.caffemodel \
     --imdb dogwalking_test \
     --cfg /home/econser/School/research/models/model_definitions/dog_walking_2/config.yml \
     --def /home/econser/School/research/models/model_definitions/dog_walking_2/faster_rcnn_end2end/test.prot\
otxt
