cd /home/econser/School/Thesis/external/py-faster-rcnn/
time ./tools/test_net.py \
     --net /home/econser/School/research/models/model_weights/pingpong_frcn_50k.caffemodel \
     --imdb pingpong_test \
     --cfg /home/econser/School/research/models/model_definitions/pingpong/config.yml \
     --def /home/econser/School/research/models/model_definitions/pingpong/faster_rcnn_end2end/test.prototxt
