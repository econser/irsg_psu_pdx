cd /home/econser/usr/py-faster-rcnn
time ./tools/test_net.py --gpu 0 \
     --net /home/econser/usr/py-faster-rcnn/output/default/train/mini_vg_vgg_cnn_m_1024_faster_rcnn_iter_1000.caffemodel \
     --imdb mini_vg_test \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/config.yml \
     --def /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/test.prototxt
