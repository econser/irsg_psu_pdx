cd /home/econser/usr/py-faster-rcnn
time ./tools/test_net.py --gpu 1 \
     --net /home/econser/research/irsg_psu_pdx/models/model_weights/minivg_500k.caffemodel \
     --imdb mini_vg_test \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/config.yml \
     --def /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/test.prototxt
