cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights /home/econser/usr/py-faster-rcnn/output/default/train/mini_vg_vgg_cnn_m_1024_faster_rcnn_iter_0.caffemodel \
     --imdb mini_vg_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/solver_full.prototxt \
     --iter 50000
