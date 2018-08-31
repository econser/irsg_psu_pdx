cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 1 \
     --weights /home/econser/usr/py-faster-rcnn/output/default/train/personwearingglasses_vgg_cnn_m_1024_faster_rcnn_iter_0.caffemodel \
     --imdb personwearingglasses_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/personwearingglasses/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/personwearingglasses/faster_rcnn_end2end/solver.prototxt \
     --iter 50000
