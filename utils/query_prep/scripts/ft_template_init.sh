cd <FRCNN_DIR>
time ./tools/train_net.py --gpu 0 \
     --weights data/imagenet_models/<MODEL_ARCH>.caffemodel \
     --imdb <MODEL_NAME>_train \
     --cfg <MODEL_DEF_DIR>/<MODEL_NAME>/config.yml \
     --solver <MODEL_DEF_DIR>/<MODEL_NAME>/faster_rcnn_end2end/solver_init.prototxt \
     --iter 0
