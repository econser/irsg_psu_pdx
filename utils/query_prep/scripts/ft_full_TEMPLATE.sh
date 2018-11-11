cd <FRCNN_DIR>
time ./tools/train_net.py --gpu 0 \
     --weights <FRCNN_DIR>/output/default/train/<MODEL_NAME>_<MODEL_ARCH>_faster_rcnn_iter_0.caffemodel \
     --imdb <MODEL_NAME>_train \
     --cfg <MODEL_DEF_DIR>/<MODEL_NAME>/config.yml \
     --solver <MODEL_DEF_DIR>/<MODEL_NAME>/faster_rcnn_end2end/solver.prototxt \
     --iter <ITERATIONS>
