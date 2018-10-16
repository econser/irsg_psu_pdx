./run_<MODEL_SHORT_NAME>_init.sh
./run_<MODEL_SHORT_NAME>_full.sh

mv <FRCNN_DIR>/output/default/train/<MODEL_NAME>_<MODEL_ARCH>_faster_rcnn_iter_<ITERATIONS>.caffemodel <MODEL_WEIGHT_DIR>/<WEIGHT_FNAME>_<ITERATIONS>.caffemodel
