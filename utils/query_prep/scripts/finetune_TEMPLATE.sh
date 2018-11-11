./ft_init_<MODEL_SHORT_NAME>.sh
./ft_full_<MODEL_SHORT_NAME>.sh

mv <FRCNN_DIR>/output/default/train/<MODEL_NAME>_<MODEL_ARCH>_faster_rcnn_iter_<ITERATIONS>.caffemodel <MODEL_WEIGHT_DIR>/<WEIGHT_FNAME>_<ITERATIONS>.caffemodel
