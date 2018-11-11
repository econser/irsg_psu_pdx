screen -dm './finetune_<MODEL_SHORT_NAME>.sh'
screen -dm 'cd <PYTHON_ROOT>; python <GMM_SCRIPT_FNAME> --c <GMM_CFG_FNAME> --o <MODEL_SHORT_NAME>_gmms.pkl'
