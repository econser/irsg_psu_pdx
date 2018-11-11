screen -dm './finetune_minivg.sh'
screen -dm 'cd <PYTHON_ROOT>; python <GMM_SCRIPT_FNAME> --c <GMM_CFG_FNAME> --o <MODEL_SHORT_NAME>_gmms.pkl'
