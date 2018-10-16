#!/bin/bash
cd <PYTHON_ROOT>

python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_PGM_ENERGY

python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_GEO_ENERGY

python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_BF_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_BF_ENERGY
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_BF_ENERGY



python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_PGM_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_PGM_VIZ

python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_GEO_VIZ

python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b POSTEST_BF_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b HARDNEG_BF_VIZ
python irsg_situation.py --cfg ./configs/<MODEL_SHORT_NAME>_configs.yml --b FULLNEG_BF_VIZ
