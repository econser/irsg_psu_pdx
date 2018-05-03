#!/bin/bash
cd /home/econser/research/code
python irsg_situation.py --cfg ./configs/configs.yml --b DW_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b DW_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b DW_FULLNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_FULLNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b PP_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b PP_FULLNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b PP_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b HS_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b HS_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b HS_FULLNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs.yml --b DW_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b DW_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b DW_FULLNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b STAN_DW_FULLNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b PP_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b PP_FULLNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b PP_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b HS_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b HS_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs.yml --b HS_FULLNEG_GEO_VIZ
