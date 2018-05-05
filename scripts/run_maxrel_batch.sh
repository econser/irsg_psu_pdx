#!/bin/bash
cd /home/econser/irsg_psu_pdx/research/code
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b DW_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b STAN_DW_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b PP_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b HS_POSTEST_PGM_ENERGY
# viz
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b DW_POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b STAN_DW_POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b PP_POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/maxrel_configs.yml --b HS_POSTEST_PGM_VIZ
