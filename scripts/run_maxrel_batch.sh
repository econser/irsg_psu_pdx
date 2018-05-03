#!/bin/bash
cd /home/econser/research/code
python irsg_situation.py --cfg maxrel_configs.yml --b DW_POSTEST_PGM_ENERGY
#python irsg_situation.py --cfg maxrel_configs.yml --b STAN_DW_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg maxrel_configs.yml --b PP_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg maxrel_configs.yml --b HS_POSTEST_PGM_ENERGY
# viz
#python irsg_situation.py --cfg maxrel_configs.yml --b DW_POSTEST_PGM_VIZ
#python irsg_situation.py --cfg maxrel_configs.yml --b STAN_DW_POSTEST_PGM_VIZ
#python irsg_situation.py --cfg maxrel_configs.yml --b PP_POSTEST_PGM_VIZ
#python irsg_situation.py --cfg maxrel_configs.yml --b HS_POSTEST_PGM_VIZ
