#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
#
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_HARDNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_FULLNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n1.yml --b DW_FULLNEG_GEO_ENERGY
#
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_HARDNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_FULLNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/dw_configs_n3.yml --b DW_FULLNEG_GEO_ENERGY
