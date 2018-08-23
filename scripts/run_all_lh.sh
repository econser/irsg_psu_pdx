#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
#
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_PGM_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_PGM_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_PGM_ENERGY
#
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_GEO_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_GEO_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_GEO_ENERGY
#
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_BF_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_BF_ENERGY
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_BF_ENERGY
#
#
#
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_PGM_VIZ
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_PGM_VIZ
#
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_GEO_VIZ
#
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_POSTEST_BF_VIZ
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_HARDNEG_BF_VIZ
#python irsg_situation.py --cfg ./configs/lh_configs.yml --b LH_FULLNEG_BF_VIZ
