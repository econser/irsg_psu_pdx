#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code

python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_PGM_ENERGY

python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_GEO_ENERGY

python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_BF_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_BF_ENERGY
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_BF_ENERGY



python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_PGM_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_PGM_VIZ

python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_GEO_VIZ

python irsg_situation.py --cfg ./configs/configs_minivg.yml --b POSTEST_BF_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b HARDNEG_BF_VIZ
python irsg_situation.py --cfg ./configs/configs_minivg.yml --b FULLNEG_BF_VIZ
