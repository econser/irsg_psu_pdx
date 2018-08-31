#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
#
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_PGM_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_PGM_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_PGM_ENERGY
#
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_GEO_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_GEO_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_GEO_ENERGY
#
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_BF_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_BF_ENERGY
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_BF_ENERGY
#
python -c "import plot_utils as p; p.personwearingglasses_ratk('pgm', ratk_zoom=100); p.personwearingglasses_ratk('geo', ratk_zoom=100); p.personwearingglasses_ratk('brute', ratk_zoom=100)"
#
#
#
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_PGM_VIZ
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_PGM_VIZ
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_PGM_VIZ
#
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_GEO_VIZ
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_GEO_VIZ
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_GEO_VIZ
#
#python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_POSTEST_BF_VIZ
#python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_BF_VIZ
#python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_BF_VIZ
