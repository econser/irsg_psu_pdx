#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLPOS_BOXES
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_HARDNEG_BOXES
python irsg_situation.py --cfg ./configs/pwg_configs.yml --b PWG_FULLNEG_BOXES
