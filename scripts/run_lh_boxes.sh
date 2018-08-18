#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_FULLPOS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_HARDNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_FULLNEG
