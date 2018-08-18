#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code
python irsg_situation.py --cfg ./configs/boxes_only.yml --b DW_FULLPOS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b DW_HARDNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b DW_FULLNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b STAN_DW_FULLPOS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b PP_POS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b PP_FULLNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b PP_HARDNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b HS_FULLPOS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b HS_HARDNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b HS_FULLNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_FULLPOS
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_HARDNEG
python irsg_situation.py --cfg ./configs/boxes_only.yml --b LH_FULLNEG
