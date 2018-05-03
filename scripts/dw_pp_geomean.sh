#!/bin/bash
cd /home/econser/research/code
python irsg_situation.py --cfg irsg.yml --b DW_FULLPOS_GEOMEAN
python irsg_situation.py --cfg irsg.yml --b DW_HARDNEG_GEOMEAN
python irsg_situation.py --cfg irsg.yml --b DW_FULLNEG_GEOMEAN
python irsg_situation.py --cfg irsg.yml --b PP_POS_GEOMEAN
python irsg_situation.py --cfg irsg.yml --b PP_FULLNEG_GEOMEAN
python irsg_situation.py --cfg irsg.yml --b PP_HARDNEG_GEOMEAN
