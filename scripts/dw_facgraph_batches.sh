#!/bin/bash
cd /home/econser/research/code
python irsg_situation.py --cfg irsg.yml --b DW_FULLPOS_BATCHONLY
python irsg_situation.py --cfg irsg.yml --b DW_HARDNEG_BATCHONLY
python irsg_situation.py --cfg irsg.yml --b DW_FULLNEG_BATCHONLY
python irsg_situation.py --cfg irsg.yml --b DW_ORIGNEG_BATCHONLY
