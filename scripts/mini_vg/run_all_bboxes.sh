#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code

python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonHasBeard.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnBench.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnHorse.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnSkateboard.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingHelmet.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingSunglasses.yml --b GEN_BBOXES
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PillowOnCouch.yml --b GEN_BBOXES
