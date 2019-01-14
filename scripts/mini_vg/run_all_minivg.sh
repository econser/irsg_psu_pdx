#!/bin/bash
cd /home/econser/research/irsg_psu_pdx/code

python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonHasBeard.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonHasBeard.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnBench.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnBench.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnHorse.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnHorse.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnSkateboard.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnSkateboard.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingHelmet.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingHelmet.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingSunglasses.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingSunglasses.yml --b TEST_PGM_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PillowOnCouch.yml --b TEST_PGM_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PillowOnCouch.yml --b TEST_PGM_VIZ

python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonHasBeard.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonHasBeard.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnBench.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnBench.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnHorse.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnHorse.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnSkateboard.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonOnSkateboard.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingHelmet.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingHelmet.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingSunglasses.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PersonWearingSunglasses.yml --b TEST_GEO_VIZ
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PillowOnCouch.yml --b TEST_GEO_ENERGY
python irsg_situation.py --cfg /home/econser/research/irsg_psu_pdx/scripts/mini_vg/configs_mvg_PillowOnCouch.yml --b TEST_GEO_VIZ
