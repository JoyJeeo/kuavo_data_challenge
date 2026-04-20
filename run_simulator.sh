#!/bin/bash
# PYTHON_HOME=/home/zhangyutao/Software/miniconda3/envs/mac_kdc_icra_sim/bin/python
source /home/zhangyutao/Software/miniconda3/etc/profile.d/conda.sh
conda activate mac_kdc_icra_sim
exec python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml