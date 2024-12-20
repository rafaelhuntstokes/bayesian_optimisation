#!/bin/bash
source /data/snoplus3/hunt-stokes/multisite_clean/mc_studies/scripts/jesushchrist/bin/activate

source /home/hunt-stokes/rattus_novus.sh
cd /home/hunt-stokes/bayesian_optimisation
python3 time_residuals.py