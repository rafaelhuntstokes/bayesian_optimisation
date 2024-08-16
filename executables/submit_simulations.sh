#!/bin/bash

JOB_ID=$1
MACRO_NAME=$2
cd rat_logs/
source /home/hunt-stokes/rattus_novus.sh

rm /data/snoplus3/hunt-stokes/bayesian_optimisation_sims/*.root
rat -n 300823 -N 100 -o "/data/snoplus3/hunt-stokes/bayesian_optimisation_sims/output_${JOB_ID}.root" /home/hunt-stokes/bayesian_optimisation/macros/${MACRO_NAME}