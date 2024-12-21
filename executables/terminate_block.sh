#!/bin/bash
source /home/hunt-stokes/rattus_novus.sh

python3 /home/hunt-stokes/bayesian_optimisation/terminate_block.py >> /home/hunt-stokes/bayesian_optimisation/terminate.log 2>&1

# Exit with the status of the Python script
exit $?