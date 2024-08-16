#!/bin/bash
source /home/hunt-stokes/rattus_novus.sh

# Log the current directory and environment variables
echo "Current directory: $(pwd)" > /home/hunt-stokes/bayesian_optimisation/terminate.log
echo "Environment variables: $(env)" >> /home/hunt-stokes/bayesian_optimisation/terminate.log

# Execute Python script and log output
python3 /home/hunt-stokes/bayesian_optimisation/terminate_loop.py >> /home/hunt-stokes/bayesian_optimisation/terminate.log 2>&1

# Exit with the status of the Python script
exit $?
