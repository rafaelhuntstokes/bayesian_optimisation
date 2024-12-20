#!/bin/bash

rm main*
rm dag_templates/tune_subdag.dag.*
rm rat_logs/*.log
rm plots/*.pdf
rm plots/*.png
rm plots/tres_comparison/*.png
rm measured_points/*.npy
rm dag_running/*
rm condor_logs/residuals_errors/*.err
rm condor_logs/residuals_outputs/*.out
rm condor_logs/residuals_logs/*.log
rm condor_logs/select_errors/*.err
rm condor_logs/select_outputs/*.out
rm condor_logs/select_logs/*.log
rm condor_logs/sim_errors/*.err
rm condor_logs/sim_outputs/*.out
rm condor_logs/sim_logs/*.log