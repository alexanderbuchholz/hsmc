#!/bin/bash
# run all simulations after another
python run_simulations_server_student.py 'loop' 'test'
python run_simulations_server_normal.py 'loop' 'test'
python run_simulations_server_log_cox.py 'loop' 'test'
python run_simulations_server_logit.py 'loop' 'test'
python run_simulations_server_probit.py 'loop' 'test'

