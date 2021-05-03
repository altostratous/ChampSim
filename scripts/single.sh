#!/bin/bash
PYTHONPATH=/home/aasgarik/projects/def-karthikp/aasgarik/UBC_ISCA21_Comp/OurChampSim
source ../dnnfault/venv/bin/activate
env
echo "$@"
exec "$@"
