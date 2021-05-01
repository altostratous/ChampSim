#!/bin/bash

for b in `ls ML-DPC/ChampSimTraces`; do for t in `ls ML-DPC/ChampSimTraces/$b`; do
    TRACE_PATH=ML-DPC/ChampSimTraces/$b/$t;
    if [[ `python -c "from random import random; print(random() < 0.1)"` = "True" ]]; then
        echo sbatch --mem 1G --time 3:00:00 scripts/single.sh ./ml_prefetch_sim.py run $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100
    fi
done; done
