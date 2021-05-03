#!/bin/bash

for b in `ls ML-DPC/ChampSimTraces`; do for t in `ls ML-DPC/ChampSimTraces/$b`; do
    if [[ $b != "gap" ]]; then continue; fi
    TRACE_PATH=ML-DPC/ChampSimTraces/$b/$t;
    if [[ `python -c "from random import random; print(random() < 0.1)"` = "True" ]]; then
        for bucket in page ip; do
            for lookahead in 0 4 8; do
                export BUCKET=$bucket
                export LOOKAHEAD=$lookahead
                echo sbatch --export=ALL --mem 1G --time 3:00:00 scripts/single.sh ./ml_prefetch_sim.py run $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-$bucket-$lookahead" --no-base --prefetch "generated/$bucket-$lookahead.txt"
            done
        done
    fi
done; done
