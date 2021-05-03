#!/bin/bash

$TRACE_TYPE='ChampSimTraces'
$ML_COMMAND=train

mkdir -p generated
for b in `ls ML-DPC/$TRACE_TYPE`; do for t in `ls ML-DPC/$TRACE_TYPE/$b`; do
    if [[ $b != "gap" ]]; then continue; fi
    TRACE_PATH=ML-DPC/$TRACE_TYPE/$b/$t;
    # if [[ `python -c "from random import random; print(random() < 0.1)"` = "True" ]]; then
        for bucket in page ip; do
            for lookahead in 0 4 8; do
                export BUCKET=$bucket
                export LOOKAHEAD=$lookahead
                PREFETCH_FILE="generated/$bucket-$lookahead.txt"
                if [[ $ML_COMMAND = "run" ]]; then
                    echo sbatch --export=ALL --mem 1G --time 3:00:00 scripts/single.sh ./ml_prefetch_sim.py run $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-$bucket-$lookahead" --no-base --prefetch $PREFETCH_FILE
                else
                    echo sbatch --export=ALL --mem 1G --time 3:00:00 scripts/single.sh ./ml_prefetch_sim.py train $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100 --generate $PREFETCH_FILE
                fi
            done
        done
    # fi
done; done
