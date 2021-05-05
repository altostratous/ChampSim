#!/bin/bash

TRACE_TYPE='ChampSimTraces'
#TRACE_TYPE='LoadTraces'
ML_COMMAND=run
#ML_COMMAND=train


mkdir -p generated
for b in `ls ML-DPC/$TRACE_TYPE`; do for t in `ls ML-DPC/$TRACE_TYPE/$b`; do
    if [[ $b != "gap" ]]; then continue; fi
    TRACE_PATH=ML-DPC/$TRACE_TYPE/$b/$t;
    # if [[ `python -c "from random import random; print(random() < 0.1)"` = "True" ]]; then
        for bucket in page ip; do
            for lookahead in 0 2 4 8; do
                export BUCKET=$bucket
                export LOOKAHEAD=$lookahead
                PREFETCH_FILE="generated/$t-$bucket-$lookahead.txt"
                if [[ $ML_COMMAND = "run" ]]; then
                    # echo $PREFETCH_FILE
                    PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
                    # echo $PREFETCH_FILE
                    if [[ -f $PREFETCH_FILE ]]; then
                        sbatch --export=ALL --mem 1G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-$bucket-$lookahead" --prefetch $PREFETCH_FILE
                    fi
                else
                    # --gres=gpu:p100:1
                    sbatch --export=ALL --mem 8G --gres=gpu:p100:1 --time 3:00:00 scripts/single.sh ./ml_prefetch_sim.py train $TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE
                fi
            done
        done
    # fi
done; done
