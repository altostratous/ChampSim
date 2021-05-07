#!/bin/bash

TRACE_TYPE='ChampSimTraces'
#TRACE_TYPE='LoadTraces'
ML_COMMAND=run
#ML_COMMAND=train


mkdir -p generated
for b in `ls ML-DPC/$TRACE_TYPE`; do for t in `ls ML-DPC/$TRACE_TYPE/$b`; do
    # if [[ $b != "spec06" ]]; then continue; fi
    TRACE_PATH=ML-DPC/$TRACE_TYPE/$b/$t;
    PREFETCH_FILE="generated/$b-$t-bo.txt"
    export ML_MODEL_NAME=BestOffset
    if [[ $ML_COMMAND = "run" ]]; then
        PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
        if [[ -f $PREFETCH_FILE ]]; then
            sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run $TRACE_PATH --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-bo" --prefetch $PREFETCH_FILE
        fi
    else
        # --gres=gpu:p100:1
        sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py train $TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE --no-base
    fi
done; done
