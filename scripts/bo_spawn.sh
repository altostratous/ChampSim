#!/bin/bash

TRACE_TYPE='LoadTraces'


mkdir -p generated
for b in `ls ML-DPC/$TRACE_TYPE`; do for t in `ls ML-DPC/$TRACE_TYPE/$b`; do
    # if [[ $b != "spec06" ]]; then continue; fi
    TRACE_PATH=ML-DPC/$TRACE_TYPE/$b/$t;
    PREFETCH_FILE="generated/$b-$t-bo.txt"
    PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
    export ML_MODEL_NAME=BestOffset
    export FUZZY_BO=True
    # --gres=gpu:p100:1
    JOB_ID=`sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py train $TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE`
    SUCCEEDED=$?
    if [[ $SUCCEEDED = 0 ]]; then
      JOB_ID=`echo $JOB_ID | awk '{print $4}'`
      sbatch --dependency=afterok:$JOB_ID --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run ML-DPC/ChampSimTraces/$b/$t --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-bo" --prefetch $PREFETCH_FILE  --no-base
    fi
done; done
