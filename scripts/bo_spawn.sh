#!/bin/bash


for b in `ls ML-DPC/ChampSimTraces`; do for t in `ls ML-DPC/ChampSimTraces/$b`; do
    TRACE_PATH=ML-DPC/ChampSimTraces/$b/$t;
    LOAD_TRACE_PATH=ML-DPC/LoadTraces/$b/`echo $t | sed 's/trace.xz/txt.xz/g' | sed 's/trace.gz/txt.xz/g'`;
    export ML_MODEL_NAME=MementoModel
    export FUZZY_BO=True
    export BO_SCORE_SCALE=$scale
    export MEMENTO_DELAY=0
    GENERATED_DIR="generated-memento-$MEMENTO_DELAY"
    mkdir -p $GENERATED_DIR
    PREFETCH_FILE="$GENERATED_DIR/$b-$t-memento.txt"
    PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
    PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.xz/txt.xz/g'`
    # --gres=gpu:p100:1
    TRAIN_COMMAND="sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py train $LOAD_TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE"
    echo $TRAIN_COMMAND
    JOB_ID=`$TRAIN_COMMAND`
    SUCCEEDED=$?
    if [[ $SUCCEEDED = 0 ]]; then
      JOB_ID=`echo $JOB_ID | awk '{print $4}'`
      sbatch --dependency=afterok:$JOB_ID --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run ML-DPC/ChampSimTraces/$b/$t --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-memento$MEMENTO_DELAY" --prefetch $PREFETCH_FILE
    fi
    if [[ -f $TRACE_PATH ]] && [[ -f $LOAD_TRACE_PATH ]]; then echo OK; fi
    exit
done; done
