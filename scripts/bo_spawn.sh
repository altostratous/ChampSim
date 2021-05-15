#!/bin/bash


for b in `ls ML-DPC/ChampSimTraces`; do for t in `ls ML-DPC/ChampSimTraces/$b`; do
    TRACE_PATH=ML-DPC/ChampSimTraces/$b/$t;
    LOAD_TRACE_PATH=ML-DPC/LoadTraces/$b/`echo $t | sed 's/trace.xz/txt.xz/g' | sed 's/trace.gz/txt.xz/g'`;
    export FUZZY_BO=True
    export BO_SCORE_SCALE=1
    export MEMENTO_DELAY=1
    export ML_MODEL_NAME=TerribleMLModel
    export CNN_LR=0.002
    export EPOCHS=100

    for bucket in ip page; do for lookahead in 5 10 15 20; do for modelclass in CNN MLP; do
        export LOOKAHEAD=$lookahead
        export BUCKET=$bucket
        export CNN_MODEL_CLASS=$modelclass
        VARIATION="$BUCKET-$LOOKAHEAD-$CNN_MODEL_CLASS"
        GENERATED_DIR="generated-cnnsearch-$VARIATION"
        mkdir -p $GENERATED_DIR
        PREFETCH_FILE="$GENERATED_DIR/$b-$t.txt"
        PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
        PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.xz/txt.xz/g'`
        # --gres=gpu:p100:1
        TRAIN_COMMAND="sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py train $LOAD_TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE"
        echo $TRAIN_COMMAND
        JOB_ID=`$TRAIN_COMMAND`
        SUCCEEDED=$?
        if [[ $SUCCEEDED = 0 ]]; then
          JOB_ID=`echo $JOB_ID | awk '{print $4}'`
          sbatch --dependency=afterok:$JOB_ID --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run ML-DPC/ChampSimTraces/$b/$t --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-cnnsearch-$VARIATION" --prefetch $PREFETCH_FILE --no-base
        fi
        if [[ -f $TRACE_PATH ]] && [[ -f $LOAD_TRACE_PATH ]]; then echo OK; fi
    done; done; done
done; done
