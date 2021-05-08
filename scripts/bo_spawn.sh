#!/bin/bash


for b in `ls ML-DPC/ChampSimTraces`; do for t in `ls ML-DPC/ChampSimTraces/$b`; do
    if [[ $b != "spec17" ]]; then continue; fi
    TRACE_PATH=ML-DPC/ChampSimTraces/$b/$t;
    LOAD_TRACE_PATH=ML-DPC/LoadTraces/$b/`echo $t | sed 's/trace.xz/txt.xz/g' | sed 's/trace.gz/txt.xz/g'`;
    export ML_MODEL_NAME=BestOffset
    export FUZZY_BO=True
    for scale in 0.5 2; do
      export BO_SCORE_SCALE=$scale
      GENERATED_DIR="generated-ss$scale"
      mkdir -p $GENERATED_DIR
      PREFETCH_FILE="$GENERATED_DIR/$b-$t-bo.txt"
      PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.gz/txt.xz/g'`
      PREFETCH_FILE=`echo $PREFETCH_FILE | sed 's/trace.xz/txt.xz/g'`
      # --gres=gpu:p100:1
      TRAIN_COMMAND="sbatch --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py train $LOAD_TRACE_PATH --num-prefetch-warmup-instructions 100 --generate $PREFETCH_FILE"
      echo $TRAIN_COMMAND
      JOB_ID=`echo $TRAIN_COMMAND`
      SUCCEEDED=$?
      if [[ $SUCCEEDED = 0 ]]; then
        JOB_ID=`echo $JOB_ID | awk '{print $4}'`
        echo sbatch --dependency=afterok:$JOB_ID --export=ALL --mem 10G --time 6:00:00 scripts/single.sh ./ml_prefetch_sim.py run ML-DPC/ChampSimTraces/$b/$t --num-prefetch-warmup-instructions 100 --num-instructions 100 --results-dir="results-bo-ss$scale" --prefetch $PREFETCH_FILE --no-base
      fi
      if [[ -f $TRACE_FILE_PATH ]] && [[ -f $LOAD_TRACE_FILE_PATH ]]; then echo OK; fi
    done
done; done
