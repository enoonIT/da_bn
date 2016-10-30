CAFFE_HOME=/home/enoon/libs/bvlc_caffe/
OUT_DIR=output_dir
for line in $(find ${1} -maxdepth 2 -mindepth 2 -type d -not -path '*/\.*'); do 
     cd $line
     echo Starting $line
     ${CAFFE_HOME}/build/tools/caffe train -solver solver.prototxt -weights ${CAFFE_HOME}/models/bvlc_alexnet/bvlc_alexnet.caffemodel 2>&1 | tee caffe_output.txt
     mkdir $OUT_DIR -p
     python $CAFFE_HOME/tools/extra/parse_log.py caffe_output.txt ${OUT_DIR}
     cd -
done
#python ~/bin/caffe_log_viewer.py ${OUT_DIR}/${NAME}.train ${OUT_DIR}/${NAME}.test ${OUT_DIR}/out_${2}.png
