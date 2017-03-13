CAFFE_HOME=/home/enoon/libs/autodial_caffe/
#MODEL_PATH=${CAFFE_HOME}/models/bvlc_alexnet/bvlc_alexnet.caffemodel
#MODEL_PATH=/home/enoon/code/inception_bn_iter_128000.caffemodel
MODEL_PATH=${4}
OUT_DIR=output_dir
N_SPLITS=5
for line in $(find ${1} -maxdepth ${3} -mindepth ${3} -type d -not -path '*/\.*'); do 
    cd $line
    for i in $(seq 1 ${N_SPLITS}); do
	echo Starting $line - split $i
	log_file=wtf_caffe_output_${i}.txt
	${CAFFE_HOME}/build/tools/caffe train -solver solver.prototxt -gpu ${2} -weights $MODEL_PATH 2>&1 | tee ${log_file}
	mkdir ${OUT_DIR}_${i} -p
	python $CAFFE_HOME/tools/extra/parse_log.py ${log_file} ${OUT_DIR}_${i}
	done
     cd -
done
#python ~/bin/caffe_log_viewer.py ${OUT_DIR}/${NAME}.train ${OUT_DIR}/${NAME}.test ${OUT_DIR}/out_${2}.png
