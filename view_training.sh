CAFFE_HOME=/home/enoon/libs/bvlc_caffe/
DIR=$(dirname "$1")
NAME=$(basename "$1")
OUT_DIR=output_dir
cd $DIR
mkdir $OUT_DIR -p
python $CAFFE_HOME/tools/extra/parse_logv2.py ${NAME} ${OUT_DIR}
python ~/bin/caffe_log_viewer.py ${OUT_DIR}/${NAME}.train ${OUT_DIR}/${NAME}.test ${OUT_DIR}/out_${2}.png

