#!/usr/bin/env bash
# RetExpert's BaseRetinaDataset expects, under --data_path:
#   MuReD : train.csv / val.csv / test.csv  +  images/
#   RFMiD : train.csv / val.csv / test.csv  +  images/{train,val,test}/
# The ISBI copies use different file names, so build that layout out of
# symlinks (no data is copied).
set -euo pipefail
SRC=/globalsc/ucl/ingi/cousint/ISBI_datasets
DST=/globalsc/ucl/ingi/cousint/SOTA_runs/data

mkdir -p "$DST/MuReD" "$DST/RFMiD/images"

ln -sfn "$SRC/MURED/train_labels_stratified.csv" "$DST/MuReD/train.csv"
ln -sfn "$SRC/MURED/val_labels_stratified.csv"   "$DST/MuReD/val.csv"
ln -sfn "$SRC/MURED/test_labels_stratified.csv"  "$DST/MuReD/test.csv"
ln -sfn "$SRC/MURED/images/images"               "$DST/MuReD/images"

ln -sfn "$SRC/RFMiD/train_labels.csv"            "$DST/RFMiD/train.csv"
ln -sfn "$SRC/RFMiD/validation_labels_29.csv"    "$DST/RFMiD/val.csv"
ln -sfn "$SRC/RFMiD/testing_labels_29.csv"       "$DST/RFMiD/test.csv"
ln -sfn "$SRC/RFMiD/Training"                    "$DST/RFMiD/images/train"
ln -sfn "$SRC/RFMiD/Validation"                  "$DST/RFMiD/images/val"
ln -sfn "$SRC/RFMiD/Test"                        "$DST/RFMiD/images/test"

echo "prepared:"
find "$DST" -maxdepth 3 \( -type l -o -type d \) | sort
