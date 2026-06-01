#!/bin/bash

# 假设你想自己手动指定 root 路径
#cd /lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript_new
cd /lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript_OC
for i in */*/*.jpg;
do
    echo "${i}"
    convert "${i}" -define tiff:tile-geometry=256x256 -define tiff:big=true "${i%.jpg}.tif"
done

