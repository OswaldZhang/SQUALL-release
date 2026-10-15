#!/bin/bash

cd `python -c "import stnet; print(stnet.config.SPATIAL_RAW_ROOT)"`
for i in */*/*.jpg;
do
    echo ${i}
    vips tiffsave ${i} ${i%.jpg}.tif --tile --tile-width=256 --tile-height=256 --bigtiff
    #magick "$i" -define tiff:tile-geometry=256x256 -define tiff:use-bigtiff=true "${i%.jpg}.tif"
    #magick convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done

