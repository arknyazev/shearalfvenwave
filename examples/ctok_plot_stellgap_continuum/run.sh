#!/bin/zsh
set -e

echo "Symlinking STELLGAP CTOK outputs needed for plotting..."
STELLGAP_OUTPUT="../stellgap_ae3d_data/CTOK_FPP/3_STELLGAP/outputs/"
ln -s $(realpath $STELLGAP_OUTPUT)/alfven_spec* .

echo "Running Python plotting script from dedicated Conda..."
conda run -n firm3d --no-capture-output python plot_stellgap_continuum.py

mkdir -p outputs

echo "Moving outputs..."
mv ctok_stellgap.html outputs/ctok_stellgap.html
mv ctok_stellgap.png outputs/ctok_stellgap.png

echo "Cleaning up..."
rm alfven_spec*

echo "Done."
