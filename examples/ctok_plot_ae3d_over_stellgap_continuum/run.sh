#!/bin/zsh
set -e

echo "Symlinking STELLGAP CTOK outputs needed for plotting..."
STELLGAP_OUTPUT="../stellgap_ae3d_data/CTOK_FPP/3_STELLGAP/outputs/"
ln -s $(realpath $STELLGAP_OUTPUT)/alfven_spec* .

echo "Symlinking AE3D CTOK outputs needed for plotting"
AE3D_OUTPUT="../stellgap_ae3d_data/CTOK_FPP/4_AE3D/outputs/egn_mode_asci.dat"
ln -s $(realpath $AE3D_OUTPUT) .

echo "Running Python script from dedicated Conda..."
conda run -n firm3d --no-capture-output python plot_ae3d_over_stellgap.py

mkdir -p outputs

echo "Moving outputs..."
OUTPUTS=(
    ctok_stellgap_ae3d_continuum_overlay.html
    ctok_stellgap_ae3d_continuum_overlay.png
)

for file in $OUTPUTS; do
    mv "$file" outputs/
done

echo "Cleaning up..."
rm egn_mode_asci.dat
rm alfven_spec*

echo "Done."
