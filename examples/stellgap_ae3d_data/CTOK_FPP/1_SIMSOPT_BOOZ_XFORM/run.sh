#!/bin/zsh
set -e # Stop on error

echo "Symlinking VMEC output as input..."
ln -s $(realpath  "../0_VMEC/outputs/wout_ctok.nc") .

echo "Using SIMSOPT BOOZXFORM in a dedicated conda enviroment..."

conda run -n for_vmec --no-capture-output python do_booz_xform.py

echo "Moving outputs..."
mv boozmn.nc outputs/boozmn.nc

echo "Removing unnecessary files..."
rm wout_ctok.nc

echo "Done."
