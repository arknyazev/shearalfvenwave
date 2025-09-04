#!/bin/zsh
set -e

echo "Symlinking boozmn.nc field input"
BOOZ_INPUT="../1_SIMSOPT_BOOZ_XFORM/outputs/boozmn.nc"
ln -s $(realpath $BOOZ_INPUT) .

echo "Symlinking grids input (first line in fourier.dat)"
GRID_INPUT="inputs/fourier.dat"
ln -s $(realpath $GRID_INPUT) .

mkdir -p outputs

echo "Running XMETRIC..."
,xmetric boozmn.nc

echo "Moving output..."
mv ae_metric.dat outputs/
mv tae_data_boozer outputs/


echo "Deleting unused files"
rm boozmn.nc fourier.dat
rm cart_coords full_torus_coords torus_slice_coords
rm surf_area_elements
rm metrics

echo "All done."
