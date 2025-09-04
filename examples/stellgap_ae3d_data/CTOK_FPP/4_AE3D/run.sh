#!/bin/zsh

set -e

echo "Symlinking spectrum input..."
SPECTRUM_INPUT="../2_XMETRIC/inputs/fourier.dat"
ln -s $(realpath $SPECTRUM_INPUT) .

echo "Symlinking plasma profiles input..."
PROFILE_INPUT="../3_STELLGAP/inputs/plasma.dat"
ln -s $(realpath $PROFILE_INPUT) .

echo "Symlinking equilibrium data input..."
AE3D_METRIC_INPUT="../2_XMETRIC/outputs/ae_metric.dat"
ln -s $(realpath $AE3D_METRIC_INPUT) .

mkdir -p outputs

echo "Running AE3D..."
,xae3d | tee outputs/ae3d.log

echo "Moving outputs..."
mv egn_mode_asci.dat outputs/
mv egn_values.dat outputs/
mv a_matrix.dat outputs/
mv b_matrix.dat outputs/
mv jdqz.dat outputs/

echo "Removing unnecessary files..."
rm omega2.dat profiles.dat test.dat
rm fourier.dat plasma.dat
rm ae_metric.dat ae_diag.dat
