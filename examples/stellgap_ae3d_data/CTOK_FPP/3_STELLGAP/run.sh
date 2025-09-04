#!/bin/zsh

set -e

echo "Symlinking spectrum input..."
SPECTRUM_INPUT="../2_XMETRIC/inputs/fourier.dat"
ln -s $(realpath $SPECTRUM_INPUT) .

echo "Symlinking plasma profiles input..."
PROFILE_INPUT="inputs/plasma.dat"
ln -s $(realpath $PROFILE_INPUT) .

echo "Symlinking equilibrium data input..."
STELLGAP_METRIC_INPUT="../2_XMETRIC/outputs/tae_data_boozer"
ln -s $(realpath $STELLGAP_METRIC_INPUT) .

mkdir -p outputs


echo "Running STELLGAP..."
conda run --no-capture-output -n stellgap mpirun -np 4 ,xstgap 99 784 | tee outputs/stellgap.log

echo "Moving outputs..."
mv alfven_spec* outputs/ 2>/dev/null

echo "Deleting extra files..."
rm tae_data_boozer
rm ion_profile modes data_post coef_arrays
rm fourier.dat plasma.dat
