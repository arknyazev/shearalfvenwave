#!/bin/zsh
set -e # stop on error

echo "Symlinking inputs..."
ln -s $(realpath inputs/input.ctok) .

echo "Using the VMEC thorugh SIMSOPT in a dedicated conda enviroment..."

conda run -n for_vmec --no-capture-output mpirun -np 1 python run_vmec_on_ctok_fpp.py

echo "Moving outputs..."
mv wout_ctok_000_000001.nc outputs/wout_ctok.nc

echo "Removing unnecessary files..."
rm input.ctok
rm input.ctok_000_000001
rm threed1.ctok

echo "Done."
