# Examples of STELLGAP & AE3D simulation data

This folder contains inputs & outputs for STELLGAP and AE3D simulations, 
along with the steps to recreate these outputs.

Computation steps are labeled as

0_vmec : preapares VMEC output for the equilibrium
1_boozxform : does Booz xform on the equilibrium using SIMSOPT
2_xmetric : computes equilibrium inputs for STELLGAP and AE3D using xmetric
3_stellgap : STELLGAP simulation
4_ae3d : AE3D simulation corresponging to 3_stellgap.

Each folder contains
 - inputs : folder with inputs needed for the step
 - outoputs : step computation results
 - run.sh : bash script to run the step
Some steps also have Python scripts if needed.
