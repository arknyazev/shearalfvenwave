import os
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi = MpiPartition(comm_world=comm)
except ImportError:
    comm = None
    mpi = None

if comm is not None:
    if comm.rank == 0:
        verbose = True
    else:
        verbose = False
else:
    verbose = True

filename = 'input.ctok' 

vmec = Vmec(filename)

files_before = set(os.listdir())
vmec.run()
files_after = set(os.listdir())
new_files = files_after - files_before
for file in new_files:
    try:
        os.remove(file)
        print(f"Deleted temporary: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")

# Rescale equilibrium
Vt = 444.385920765916
Bt = 5.86461234641553
Vc = vmec.wout.volume_p
Bc = vmec.wout.volavgB
phic = vmec.wout.phi[-1]
boundary = vmec._boundary

dofs = boundary.get_dofs()
dofs *= (Vt/Vc)**(1/3)

boundary.set_dofs(dofs)
phic *= (Vt/Vc)**(2/3) * (Bt/Bc)
vmec.indata.phiedge = phic
vmec.run()