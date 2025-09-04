import os
import booz_xform as bx

eq = bx.Booz_xform()
eq.read_wout('wout_ctok.nc', flux=True)
eq.run()
eq.write_boozmn('boozmn.nc')
