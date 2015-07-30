from .core import *

# you really shouldn't define stuff in __init__.py
import gary.potential as gp
from gary.units import galactic
potential = gp.LogarithmicPotential(v_c=np.sqrt(2), r_h=1., q1=1., q2=1., q3=1., units=galactic)
