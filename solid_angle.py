import numpy
import mycosmology as cosmos

z1 = 0.01
z2 = 0.2
omega_north = 2.14
Da1 = cosmos.Da(z1)*(1.0+z1)
Da2 = cosmos.Da(z2)*(1.0+z2)
volume = omega_north*(1.0/3.0)*(Da2**3-Da1**3)
z1 = 0.01
z2 = 1.0
Da1 = cosmos.Da(z1)*(1.0+z1)
Da2 = cosmos.Da(z2)*(1.0+z2)
volume2 = omega_north*(1.0/3.0)*(Da2**3-Da1**3)
print(volume2/volume)
#print 3660.0*400**3/volume


