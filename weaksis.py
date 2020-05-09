#--weak lensing from SIS density profile----------
import numpy as np
from scipy import integrate 

# Basic Parameters---------------------------
h       = 1.0
w       = -1.0
vc = 2.9970e5                   #km/s
G = 4.3e-9                      #(Mpc/h)^1 (Msun/h)^-1 (km/s)^2 
H0 = 100.0                      #km/s/(Mpc/h)
pc = 3.085677e16                #m
kpc = 3.085677e19               #m
Mpc = 3.085677e22               #m
Msun = 1.98892e30               #kg
yr = 31536000.0/365.0*365.25    #second
Gyr = yr*1e9        
Gg    =6.67408e-11
ckm   =3.240779e-17
ckg   =5.027854e-31
Om0   = 0.28
Ol0   = 0.72
Ok0    = 1.0-Om0-Ol0
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*Om0       # M_sun Mpc^-3 *h*h
pi      = np.pi
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm)

class weaksis(object):
  def __init__(self,vrot=None,vdis=None,zl=None,zs=None):
     self.vrot=vrot
     self.vdis=vdis
     self.zl  = zl
     self.zs  = zs
#----cosmology from Nan Li-------------------------------
  def efunclcdm(self,x):
     res = 1.0/np.sqrt(Om0*(1.0+x)**3+Ok0*(1.0+x)**2+Ol0*(1.0+x)**(3*(1.0+w)))
     return res
  def Hz(self,x):
     res = H0/self.efunclcdm(x)
     return res
  def a(self,x):
     res = 1.0/(1.0+x)
     return res
  def Dh(self,):
     res = vc/H0
     return res
  def Da(self,x):
     res = self.Dh()*integrate.romberg(self.efunclcdm, 0, x)
     return res
#----end of cosmology functions---------------------------------
 
  def sisESD(self,xi):
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
     Sig = fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     ESD = Sig*self.vdis**2*ckg/(2.0*Gg*ckm*dl*xi*pi/180.0/60.0) 
     return ESD
  def sisKappa(self,xi1,xi2):
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
     Sig = fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     Sigc= fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     kap = self.vdis**2*ckg/(2.0*Gg*ckm*Sigc*dl*np.sqrt(xv*xv+yv*yv))

     return kap 
  def sisShear(self,xi1,xi2):
     xv,yv  = np.meshgrid(xi1,xi2)
     kappa  = self.sisKappa(xi1,xi2)
     the2   = (xv*xv+yv*yv)
     cph    = (xv*xv-yv*yv)/the2
     sph    = 2.0*xv*yv/the2
     
     gm1    = kappa*cph 
     gm2    = kappa*sph 
	
     return {'g_1':gm1,'g_2':gm2}



     return 0

