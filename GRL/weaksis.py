# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:54:35 2020

@author: tcf12
"""

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
lamda =0.5
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm)

shapen     = 0.1
ngdens     = 50.0
class weaksis(object):
  def __init__(self,vrot=None,vdis=None,zl=None,zs=None,noise=None):
     self.vrot=vrot
     self.vdis=vdis
     self.zl  = zl
     self.zs  = zs
     self.noise  = noise
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
     nbins      = len(xi)
     error      = np.random.normal(loc=0.0,scale=shapen,size=nbins-1)/np.sqrt((2.0*pi*xi[1:nbins]-2.0*pi*xi[0:nbins-1])*ngdens)
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
     theta      = dl*(xi[1:nbins]+xi[0:nbins-1])*pi/360.0/60.0
     print(theta,error)
     Sigc = fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     #print self.noise
     if self.noise==False:
       tmp = Sigc*self.vdis**2*ckg/(2.0*Gg*ckm*dl*theta*pi/180.0/60.0) 
     if self.noise==True:
       tmp = Sigc*(error+self.vdis**2*ckg)/(2.0*Gg*ckm*dl*theta*pi/180.0/60.0) 
     res={'Rp':theta,'ESD':tmp,'ERROR':error}
     return res 
  def Xi0(self,):
      dl  = self.Da(self.zl)
      ds  = self.Da(self.zs)
      xi0=4*pi*(self.vdis*self.vdis)*dl*(ds-dl)/(vc*vc)/ds
      return xi0

  def sisKappa(self,xi1,xi2):
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
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
     return {'f_1':gm1,'f_2':gm2}
 
  def GRMKappa(self,xi1,xi2,phi):
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     Rsis= self.Xi0()
     w = 9*lamda*self.vdis/Rsis
     w1=w*np.cos(phi)
     w2=w*np.sin(phi)
     #ds  = self.Da(self.zs)
     #Sigc= fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     #kap = self.vdis**2*ckg/(2.0*Gg*ckm*Sigc*dl*np.sqrt(xv*xv+yv*yv))
     kap=self.sisKappa(xi1,xi2)
     kap1 = kap*dl*(w2*xv-w1*yv)/vc
     return kap1 
  
  def GRMShear(self,xi1,xi2,phi):
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     Rsis= self.Xi0()
     w = 9*lamda*self.vdis/Rsis
     w1=w*np.cos(phi)
     w2=w*np.sin(phi)
     kappa  = self.sisKappa(xi1,xi2)
     the  = np.sqrt(xv*xv+yv*yv)
     gm1 = kappa*dl*(w2*xv**3+w1*yv**3+3*xv*yv*(w1*xv+w2*yv))/(3*vc*the*the)
     gm2 = 2*kappa*dl*(w2*yv**3-w1*xv**3)/(3*vc*the*the)
     return {'g_1':gm1,'g_2':gm2}
 
  def TotalKappa(self,xi1,xi2,phi):
     kappa  = self.sisKappa(xi1,xi2)
     GRkappa= self.GRMKappa(xi1,xi2,phi)
     Totalkappa=kappa+GRkappa
     return  Totalkappa
 
  def TotalShear(self,xi1,xi2,phi):
    gmsis=self.sisShear(xi1,xi2)
    gmsis1=gmsis['f_1']
    gmsis2=gmsis['f_2']
    
    gmGRM=self.GRMShear(xi1,xi2,phi)
    gmGRM1=gmGRM['g_1']
    gmGRM2=gmGRM['g_2']
    
    gm1=gmsis1+gmGRM1
    gm2=gmsis2+gmGRM2
    return {'h_1':gm1,'h_2':gm2}
