# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:54:35 2020

@author: tcf12
"""
#--weak lensing from SIS density profile----------
import numpy as np
from scipy import integrate 
import galsim as gs
from scipy.ndimage import shift
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
#lamda =0.1
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm)

shapen     = 0.3
ngdens     = 50.0
class weaksis(object):
  def __init__(self,vrot=None,vdis=None,lam=None,zl=None,zs=None):
     self.vrot=vrot
     self.vdis=vdis
     self.zl  = zl
     self.zs  = zs
     self.lam = lam
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
  def kaisersquires(self,xi1,xi2,gamma,noise,galsim):
     nn    = len(xi1)
     the   = np.max(xi1)*pi*2.0/180.0/60.0
     xv    = np.arange(0,nn,dtype=int)
     yv    = np.arange(0,nn,dtype=int)
     l1,l2 = np.meshgrid(xv,yv)
     #l1    = np.roll(l1-float(nn/2-0.5),-int(nn/2-0.5),axis=0)
     #l2    = np.roll(l2-float(nn/2-0.5),-int(nn/2-0.5),axis=0)
     l1    = shift(l1-float(nn/2.0-0.5),[(0.5-nn/2.0),(0.5-nn/2.0)],mode='wrap',order=5)
     l2    = shift(l2-float(nn/2.0-0.5),[(0.5-nn/2.0),(0.5-nn/2.0)],mode='wrap',order=5)
     l1    = 2.0*pi*l1/the 
     l2    = 2.0*pi*l2/the 
     gm1   = gamma['h_1']
     gm2   = gamma['h_2']
     fgm1  = np.fft.fft2(gm1)
     fgm2  = np.fft.fft2(gm2)
     fkape = ((l1**2-l2**2)*fgm1+(2.0*l1*l2)*fgm2)/(l1*l1+l2*l2)
     fkapb = ((l1**2-l2**2)*fgm1+(2.0*l1*l2)*fgm2)*0/(l1*l1+l2*l2)
     fkape[0,0] = 0.0
     kappe = np.fft.ifft2(fkape)
     kappb = np.fft.ifft2(fkapb)
     kappa = [kappe.real,kappb.real]
     ns1   = np.random.normal(loc=0.0,scale=shapen,size=(nn,nn))
     ns2   = np.random.normal(loc=0.0,scale=shapen,size=(nn,nn))
     if galsim==True:
       kappa = gs.lensing_ps.kappaKaiserSquires(gm1,gm2)

     if noise==True:
       gm1 = gm1+ns1
       gm2 = gm2+ns2
     return kappa
  def sisESD(self,xi,noise):
     nbins = len(xi)
     dl    = self.Da(self.zl)
     ds    = self.Da(self.zs)
     theta = dl*((0.5*xi[1:nbins]+0.5*xi[0:nbins-1])*pi/180.0/60.0)
     Sigc  = fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     gm    = self.vdis**2*ckg/(2.0*Gg*ckm*Sigc*theta)
     ngals = (pi*(xi[1:nbins]**2-xi[0:nbins-1]**2)*ngdens)
     error = (gm*0.0+shapen)/np.sqrt(ngals)
     if noise==False:
       tmp = Sigc*gm
     if noise==True:
       tmp = np.zeros(nbins-1)
       for j in range(nbins-1):
         gmt    = np.random.normal(loc=gm[j],scale=shapen,size=int(np.round(ngals[j])))
         tmp[j] = Sigc*np.mean(gmt)
     res={'Rp':theta,'ESD':tmp,'ERROR':error*Sigc}
     return res 
  def Xi0(self,):
      dl  = self.Da(self.zl)
      ds  = self.Da(self.zs)
      xi0=4*pi*(self.vdis*self.vdis)*dl*(ds-dl)/(vc*vc)/ds
      #print(2.0*self.vdis**2*ckg)*xi0/(Gg*ckm)
      return xi0

  def sisKappa(self,xi1,xi2,noise):
     xv,yv  = np.meshgrid(xi1,xi2)
     nx  = len(xi1)
     ny  = len(xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
     Sigc= fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     ngals= int(np.round(((4.0*np.max(xi1)*np.max(xi2))/float(nx)/float(ny))*ngdens))
     kap = self.vdis**2*ckg/(2.0*Gg*ckm*Sigc*dl*np.sqrt(xv*xv+yv*yv))
     if noise==False:
       kap = kap
     if noise==True:
       tmp = np.zeros((nx,ny))
       for i in range(ny):
         for j in range(ny):
           gmt    = np.random.normal(loc=kap[i,j],scale=shapen,size=ngals)
           kap[i,j] = np.mean(gmt)
     
     return kap 
 
  def sisShear(self,xi1,xi2,noise):
     xv,yv  = np.meshgrid(xi1,xi2)
     kappa  = self.sisKappa(xi1,xi2,noise)
     the2   = (xv*xv+yv*yv)
     cph    = (xv*xv-yv*yv)/the2
     sph    = 2.0*xv*yv/the2
     
     gm1    = kappa*cph 
     gm2    = kappa*sph 
     return {'f_1':gm1,'f_2':gm2}
 
  def GRMKappa(self,xi1,xi2,phi):
     lamda  = self.lam
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     Rsis= self.Xi0()
     w = 9*lamda*self.vdis/Rsis   
     w1=w*np.cos(phi)
     w2=w*np.sin(phi)
     noise=False
     kap=self.sisKappa(xi1,xi2,noise)
     kap1 = kap*dl*(w2*xv-w1*yv)/vc
     return kap1 
  
  def GRMShear(self,xi1,xi2,phi):
     lamda  = self.lam
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     Rsis= self.Xi0()
     w = 9*lamda*self.vdis/Rsis
     w1=w*np.cos(phi)
     w2=w*np.sin(phi)
     noise=False
     kappa  = self.sisKappa(xi1,xi2,noise)
     the  = np.sqrt(xv*xv+yv*yv)
     gm1 = kappa*dl*(w2*xv**3+w1*yv**3+3*xv*yv*(w1*xv+w2*yv))/(3*vc*the*the)
     gm2 = 2*kappa*dl*(w2*yv**3-w1*xv**3)/(3*vc*the*the)
     return {'g_1':gm1,'g_2':gm2}
 
  def TotalKappa(self,xi1,xi2,phi,noise):
     nx  = len(xi1)
     ny  = len(xi2)
     kappa  = self.sisKappa(xi1,xi2,False)
     GRkappa= self.GRMKappa(xi1,xi2,phi)
     ngals= int(np.round(((4.0*np.max(xi1)*np.max(xi2))/float(nx)/float(ny))*ngdens))
     Totalkappa=kappa+GRkappa
     #print(3.0*2.0*self.vdis**2*ckg/Gg/ckm)
     if noise==False:
       Totalkappa = Totalkappa
     if noise==True:
       tmp = np.zeros((nx,ny))
       for i in range(ny):
         for j in range(ny):
           gmt    = np.random.normal(loc=Totalkappa[i,j],scale=shapen,size=ngals)
           Totalkappa[i,j] = np.mean(gmt)
     return  Totalkappa 
  def TotalESD(self,xi1,xi2,phi,noise):
     xv,yv  = np.meshgrid(xi1,xi2)
     xv  = xv*pi/180.0/60.0
     yv  = yv*pi/180.0/60.0
     dl  = self.Da(self.zl)
     ds  = self.Da(self.zs)
     nrbin = 700
     Sigc  = fac*ds/(dl*(ds-dl))/(1.0+self.zl)/(1.0+self.zl)
     kap2d = Sigc*self.TotalKappa(xi1,xi2,phi,False)
     Rlow  = 0.0
     Rhig  = np.max(dl*xi1*pi/180.0/60.0)
     step  = Rhig/float(nrbin)
     Rp2d  = dl*np.sqrt(xv**2+yv**2)
     Rp    = np.zeros(nrbin)
     esd   = np.zeros(nrbin)
     error = np.zeros(nrbin)
     for i in range(nrbin):
	 ixa   = Rp2d>=Rlow+step*float(i)
	 ixb   = Rp2d<=Rlow+step*float(i+1)
	 Rp[i] = 0.5*(2.0*Rlow+step*float(i)+step*float(i+1))
	 esd[i]= np.mean(kap2d[ixb])-np.mean(kap2d[ixa&ixb])
     res = {'Rp':Rp,'ESD':esd}
     return res
  def TotalShear(self,xi1,xi2,phi,noise):
     
    nx  = len(xi1)
    ny  = len(xi2)
    gmsis=self.sisShear(xi1,xi2,noise)
    gmsis1=gmsis['f_1']
    gmsis2=gmsis['f_2']
    
    ngals= int(np.round(((4.0*np.max(xi1)*np.max(xi2))/float(nx)/float(ny))*ngdens))
    gmGRM=self.GRMShear(xi1,xi2,phi)
    gmGRM1=gmGRM['g_1']
    gmGRM2=gmGRM['g_2']
    
    gm1=gmsis1+gmGRM1
    gm2=gmsis2+gmGRM2
    if noise==False:
       gm1 = gm1
       gm2 = gm2
    if noise==True:
       tmp1 = np.zeros((nx,ny))
       tmp2 = np.zeros((nx,ny))
       for i in range(ny):
         for j in range(ny):
           gmt1    = np.random.normal(loc=gm1[i,j],scale=shapen,size=ngals)
           gmt2    = np.random.normal(loc=gm2[i,j],scale=shapen,size=ngals)
           gm1[i,j] = np.mean(gmt1)
           gm2[i,j] = np.mean(gmt2)
    return {'h_1':gm1,'h_2':gm2}
