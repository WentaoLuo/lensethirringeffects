# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#!/home/wtluo/anaconda/bin/python2.7

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import weaksis as wl
from scipy.ndimage import shift
import galsim as gs

def plot_shears_kappa(kappa,shear1,shear2,nn):

    g1 = shear1
    g2 = -shear2
    plt.figure(figsize=(10,10),dpi=80)
    plt.imshow(kappa,interpolation='NEAREST')
    #plt.colorbar()
    nnn  = nn
    ndiv = 4 
    scale_shear = 200 

    for i in xrange(ndiv/2,nnn,ndiv):
       for j in xrange(ndiv/2,nnn,ndiv):
           gt1 = g1[i,j]
           gt2 = g2[i,j]

           ampli = np.sqrt(gt1*gt1+gt2*gt2)
           alph = np.arctan2(gt2,gt1)/2.0

           st_x = i-ampli*np.cos(alph)*scale_shear
           md_x = i
           ed_x = i+ampli*np.cos(alph)*scale_shear

           st_y = j-ampli*np.sin(alph)*scale_shear
           md_y = j
           ed_y = j+ampli*np.sin(alph)*scale_shear
           #plt.plot([md_x,ed_x],[md_y,ed_y],'k-',linewidth=2)
           #plt.plot([md_x,st_x],[md_y,st_y],'k-',linewidth=2)

    plt.xlim(0,nn)
    plt.ylim(0,nn)

def selectregions(kap):
  nn  = len(kap[:,0])
  for i in range(nn):
     for j in range(nn):
	 if j>nn/2:
            if i>=nn/2 and j>i:
	       kap[i,j] =0.0
            if i<=nn/2 and j>nn-i:
	       kap[i,j] =0.0
	 if j<nn/2:
            if i>=nn/2 and j<nn-i:
	       kap[i,j] =0.0
            if i<=nn/2 and j<i:
	       kap[i,j] =0.0
  klow  = kap[nn/2:nn,:]
  khig  = kap[0:nn/2,:]
  return {'all':kap,'khig':khig,'klow':klow}

def stackedsims(vdis,xi1,xi2,nstack,sim_id):
  nx = len(xi1) 
  ny = len(xi2) 
  kap= np.zeros((nx,ny))
  zl  = 0.2
  zs  = 0.4
  vrot= 100.0
  for i in range(nstack):
     if sim_id==1:
       phi = 0.0
       lam = 0.1
     if sim_id==2:
       phi = 0.0
       lam = 0.1+np.random.normal(loc=0.0,scale=0.5)
     if sim_id==3:
       phi = 0.0+np.random.normal(loc=0.0,scale=0.05)
       lam = 0.1
     if sim_id==4:
       phi = 0.0+np.random.normal(loc=0.0,scale=0.05)
       lam = 0.1+np.random.normal(loc=0.0,scale=0.5)
     if sim_id==5:
       phi = 0.0+np.random.normal(loc=0.0,scale=0.05)
       lam = 0.05+np.random.normal(loc=0.0,scale=0.5)
     noise= True
     wlsis= wl.weaksis(vrot,vdis,lam,zl,zs)
     #gamma= wlsis.TotalShear(xi1,xi2,phi,noise)
     #kap  = kap+wlsis.kaisersquires(xi1,xi2,gamma,noise)
     kap  = kap+wlsis.TotalKappa(xi1,xi2,phi,noise)
  return kap/nstack

def lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise):
  #lam  = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
  #vd   = np.array([200.0,250.0,300.0,350.0,400.0])
  lam  = np.linspace(0.01,0.5,8)
  vd   = np.linspace(200.0,1300.0,8)
  color= np.array(['y','g','b','pink','m','cyan','k','r'])
  deltk= np.zeros(len(lam))
  coeff= zeros((8,2))
  plt.figure()
  for j in range(len(vd)):
    for i in range(len(lam)):
      wlsis= wl.weaksis(vrot,vd[j],lam[i],zl,zs)
      kap  = wlsis.GRMKappa(xi1,xi2,phi)
      #kap  = wlsis.TotalKappa(xi1,xi2,phi,noise)
      #kap  = selectregions(kap)
      #klow = kap['klow']
      #khig = kap['khig']
      #kall = kap['all']
      nn   = len(kap[:,0])
      khig = kap[0:nn/2,:]
      klow = kap[nn/2:nn,:]
      kall = kap
      #if j == 4 and i==6:
#	 cbar=plt.imshow(kall,interpolation='nearest')
#	 cbar.set_label(r'$\kappa$')
#	 colorbar()
#	 plt.show()
      deltk[i]= np.mean(khig)-np.mean(klow)
    coeff[j,:]= np.polyfit(lam,deltk,1)
    #print '------First-----------------------'
    #print j, coeff[j,0],vd[j],coeff[j,1]
    plt.plot(lam,deltk,'-',ms=5,linewidth=2.5,\
             color=color[j],label=r'$\sigma_v=$'+str(np.round(vd[j],1)))
    #plt.plot(lam,lam*2.832819e-5*vd[j],'k-')
  #secon = np.polyfit(vd,coeff[:,0],1)
  #print secon
  #plt.plot(vd,coeff[:,0],'ro',ms=10)
  #plt.plot(vd,secon[0]*vd+secon[1],'k-',ms=10)
  plt.xlabel(r'$\mathrm{\lambda}$',fontsize=15)
  plt.ylabel(r'$\mathrm{\delta\kappa}$',fontsize=15)
  plt.xlim(0.01,0.5)
  plt.legend(loc='upper left')
  plt.show()
  #coeffs = np.polyfit(lam,deltk)
  return 0
#----------------------------------------------------
def main():
  xi  = np.linspace(0.02,10,50)
  nn  = 8
  ang = 64
  galsim = False
  xi1 = np.linspace(-ang,ang,nn)
  xi2 = np.linspace(-ang,ang,nn)
  #---I:Test k coordinates----------------------------------
  #xv  = np.arange(0,nn,dtype=int)
  #yv  = np.arange(0,nn,dtype=int)
  #l1,l2 =np.meshgrid(xv,yv)
  #print(np.roll(l1-7.5,-7,axis=1)[0,:])
  #print(np.roll(l2-7.5,-7,axis=0)[0,:])
  #--- end of Test k coordinates----------------------------------
  vdis= 1300.0
  vrot= 100.0
  zl  = 0.2
  zs  = 0.4
  noise = False
  #---II:lambda vs delta_kappa relation without noise---------
  #lamvsdkapp=lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise)
  #---end lambda vs delta_kappa relation without noise---------
  #---III:test Kaiser Squires gamma to kappa conversion-----
  #wlsis= wl.weaksis(vrot,vdis,lam,zl,zs)
  #kap  = wlsis.TotalKappa(xi1,xi2,phi,noise)
  #gamma= wlsis.TotalShear(xi1,xi2,phi,noise)
  #ke,kb = wlsis.kaisersquires(xi1,xi2,gamma,noise,galsim)

  #plt.imshow(ke,interpolation='nearest')
  #plt.colorbar()
  #plt.show()
  #plt.imshow(kap,interpolation='nearest')
  #plt.colorbar()
  #plt.show()
  #---end of Kaiser Squires test---------------------
  #---IV:test filters(Gauss,Wiener,& maximum entropy)-----
  #---end of filters(Gauss,Wiener,& maximum entropy)-----
  #---V:stacked simulation------------------------
  sim_id=5
  nstack=100
  kap = stackedsims(vdis,xi1,xi2,nstack,sim_id)
  khig  = kap[nn/2:nn,:]
  klow  = kap[0:nn/2,:]
  print(np.mean(khig),np.std(khig))
  print(np.mean(klow),np.std(klow))
  print(np.mean(klow)-np.mean(khig),np.std(khig)/np.sqrt(nstack))
  plt.imshow(kap,interpolation='nearest')
  plt.colorbar()
  plt.show()
  #-----end ofstacked simulation------------------------

if __name__=="__main__":
    main()
   
