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
def stackedsims(vrot,vdis,xi1,xi2,nstack):
  nx = len(xi1) 
  ny = len(xi2) 
  kap= np.zeros((nx,ny))
  for i in range(nstack):
     phi = 0
     zl  = 0.2
     zs  = 0.4
     noise= True
     wlsis= wl.weaksis(vrot,vdis,zl,zs)
     gamma= wlsis.TotalShear(xi1,xi2,phi,noise)
     kap  = kap+wlsis.kaisersquires(xi1,xi2,gamma,noise)
  return kap/nstack

def lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise):
  lam  = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
  vd   = np.array([200.0,250.0,300.0,350.0,400.0])
  color= np.array(['y','g','b','pink','m'])
  deltk= np.zeros(len(lam))
  plt.figure()
  for j in range(len(vd)):
    for i in range(len(lam)):
      wlsis= wl.weaksis(vrot,vd[j],lam[i],zl,zs)
      kap  = wlsis.TotalKappa(xi1,xi2,phi,noise)
      kap  = selectregions(kap)
      klow = kap['klow']
      khig = kap['khig']
      kall = kap['all']
      #if j == 4 and i==6:
#	 cbar=plt.imshow(kall,interpolation='nearest')
#	 cbar.set_label(r'$\kappa$')
#	 colorbar()
#	 plt.show()
      deltk[i] = np.mean(khig)-np.mean(klow)
    plt.plot(lam,deltk,'-',linewidth=2.5,color=color[j],\
             label=r'$\sigma_v=$'+str(vd[j]))
  plt.xlabel(r'$\mathrm{\lambda}$',fontsize=15)
  plt.ylabel(r'$\mathrm{\delta\kappa}$',fontsize=15)
  plt.xlim(0.1,0.8)
  plt.legend(loc='upper left')
  plt.show()
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
  vdis= 300.0
  vrot= 100.0
  lam = 0.5
  phi= 0
  zl  = 0.2
  zs  = 0.4
  noise = False
  #---II:lambda vs delta_kappa relation without noise---------
  #lamvsdkapp=lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise)
  #---end lambda vs delta_kappa relation without noise---------
  #---III:test Kaiser Squires gamma to kappa conversion-----
  wlsis= wl.weaksis(vrot,vdis,lam,zl,zs)
  kap  = wlsis.TotalKappa(xi1,xi2,phi,noise)
  gamma= wlsis.TotalShear(xi1,xi2,phi,noise)
  ke,kb = wlsis.kaisersquires(xi1,xi2,gamma,noise,galsim)

  plt.imshow(ke,interpolation='nearest')
  plt.colorbar()
  plt.show()
  plt.imshow(kap,interpolation='nearest')
  plt.colorbar()
  plt.show()
  #---end of Kaiser Squires test---------------------
  #---IV:test filters(Gauss,Wiener,& maximum entropy)-----
  #---end of filters(Gauss,Wiener,& maximum entropy)-----
  #---V:stacked simulation------------------------
  #kap = stackedsims(vrot,vdis,xi1,xi2,nstack)
  #-----end ofstacked simulation------------------------
  #---VI:select_regions------------------------------
  #kap   = selectregions(kap)
  #klow  = kap[nn/2:nn,:]
  #khig  = kap[0:nn/2,:]
  #ih    = np.abs(khig)>0.0
  #il    = np.abs(klow)>0.0
  #print(np.mean(khig[ih]/nstack),np.std(khig[ih]/nstack))
  #print(np.mean(klow[il]/nstack),np.std(klow[il]/nstack))
  #--end select region ----------------------------------

if __name__=="__main__":
    main()
   
