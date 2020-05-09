#!/home/wtluo/anaconda/bin/python2.7

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import weaksis as wl

def plot_shears_kappa(kappa,shear1,shear2):

    g1 = shear1
    g2 = -shear2
    plt.figure(figsize=(10,10),dpi=80)
    plt.imshow(kappa,interpolation='NEAREST')
    plt.colorbar()
    nnn  = 64
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
           plt.plot([md_x,ed_x],[md_y,ed_y],'w-')
           plt.plot([md_x,st_x],[md_y,st_y],'w-')

    plt.xlim(0,64)
    plt.ylim(0,64)
#----------------------------------------------------
def main():
  xi  = np.linspace(50,300,50)
  xi1 = np.linspace(-32,32,64)
  xi2 = np.linspace(-32,32,64)

  vdis= 300.0
  vrot= 0.0
  zl  = 0.2
  zs  = 0.4

  wlsis= wl.weaksis(vrot,vdis,zl,zs)
  esd  = wlsis.sisESD(xi)
  #plt.plot(xi,esd)
  #plt.xscale('log')
  #plt.yscale('log')
  #plt.xlabel(r'$\mathrm{\xi}(arcmin)$')
  #plt.ylabel(r'$\mathrm{ESD(M_{\odot}h/pc^2)}$')
  kap  = wlsis.sisKappa(xi1,xi2)
  gm   = wlsis.sisShear(xi1,xi2)
  shear1 = gm['g_1']
  shear2 = gm['g_2']
  plot_shears_kappa(kap,shear1,shear2)
  plt.xlim(0,64)
  plt.ylim(0,64)
  plt.show()

if __name__=="__main__":
   main()

