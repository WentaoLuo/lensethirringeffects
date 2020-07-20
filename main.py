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
from scipy.optimize import minimize
import emcee
import corner

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

def stackedsims(vdis,lam,xi1,xi2,nstack,sim_id,phi):
  nx = len(xi1) 
  ny = len(xi2) 
  kap= np.zeros((nx,ny))
  zl  = 0.2
  zs  = 0.4
  val1= np.zeros(nstack)
  val2= np.zeros(nstack)
  vrot= 100.0
  for i in range(nstack):
     if sim_id==1:
       lam = lam
     if sim_id==2:
       lam = lam+np.random.normal(loc=0.0,scale=0.05)
     if sim_id==3:
       lam = lam
     if sim_id==4:
       lam = lam+np.random.normal(loc=0.0,scale=0.05)
     if sim_id==5:
       lam = lam+np.random.normal(loc=0.0,scale=0.05)
     noise= True
     wlsis= wl.weaksis(vrot,vdis,lam,zl,zs)
     kap  = kap+wlsis.TotalKappa(xi1,xi2,phi*pi/180.0,noise)
     tmp  = wlsis.TotalKappa(xi1,xi2,phi*pi/180.0,noise)
     khig   = tmp[nx/2:nx,:]
     klow   = tmp[0:nx/2,:]
     val1[i]= np.mean(klow)
     val2[i]= np.mean(khig)
  return {'meankappa':kap/nstack,'val1':val1,'val2':val2}
#----------------------------------------------------
def lnlike(theta,phi,deltk,error):
  lam,vd = theta
  model  = lam*2.832819e-5*vd*np.cos(phi*pi/180.0)
  diff   = -0.5 * np.sum((deltk - model)**2 /error/error + np.log(error*error))
  return diff.sum()
def lnprob(theta,phi,deltk,error):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp+lnlike(theta,phi,deltk,error)
def lnprior(theta):
  vd,lam = theta
  if vd==800.0:
    if 700<vd<900 and 0.0<lam<0.5:
       return 0.0
  if vd==1000.0:
    if 800<vd<1200 and 0.0<lam<0.5:
       return 0.0
  return -np.inf
#----------------------------------------------------
def mcmcs(xi1,xi2,lam,vdis,psi,zl,zs,noise,nstack,sim_id):
  phi   = np.linspace(-90,90,50)
  #print(phi)
  deltk = np.zeros(len(phi))
  err   = np.zeros(len(phi))
  ngals = 50.0
  nshape= 0.3
  area  = 16.0*8.0
  #nstack= 10
  error = nshape/np.sqrt(ngals*area*nstack)
  #sim_id= 1
  vdis0 = vdis
  lam0  = lam
  for k in range(len(phi)):
    struc   = stackedsims(vdis,lam,xi1,xi2,nstack,sim_id,phi[k])
    kap     = struc['meankappa']
    val1    = struc['val1']
    val2    = struc['val2']
    mv1     = np.mean(val1)
    mv2     = np.mean(val2)
    deltk[k]= mv1-mv2
    err[k]  = error
  model= lam*5.69278126993e-10*vdis*vdis*np.cos(phi*pi/180.0)
  dtheta=(np.sum(phi*deltk)/np.sum(deltk))
  #print(dtheta)
  pars= np.array([vdis0,lam0])
  plt.plot(phi,model,'r-',lw=3.0)
  plt.errorbar(phi,deltk,yerr=err,fmt='k.',ms=10.0,elinewidth=1.5)
  #plt.plot([dtheta,dtheta],[-0.002,0.01],'k--')
  plt.xlim(-90,90)
  plt.ylim(-0.0001,0.002)
  plt.xlabel(r'$\mathrm{\delta\theta}$',fontsize=15)
  plt.ylabel(r'$\mathrm{\delta\kappa cos(\delta\theta)}$',fontsize=15)
  plt.show()
  ndim,nwalkers = 2,200
  pos = [pars+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
  sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(phi,deltk,err))
  sampler.run_mcmc(pos,3000)

  burnin = 1000
  samples=sampler.chain[:,burnin:,:].reshape((-1,ndim))
  vd,lamda = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))
  #print 'Vdis: ',vd
  #print 'lam: ',lam
  #print vdis0,nstack, dtheta,vd[0],vd[1],vd[2],lamda[0],lamda[1],lamda[2]
  fig = corner.corner(samples,labels=["VD",r"$\lambda$"],\
               truths=[vd[0],lamda[0]])
  plt.savefig('mcmc.eps')
  plt.show()
  return 0
def lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise):
  lam  = np.array([0.8,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
  vd   = np.array([1300.0,500.0,600.0,700.0,800.0,900.0,1000.0,1300.0])
  #lam  = np.linspace(0.1,0.5,6)
  #vd   = np.linspace(1000.0,1300.0,6)
  #vd   = np.linspace(200.0,1300.0,90)
  phi  = np.linspace(-90,90,50)
  #print vd[0],lam[0]
  color= np.array(['y','g','b','pink','m','cyan','k','r'])
  #deltk= np.zeros(len(lam))
  #deltk0= np.zeros(len(lam))
  deltk0= np.zeros(len(phi))
  deltk= np.zeros(len(phi))
  coeff= zeros((8,2))
  model= 0.5*5.69278126993e-10*1000.0*1000.0*np.cos(phi*pi/180.0)
  error= np.zeros(50)+0.3/np.sqrt(100.0*16.0*8.0)
  plt.figure()
  #wlsis= wl.weaksis(vrot,1000,0.2,zl,zs)
  #kap0  = wlsis.GRMKappa(xi1,xi2,0.0*phi[0]*pi/180.)
  #nn    = len(kap0[:,0])
  #khig0 = kap0[0:nn/2,:]
  #klow0 = kap0[nn/2:nn,:]
  #deltk0= np.mean(khig0)-np.mean(klow0)
  #for j in range(len(vd)):
  for j in range(1):
    #for i in range(len(lam)):
    for i in range(1):
      for k in range(len(phi)):
      #for k in range(1):
	#print(phi[k],lam[i])
        wlsis= wl.weaksis(vrot,vd[j],lam[i],zl,zs)
        kap0  = wlsis.GRMKappa(xi1,xi2,phi[k]*pi/180.)
        #kap0  = wlsis.GRMKappa(xi1,xi2,0.0*pi/180.)
        nn    = len(kap0[:,0])
        khig0 = kap0[0:nn/2,:]
        klow0 = kap0[nn/2:nn,:]
        #deltk0[i]= np.mean(khig0)-np.mean(klow0)
	#print deltk0
        wlsis= wl.weaksis(vrot,vd[j],lam[i],zl,zs)
        #kap  = wlsis.TotalKappa(xi1,xi2,phi[k]*pi/180.0,noise)
        #kap  = wlsis.TotalKappa(xi1,xi2,0.0*pi/180.0,noise)
        kap  = wlsis.TotalKappa(xi1,xi2,phi[k]*pi/180.0,noise)
      #kap  = selectregions(kap)
      #klow = kap['klow']
      #khig = kap['khig']
      #kall = kap['all']
        #nn   = len(kap[:,0])
        kall = kap
        khig = kap[0:nn/2,:]
        klow = kap[nn/2:nn,:]
        #deltk[k]= (np.mean(khig)-np.mean(klow))/deltk0
        deltk[k]= (np.mean(khig)-np.mean(klow))
        deltk0[k]= (np.mean(khig0)-np.mean(klow0))
        #deltk0[i]= (np.mean(khig0)-np.mean(klow0))
	#print deltk0[k]
        #deltk[i]= (np.mean(khig)-np.mean(klow))
      #if j == 4 and i==6:
#	 cbar=plt.imshow(kall,interpolation='nearest')
#	 cbar.set_label(r'$\kappa$')
#	 colorbar()
#	 plt.show()
        #deltk[i]= np.mean(khig)-np.mean(klow)
    #coeff[j,:]= np.polyfit(lam,deltk0,1)
    #print coeff
    #print '------First-----------------------'
    #print j, coeff[j,0],vd[j],coeff[j,1]
    #plt.plot(lam,deltk0,'o',ms=10,linewidth=2.5,\
    #         color=color[j],label=r'$\sigma_v=$'+str(np.round(vd[j],1)))
    #plt.plot(lam,lam*5.69278126993e-10*vd[j]*vd[j]-lam*vd[j]*7.19674146e-22+1.78901103e-19,'k-')
    #plt.plot(lam,lam*5.69278126993e-10*vd[j]*vd[j],'k-')
    #plt.plot(lam,lam*coeff[j,0],'k-')
  #secon = np.polyfit(vd,coeff[:,0],2)
  #print secon
      #plt.plot(phi,deltk,'-',linewidth=3,color=color[j],label=r'$\mathrm{\sigma_v=}$'+str(np.round(vd[j],1)))
      #plt.plot(phi,deltk0,'-',linewidth=3,color=color[i],label=r'$\mathrm{\lambda=}$'+str(np.round(lam[i],2)))
      #plt.plot(phi,deltk,'k-',linewidth=3,label=r'Noisy')
      #plt.plot(phi,deltk0,'r-',linewidth=3,label=r'No noise')
    #plt.plot(phi,np.cos(phi*pi/180.0),'r.',ms=5)
  #np.random.seed(42)
  #nll = lambda *args: -lnlike(*args)
  #initial = np.array([0.2, 1200.0]) + 10.5 * np.random.randn(2)
  #soln = minimize(nll, initial, args=(phi, deltk, error))
  #lam_ml, vd_ml = soln.x
  #print(lam_ml)
  #print(vd_ml)
  #print(np.sum(phi*deltk)/np.sum(deltk))
  #model2= lam_ml*2.832819e-5*vd_ml*np.cos(phi*pi/180.0)
  #plt.plot(vd,coeff[:,0],'ro',ms=10)
  #plt.plot(vd,secon[0]*vd+secon[1],'k-',ms=10)
  #plt.xlabel(r'$\mathrm{\lambda}$',fontsize=15)
  #plt.ylabel(r'$\mathrm{\delta\kappa}$',fontsize=15)
  #plt.xlim(0.09,0.81)
  #plt.ylim(0.0,0.0008)
  #plt.legend(loc='upper left')
  #plt.show()
  #plt.plot(vd,coeff[:,0],'ro',ms=10)
  #plt.plot(vd,secon[0]*vd*vd,'k-',ms=10)
  #plt.xlabel(r'$\mathrm{\sigma_v}$',fontsize=15)
  #plt.ylabel(r'$\mathrm{coeffs}$',fontsize=15)
  #plt.show()
  #plt.xlabel(r'$\mathrm{\delta\theta}$',fontsize=15)
  #plt.ylabel(r'$\mathrm{\delta\kappa cos(\delta\theta)}$',fontsize=15)
  #print(deltk-model,np.max(deltk-model))
  plt.subplot(2,1,1)
  plt.errorbar(phi,deltk,yerr=error,fmt='ko',elinewidth=2.0,label='Mock')
  plt.plot(phi,model,'r-',linewidth=3,label='Input')
  #plt.plot(phi,model2,'b--',linewidth=3,label='MaxLike')
  plt.xlim(-90,90)
  plt.ylim(-0.0001,0.001)
  plt.ylabel(r'$\mathrm{\delta\kappa cos(\delta\theta)}$',fontsize=15)
  plt.xticks(())
  plt.legend(loc='upper left')
  plt.subplot(2,1,2)
  plt.errorbar(phi,deltk-model,yerr=error,fmt='ko',elinewidth=2.0,label='Mock')
  plt.plot(phi,np.zeros(50),'r-')
  plt.ylim(-0.0001,0.0006)
  plt.xlim(-90,90)

  plt.xlabel(r'$\mathrm{\delta\theta}$',fontsize=15)
  plt.ylabel(r'$\mathrm{\delta\kappa cos(\delta\theta)}$',fontsize=15)
  #plt.ylabel(r'$\mathrm{Residual}$',fontsize=15)
  plt.legend(loc='upper left')
  plt.show()
  #coeffs = np.polyfit(lam,deltk)
  return 0
#----------------------------------------------------
def main():
  xi  = np.linspace(0.02,10,50)
  nn  = 8
  ang = 16
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
  vdis= 1000.0
  vrot= 100.0
  zl  = 0.2
  zs  = 0.4
  phi = 0.0
  noise = True
  #noise = False
  #---II:lambda vs delta_kappa relation without noise---------
  #lamvsdkapp=lambda_deltakappa_plot(xi1,xi2,vrot,vdis,phi,zl,zs,noise)
  #---end lambda vs delta_kappa relation without noise---------
  #---III:stacked simulation------------------------
  #sim_id= 1 
  #ngals = 50.0
  #nshape= 0.3
  #area  = 16.0*8.0
  #nstack= 10
  #error = 0.3/np.sqrt(ngals*area*nstack)
  #struc    = stackedsims(vdis,lam,xi1,xi2,nstack,sim_id)
  #kap     = struc['meankappa']
  #val1    = struc['val1']
  #val2    = struc['val2']
  #khig   = kap[nn/2:nn,:]
  #klow   = kap[0:nn/2,:]
  #mv1    = np.mean(val1)
  #mv2    = np.mean(val2)
  #print(np.mean(val1),np.mean(val2),np.sqrt(np.var(val1)+np.var(val2)))
  #print(np.mean(val1)-np.mean(val2),error)
  #plt.hist(val1,20,histtype='step',facecolor='none',lw=3,label='High',color='r')
  #plt.hist(val2,20,histtype='step',facecolor='none',lw=3,label='Low',color='b')
  #plt.xlabel(r'$\mathrm{<\kappa>}$',fontsize=15)
  #plt.ylabel(r'$\mathrm{Number}$',fontsize=15)
  #plt.ylim(0,40)
  #plt.plot([mv1,mv1],[0,50],'r--',lw=3)
  #plt.plot([mv2,mv2],[0,50],'b--',lw=3)
  #plt.imshow(kap,interpolation='nearest')
  #plt.colorbar()
  #plt.title(r'$\mathrm{Sim\_4}$')
  #plt.legend()
  #plt.show()
  #-----end of stacked simulation------------------------
  #---IV: MCMC------------------------
  import sys
  zl     = 0.2
  zs     = 0.4
  noise  = False
  vdis   = float(sys.argv[1])
  phi    = 0.0
  lam    = 0.8
  sim_id = int(sys.argv[2])
  nstack = int(sys.argv[3])
  results= mcmcs(xi1,xi2,lam,vdis,phi,zl,zs,noise,nstack,sim_id)
  #for i in range(10):
  #   results= mcmcs(xi1,xi2,lam,vdis,phi,zl,zs,noise,nstack,sim_id)
  #---END of MCMC

if __name__=="__main__":
    main()
   
