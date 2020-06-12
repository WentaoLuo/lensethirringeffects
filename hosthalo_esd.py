#!/home/wtluo/anaconda/bin/python2.7

#----README--------------------------------------
# This is a small code to convert 2D dark matter
# particle distribution to 2D ESD from simulation.
# This demo only plots the profile given a sample
# halo from illustris simulation snap128-halo50.
# You can also do the stacking bby modify this version.
# Have fun!
#-------------------------------------------------
import numpy as np
import illustris_python as illus
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3d
import mycosmology as cosmos
pi =np.pi

vc      = 2.9970e5
h       = 0.6774
w       = -1.0
omega_m = 0.3089
omega_l = 1.0-omega_m
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
ns      = 0.9667
alphas  = -0.04
sigma8  = 0.8159
Gg    =6.67408e-11
ckm   =3.240779e-17
ckg   =5.027854e-31

fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm)
#---------------------------------------------------
#from sklearn.cluster import KMeans
def kmeans(x,y):
  pos   = np.array([x,y]).T
  kmeans= KMeans(n_clusters=8,random_state=0,max_iter=2000).fit(pos)
  cntrs = kmeans.cluster_centers_
  xc    = cntrs[:,0]
  yc    = cntrs[:,1]

  print(cntrs) 
  return {'xc':xc,'yc':yc}
#----------------------------------------------------
def haloparams(logM,con):
   ccon      = con*(10.0**(logM-14.0))**(-0.11)
   efunc     = 1.0/np.sqrt(omega_m*(1.0+zl)**3+\
                omega_l*(1.0+zl)**(3*(1.0+w))+\
                omega_k*(1.0+zl)**2)
   rhoc      = rho_crt0/efunc/efunc
   omegmz    = omega_m*(1.0+zl)**3*efunc**2
   ov        = 1.0/omegmz-1.0
   dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
   rhom      = rhoc*omegmz

   r200 = (10.0**logM*3.0/200./rhom/pi)**(1./3.)
   rs   = r200/ccon
   delta= (200./3.0)*(con**3)\
          /(np.log(1.0+con)-con/(1.0+con))

   amp  = 2.0*rs*delta*rhoc*10e-14
   res  = np.array([amp,rs,r200])

   return res
def nfwfuncs(r,rsub):
    x   = r/rsub
    x1  = x*x-1.0
    x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
    x3  = np.sqrt(np.abs(1.0-x*x))
    x4  = np.log((1.0+x3)/(x))
    s1  = r*0.0
    s2  = r*0.0
    ixa = x>0.
    ixb = x<1.0
    ix1 = ixa&ixb
    s1[ix1] = 1.0/x1[ix1]*(1.0-x2[ix1]*x4[ix1])
    s2[ix1] = 2.0/(x1[ix1]+1.0)*(np.log(0.5*x[ix1])\
              +x2[ix1]*x4[ix1])

    ix2 = x==1.0
    s1[ix2] = 1.0/3.0
    s2[ix2] = 2.0+2.0*np.log(0.5)

    ix3 = x>1.0
    s1[ix3] = 1.0/x1[ix3]*(1.0-x2[ix3]*np.arctan(x3[ix3]))
    s2[ix3] = 2.0/(x1[ix3]+1.0)*(np.log(0.5*x[ix3])+\
              x2[ix3]*np.arctan(x3[ix3]))

    res = s2-s1
    return res
def NFWcen(theta,r):
  logM,con = theta
  amp,rs,r200   = haloparams(logM,con)
  res           = amp*nfwfuncs(r,rs)
  return res


def dens2kappa(x,y,mass):
  zl    = 0.2
  zs    = 0.4
  nn    = 64
  phi  = np.linspace(-90,90,10)
  dl    = cosmos.Da(zl)
  ds    = cosmos.Da(zs)
  Sigc  = fac*ds/(dl*(ds-dl))/(1.0+zl)/(1.0+zl)
  xx    = np.linspace(-4,4,nn) # griding
  pixs  = 64.0/float(nn)                # (Mpc/h)^2
  xv,yv = np.meshgrid(xx,xx)
  kappa = np.zeros((nn,nn))
  deltk = np.zeros(10)
  #print(Sigc)
  for k in range(len(phi)):
    x   = x*np.cos(phi[k]*pi/180.0)+y*np.sin(phi[k]*pi/180.0)
    y   = -x*np.cos(phi[k]*pi/180.0)+y*np.cos(phi[k]*pi/180.0)
    for i in range(nn-1):
      for j in range(nn-1):
	 ixa= x>=xv[i,j]
	 ixb= x<=xv[i,j+1]
	 iya= y>=yv[i,j]
	 iyb= y<=yv[i+1,j]
	 ixy= ixa&ixb&iya&iyb
	 nps= float(len(x[ixy]))
	 #print(nps)
	 kappa[i,j]=1e-12*mass*nps/pixs/Sigc
    khig = kappa[0:nn/2,:]
    klow = kappa[nn/2:nn,:]
    deltk[k]= (np.mean(khig)-np.mean(klow))

  plt.plot(phi,deltk,'k-',linewidth=3,label=r'kappa')
  plt.xlabel(r'$\mathrm{\delta\theta}$',fontsize=15)
  plt.ylabel(r'$\mathrm{\delta\kappa cos(\delta\theta)}$',fontsize=15)
  plt.xlim(-90,90)
  plt.title('illustris halo')
  #plt.legend(loc='upper left')
  plt.show()
  return {'phi':phi,'dkap':deltk}
def dens2esd(x,y,rb,nbin,mass):
  dis = np.sqrt(x*x+y*y) 
  esd = np.zeros(nbin)
  for i in range(nbin):
      ixa = dis >= rb[i]
      ixb = dis <= rb[i+1]
      iy  = dis <= (rb[i]/2.0+rb[i+1]/2.0)
      ix  = ixa&ixb
      denin = mass*len(x[iy])/2.0/pi/(rb[i]/2.0+rb[i+1]/2.0)**2
      annul = 2.0*pi*(rb[i+1]*rb[i+1]-rb[i]*rb[i])
      denat = mass*len(x[ix])/annul
      esd[i]= denin-denat
  return esd/10e+12  # 10e+12 factor comes from Mpc^2 to pc^2
def main():
 SnapID = 91
 #HaloID = 401 

 mass   = 3.8*1e+9  # particle mass 
 mss    = 7.0*1e+8  # stellar particle mass 
 
 baseDir= '../'
 darkm  = "Massbin_0dm.dat"
 stellar= "Massbin_0st.dat"
 fdark  = open(darkm,'w')
 #fstar  = open(stellar,'w')
 for i in range(50000):
   HaloID = 1+i 
   dm     = illus.snapshot.loadHalo(baseDir,SnapID,HaloID,'dm') 
   ms     = illus.snapshot.loadHalo(baseDir,SnapID,HaloID,'stars') 
   gs     = illus.snapshot.loadHalo(baseDir,SnapID,HaloID,'gas') 
   npp    = dm["count"]
   nsp    = ms["count"]
   x1     = dm["Coordinates"][:,0] # kpc/h
   x2     = dm["Coordinates"][:,1] # kpc/h
   x3     = dm["Coordinates"][:,2] # kpc/h
   xs1    = ms["Coordinates"][:,0] # kpc/h
   xs2    = ms["Coordinates"][:,1] # kpc/h
   xs3    = ms["Coordinates"][:,2] # kpc/h
   #xg1    = gs["Coordinates"][:,0] # kpc/h
   #xg2    = gs["Coordinates"][:,1] # kpc/h
   #xg3    = gs["Coordinates"][:,2] # kpc/h
   xc1    = np.mean(x1)
   xc2    = np.mean(x2)
   xc3    = np.mean(x3)
   Mh  = np.log10(npp*mass) 
   Ms  = np.log10(nsp*mss) 
   print HaloID,Mh,Ms
 #  if Mh<=11.75 and Mh>=11.61:# and Ms>=10.5 and Ms<=10.8:
 #    for j in range(len(x1)):
 #       fdark.write("%10.6f  %10.6f  %10.6f\n"%((x1[j]-xc1)/1000.0,(x2[j]-xc2)/1000.0,(x3[j]-xc3)/1000.0))
     #for ix in range(len(xs1)):
     #   fstar.write("%10.6f  %10.6f  %10.6f\n"%((xs1[ix]-xc1)/1000.0,(xs2[ix]-xc2)/1000.0,(xs3[ix]-xc3)/1000.0))
	#print (xs1[ix]-xc1)/1000.0,(xs2[ix]-xc2)/1000.0,(xs3[ix]-xc3)/1000.0
   d2kap = dens2kappa((x1-xc1)/1000.0,(x3-xc3)/1000.0,mass)
   #plt.imshow(d2kap,interpolation='nearest')
   #plt.title('illustrisTNG300-300 halo kappa map')
   #plt.colorbar()
   plt.show()
   plt.plot((x1-xc1)/1000.0,(x3-xc3)/1000.0,'k.',label='dm')
   plt.plot((xs1-xc1)/1000.0,(xs3-xc3)/1000.0,'r.',label='stellar')
   #plt.plot((xg1-xc1)/1000.0,(xg2-xc2)/1000.0,'g.')
   #plt.plot(xs1,xs2,'r.')
   plt.title('HaloID'+str(HaloID+i))
   plt.xlim([-3,3])
   plt.xlabel('X Mpc/h')
   plt.ylim([-3,3])
   plt.ylabel('Y Mpc/h')
   #plt.legend()
   plt.savefig('halo_'+str(HaloID+i)+'.png')
   plt.show()
 #fdark.close()
 #fstar.close()
 """ 
 GroupFirstSub = il.groupcat.loadHalos(basePath,snapNum,fields=['GroupFirstSub'])
 ptNumDm       = il.snapshot.partTypeNum('dm')
 ptNumGas      = il.snapshot.partTypeNum('gas')
 ptNumStars    = il.snapshot.partTypeNum('stars')
 all_fields = il.groupcat.loadSingle(basePath,snapNum,subhaloID=GroupFirstSub[i])
 stars_mass = all_fields['SubhaloMassInHalfRadType'][ptNumStars]
 gas_mass   = all_fields['SubhaloMassInHalfRadType'][ptNumGas]
 dm_mass    = all_fields['SubhaloMassInHalfRadType'][ptNumDm]
 """
#---plots to check if everything is in control-------- 
 #print xc1,xc2,xc3
 #fig,    = plt.figure()
 #axs    = p3d.Axes3D(fig)
 #axs.plot((x1-xc1)/1000.,(x2-xc2)/1000.,'k.')
 #axs.set_xlim3d([-2,2])
 #axs.set_xlabel('X kpc/h')
 #axs.set_ylim3d([-2,2])
 #axs.set_xlabel('Y kpc/h')
 #axs.set_zlim3d([-2,2])
 #axs.set_xlabel('Z kpc/h')
 #plt.plot((x1-xc1)/1000.,(x2-xc2)/1000.,'k.')
 #plt.xlim([-2,2])
 #plt.xlabel('X kpc/h')
 #plt.ylim([-2,2])
 #plt.ylabel('Y kpc/h')
 #plt.show()
#--------------------------------------------------------
#---starts to calculate the ESD--------------------------
 """
 Rmax = 1.0
 Rmin = 0.01
 Nbin = 10
 rbin = np.zeros(Nbin+1)
 Rp   = np.zeros(Nbin)
 xtmp = (np.log10(Rmax)-np.log10(Rmin))/Nbin
 for i in range(Nbin):
     ytmp1 = np.log10(0.01)+float(i)*xtmp
     ytmp2 = np.log10(0.01)+float(i+1)*xtmp
     rbin[i] = 10.0**ytmp1
     rbin[i+1] = 10.0**ytmp2
     Rp[i] =(rbin[i]+rbin[i+1])/2.0
 halonum=[94.0,177.0,87.0,24.0,94.0]
 x1,x2,x3    = np.loadtxt('Massbin_2dm.dat',unpack=True)
 xs1,xs2,xs3 = np.loadtxt('Massbin_2st.dat',unpack=True)
 #centers     = kmeans(xs1,xs2)
 #xsc1        = centers['xc'] 
 #xsc2        = centers['yc'] 
 #print kmeans
 ixa          = np.sqrt(xs1*xs1+xs2*xs2)>=0.1
 ixb          = np.sqrt(xs1*xs1+xs2*xs2)<=0.1
 print float(len(xs1[ixa]))/float(len(xs1[ixb]))
 #plt.plot(x1,x2,'k.',label='dark matter')
 #plt.plot(xs1,xs2,'r.',label='stars')
 #plt.plot(xsc1,xsc2,'r+',ms=10)
 #plt.xlim([-1,1])
 #plt.xlabel('X kpc/h')
 #plt.ylim([-1,1])
 #plt.ylabel('Y kpc/h')
 #plt.legend()
 #plt.savefig('mass33bin_dmstar.png')
 #plt.show()
 ESD = dens2esd((x1),(x2),rbin,Nbin,mass)*0.677*0.677
 Mh  = np.log10(float(len(x1))*mass/halonum[4]) 
 con = 5.26*(10.00**(Mh-14.0))**(-0.1)
 #con = 10.0
 #print(Mh,con)
 Model= NFWcen([Mh,con],Rp)
 ratio= np.mean(ESD[4:10]/Model[4:10])
#---plot check if it make sense----------------------------
 plt.plot(Rp,ESD/ratio,'r-',linewidth=2.5,label='simulation')
 plt.plot(Rp,Model,'k-',linewidth=2.5,label='model')
 plt.xlabel('Rp Mpc/h')
 plt.ylabel('ESD ')
 plt.xlim(0.01,1.)
 plt.ylim(0.1,200)
 plt.legend()
 plt.xscale('log')
 plt.yscale('log')
 plt.show()
 """ 
#---END----------------------------------------------------
if __name__=='__main__':
   main()
