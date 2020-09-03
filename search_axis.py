#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt

def theta(x,y,xc,yc,pmem):
        idx= pmem>=0.8
        n  =np.size(pmem[idx])
        pm =pmem[idx]
        qxx=0.0
        qyy=0.0
        qxy=0.0
        T  =0.0
        rr =(x-xc)*(x-xc)+(y-yc)*(y-yc)
        for i in range(n):
                if ~np.isnan(x[i]) and rr[i]!=0.0 and pm[i]>0.2:
                        qxx=qxx+(x[i]-xc)*(x[i]-xc)*pm[i]/rr[i]
                        qyy=qyy+(y[i]-yc)*(y[i]-yc)*pm[i]/rr[i]
                        qxy=qxy+(x[i]-xc)*(y[i]-yc)*pm[i]/rr[i]
                        T  =T+pm[i]#/rr[i]
        qxx=qxx/T
        qyy=qyy/T
        qxy=qxy/T

        theta=0.5*np.arctan(2.*qxy/(qxx-qyy))
        e1   =(qxx-qyy)/(qxx+qyy)
        e2   =(qxy)/(qxx+qyy)
        return theta

#-------------------------------------------------------------------
pi   = np.pi
data = np.loadtxt('planck18_galaxy',unpack=True,skiprows=1)
gid  = data[0,:]
ra   = data[1,:] 
dec  = data[2,:]
z    = data[3,:]
brg  = data[9,:]
rich = data[10,:]

idx  = gid==1.0

print(len(ra[idx]))
ra   = ra[idx]
dec  = dec[idx]
z    = z[idx]
brg  = brg[idx]
rich = rich[idx]
ixc  = brg==1
rac  = ra[ixc]
decc = dec[ixc]
zc   = z[ixc]
xx   = (ra-rac)*np.cos(pi*(dec-decc)/180.0)
yy   = dec-decc
zz   = z-zc

plt.plot(ra-rac,dec-decc,'ro')
plt.show()
"""
sig  = np.std(zz)
print(sig)
print(sig*300000.0)
print(0.00065*300000.0)
print(15.0005)
plt.hist(zz,10,histtype='stepfilled',alpha=0.4,facecolor='blue')
plt.plot([0,0],[0,200],'k-',lw=3)
plt.ylim(0,150)
plt.xlabel(r'$\mathrm{\Delta z}$',fontsize=20)
plt.ylabel(r'$\mathrm{Number}$',fontsize=20)
plt.savefig('group_vdis.eps')
plt.show()


#zz   = z
#Zmax = np.max(zz)
Zmax = 0.0001
Z0   = -0.00065
the  = 0.15
#model= Zmax+Z0*np.sin()
nbin = 10
phi  = np.linspace(-0.5*pi,0.5*pi,nbin)
model= Zmax+Z0*np.sin(phi-the)
nsamp  = 200
ngals= len(xx)
dzmn = np.zeros((nsamp,nbin))  
dzmd = np.zeros((nsamp,nbin))  
dzerr= np.zeros((nsamp,nbin))  
for j in range(nsamp):
  for i in range(nbin):
    irand = np.random.randint(low=0,high=ngals-1,size=ngals)
    xxr    = xx[irand]
    yyr    = yy[irand]
    zzr    = zz[irand]
    ix_up = yyr>= xxr*np.sin(phi[i])
    ix_dw = yyr<= xxr*np.sin(phi[i])   
    dzmn[j,i] = np.mean(zzr[ix_up])-np.mean(zzr[ix_dw])
    #dzmd[i] = np.median(zz[ix_up])-np.median(zz[ix_dw])
    #dzerr[i] = np.std(zz-zc)
    #dzerr[i] = np.percentile(zz-zc,68)
dzmd = np.array([np.mean(dzmn[:,i]) for i in range(nbin)])
dzme = np.array([np.std(dzmn[:,i]) for i in range(nbin)])
#plt.plot(phi,dz,'k-')
plt.plot(phi,model,'r-',lw=3,label='Model')
plt.plot([-2,2],[Zmax,Zmax],'k--',lw=2,label=r'$\mathrm{dz_{off}}$')
plt.plot([-2,2],[0,0],'k-',lw=2)
plt.plot([the,the],[-0.0009,0.001],'b--',lw=2,label=r'$\mathrm{\phi_0}$')
#plt.errorbar(phi,dzmn,yerr=dzerr/5,fmt='k.',ms=10)
#plt.plot(phi,dzmn,'k*',ms=20,label='Data')
plt.errorbar(phi,dzmd,yerr=dzme,fmt='k*',ms=10,label='Data')
#plt.errorbar(phi,dzmd,yerr=dzerr/5,fmt='k+',ms=10)
plt.xlim(-2.0,2.0)
plt.ylim(-0.0009,0.0009)
plt.xlabel(r'$\mathrm{\phi}$',fontsize=15)
plt.ylabel(r'$\mathrm{dz}$',fontsize=15)
plt.legend(numpoints=1)
plt.savefig('group_rotation.eps')
plt.show()

xx   = np.linspace(-4,4,10)
yy0  = 0.0*xx*np.cos(0.0*np.pi/180.0)
yy1  = xx*np.sin(45.0*np.pi/180.0)
yy2  = xx*np.sin(-45.0*np.pi/180.0)
plt.plot((ra-rac)*np.cos(np.pi*(dec-decc)/180.0),dec-decc,'k.')
plt.plot(xx,yy0,'k-')
plt.plot(xx,yy1,'k--')
plt.plot(xx,yy2,'k--')
plt.show()
"""



