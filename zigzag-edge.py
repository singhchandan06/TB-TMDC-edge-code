# import the python library
from numpy import *
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
from numpy import linalg as LA
import random
import math 
import time
# write a function file for zigzag tight-binding Hamiltonian with hopping integrals and on-site energies term
def ham_zigzag(kx):
    
    t0 = -0.146; t1 = -0.114; t2 = 0.506; t11 =  0.085; t12 = 0.162 ; t22 =  0.073;
    r0 = 0.06;  r1 = -0.236; r2 = 0.067; r11 = 0.016; r12 = 0.087; u0 = -0.038;
    u1 = 0.046; u2 = 0.001; u11 = 0.266; u12=-0.176; u22 = -0.150;E1 = 0.683; E2 =  1.707;
    lam = 0.073
    #lam = 0
    
    N = 8
    
    alpha = 0.5*kx*a
    
    ##hopping matrix###
    h1 = zeros([3,3],dtype='complex')
    h2 = zeros([3,3],dtype='complex')
    
    #######hopping matrix elements[h1] #################
    ##diagonal elements#################################
    h1[0,0] = E1 + 2*t0*cos(2*alpha)
    h1[1,1] = E2 + 2*t11*cos(2*alpha)
    h1[2,2] = E2 + 2*t22*cos(2*alpha)
    
    #########off diagonal elements#######################
    h1[0,1] = 2*sin(2*alpha)*t1*1j
    h1[0,2] = 2*cos(2*alpha)*t2
    h1[1,2] = 2*sin(2*alpha)*t12*1j
    
    h1[2,1] = h1[1,2].T.conj()
    h1[1,0] = h1[0,1].T.conj()
    h1[2,0] = h1[0,2]
    
    #######hopping matrix elements[h2] #################
    ##########diagonal elements#########################
    h2[0,0] = 2*cos(alpha)*t0
    h2[1,1] = 0.5*(t11 + 3*t22)*cos(alpha)
    h2[2,2] = 0.5*cos(alpha)*(3*t11 + t22)
    
    ####off diagonal elements
    h2[0,1] = (t1 - sqrt(3)*t2)*sin(alpha)*1j
    h2[0,2] = -0.5*cos(alpha)*(sqrt(3)*t1 + t2)
    h2[1,2] = -sin(alpha)*((sqrt(3)/2)*t11 + 2*t12 - (sqrt(3)/2)*t22)*1j
    
    h2[1,0] =  -(t1 + sqrt(3)*t2)*sin(alpha)*1j
    h2[2,0] =  cos(alpha)*(sqrt(3)*t1 - t2)
    h2[2,1] =  -sin(alpha)*((sqrt(3)/2)*t11 - 2*t12 - (sqrt(3)/2)*t22)*1j
    
    
    
    #############Ribbon Hamiltonian without soc(H0)##################################################
    a2= ones(N-1)
    I = eye(N)
    
    Hdiag = kron(I,h1)
    Hnondiag =  kron(diag(a2, -1),h2)
    
    
    H0 = Hdiag + Hnondiag + Hnondiag.T.conj()
    
    ###spin-orbit-part##############################
    
    ###########Pauli matix############
        
    sx = array([[0,1],[1,0]])
    sy = array([[0,-1j],[1j,0]])
    sz = array([[1,0],[0,-1]])
    ##################################
    I2 = array([[1,0],[0,1]])
    Lz = 0.5*lam*array([[0,0,0],[0,0,2j],[0,-2j,0]])
    sz_N = kron(sz,I)
    Ham_soc = kron(sz_N,Lz)
    #H_mag = kron(0.5*g*mub*Hx*sx,eye(3*N))
    #H_mag = kron(0.5*g*mub*Hy*sy,eye(3*N))
    #H_mag = kron(0.5*g*mub*Hz*sz,eye(3*N))
    
    return H0
  # Calculate the zigzag edge band structure for MoS2 monolayer
  a = 3.19
#b = sqrt(3)*a
N = 8
nk = 101

kx0 = linspace(-pi/a,pi/a,nk)
#ky0 = linspace(-pi/(sqrt(3)*a),pi/(sqrt(3)*a),nk)
ssp = zeros([3*N,nk])
eig = []
for i in arange(nk):
    kx = kx0[i]
    Hamil = ham_zigzag(kx)
    evals, evecs = LA.eigh(Hamil)
    evals = sorted(evals)
    ssp[:,i] = evals

# Plot the band structure for edge width N = 8
fig, ax = plt.subplots()
k_node = array([X0,X1,X2])
for n in range (len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color='k')
# Y-axis label
ax.set_ylabel("$Energy$ (eV)", fontsize=24)
ax.set_xlim(k_node[0],k_node[-1])

# X-axis values
ax.set_xticks([X0, X1, X2])

# X-axis label for the above values
ax.set_xticklabels(['$-\pi/a$', '$\Gamma$', '$\pi/a$'])
#ax.set_yticklabels(fontsize=20)    


for j in arange(3*N):    ## to plot multiple plot together 
    
    plt.plot(kx0,ssp[j,:],'r')



plt.ylim(-1.5,3.5)
ax.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.tight_layout()
plt.show()

#pdf
fig.tight_layout()
fig.savefig('MoS2_zigzag.eps', format='eps',dpi=200)
print('Done.\n')
