# Copyright 2017 J. Keaveney

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#	 http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solve the dielectric tensor for the roots of the complex refractive index by setting determinant to 0

Use analytic solutions for the 'easy' geometries - Faraday (B-field aligned with wavevector) and Voigt (B-field orthogonal to k-vector)
Use sympy to calculate solutions for all other non-trivial geometries.
Since the solutions for the non-trivial geometries depend on the susceptibility, array operations don't completely work, so it's *much* slower to calculate

Last updated 2018-02-19 JK
"""
# py 2.7 compatibility
from __future__ import (division, print_function, absolute_import)


from sympy import Symbol, cos, sin, pi, simplify, eye, powsimp, powdenest, lambdify, solve, solveset
#from sympy.solvers.solveset import linsolve as solve
from sympy.matrices import det, Matrix
from sympy import Array

import numpy as np

import scipy.linalg as la
from scipy.linalg import qr
import scipy

import time

from .FundamentalConstants import e0

def square_root(alpha, z):
	"Square root with different branch cut defined by alpha parameter."
	##FROM https://flothesof.github.io/branch-cuts-with-square-roots.html
	argument = np.angle(z) # between -pi and +pi
	modulus = np.abs(z)
	argument = np.mod(argument + alpha, 2 * np.pi) - alpha
	return np.sqrt(modulus) * np.exp(1j * argument / 2)

def non_standard_n1_n2_RotMat(chiL, chiR, chiZ, THETA):
	'''
	Wave equation solution method for an arbitrary, non-standard magnetic field angle THETA.
	This function is called by the solve_diel function if the system is not in the Voigt or Faraday geometry.

	Returns the rotation matrix to transform the coordinate system into the normal mode basis,
	and returns the two refractive index arrays.

	Note: This function is not valid for the systems in the Voigt or Faraday
	geometries due to the cotan(THETA) and cosec(THETA) terms.

	Inputs:

		chiL, chiR, chiZ	:	1D numpy arrays, of length N, that are the frequency-dependent electric susceptibilities
		THETA				:	Float, Magnetic field angle in radians

	Outputs:
		RotMat	:	Rotation matrix to transform coordinate system, dimensions (3, 3, N)
		n1		:	First solution for refractive index, dimensions (N)
		n2		:	Second solution for refractive index, dimensions (N)
	'''


	# Storing values of trigonometric functions for future use
	C = np.cos(THETA)		#cos
	S = np.sin(THETA)		#sin
	cT = 1/np.tan(THETA) 	#cotan
	cS = 1/np.sin(THETA)	#cosec

	# Relating the elements of the dielectric tensor to the electric susceptibility
	e_x = 0.5*(2.+chiL+chiR)
	e_xy = 0.5j*(chiR-chiL)
	e_z = 1.0+chiZ


	##METHOD USING DEFAULT NUMPY BRANCH CUT

	# Calculating the elements of the complex refractive index
	# (0j in elemB to ensure numpy succesfully computes the square root for otherwise non-complex values)
	elemA = 2*e_x*e_z+(e_x**2-e_x*e_z+e_xy**2)*S**2
	elemB = np.sqrt((e_x**2-e_x*e_z+e_xy**2)**2*S**4-4*e_xy**2*e_z**2*C**2 +0j)
	elemC = 2*(e_x*S**2+e_z*C**2)

	# Using the elements to calculate the solutions for the refractive index and their squares
	n1_sq = (elemA - elemB)/elemC
	n2_sq = (elemA + elemB)/elemC
	n1 = np.sqrt(n1_sq)
	n2 = np.sqrt(n2_sq)


	##METHOD USING A CUSTOMISED BRANCH CUT USING square_root FUNCTION (Commented out by default)

	# alpha defines the branch cut location and is arbitrary

	'''alpha = 3*np.pi/2

	elemA = 2*e_x*e_z+(e_x**2-e_x*e_z+e_xy**2)*S**2
	elemB = square_root(alpha, (e_x**2-e_x*e_z+e_xy**2)**2*S**4-4*e_xy**2*e_z**2*C**2 +0j)
	elemC = 2*(e_x*S**2+e_z*C**2)

	n1_sq = (elemA - elemB) / elemC
	n2_sq = (elemA + elemB) / elemC
	n1 = square_root(alpha, n1_sq)
	n2 = square_root(alpha, n2_sq)'''

	## Calculating the elements of the eigenvectors for n1 and n2 according to the analytical solution

	# n1
	n1e1 = (2*e_z*cT*(e_x+e_z*cT**2))/(-e_x**2-e_xy**2+3*e_x*e_z+
			2*e_z**2*cT**2 + (elemB - 2*e_x*e_z)*cS**2)

	n1e2 = (2*e_xy*(e_x + e_z*cT**2)*(e_x**2+e_xy**2-3*e_x*e_z - 2*e_z*(e_x + e_z)*cT**2 -
			2*e_z**2*cT**4 - (elemB -2*e_x*e_z)*cS**2)*S)/((e_x**2 + e_xy**2 - 3*e_x*e_z -
			2*e_z**2*cT**2 - (elemB - 2*e_x*e_z)*cS**2) * (e_x**2 - e_xy**2 +
			e_x*e_z + 2*e_x*e_z*cT**2 + (elemB - 2*e_x*e_z)*cS**2))

	# n2
	n2e1 = -((2*e_z*cT*(e_x+e_z*cT**2))/(e_x**2 + e_xy**2 - 3*e_x*e_z -
				2*e_z**2*cT**2 + (elemB + 2*e_x*e_z)*cS**2))

	n2e2 = (2*e_xy*(e_x+e_z*cT**2)*S**3*(elemB+2*e_x*e_z +
			e_z*(-e_x - 2*e_z + e_x*np.cos(2*THETA))*cT**2 + (e_x**2 + e_xy**2 -
			3*e_x*e_z)*S**2))/((elemB+2*e_x*e_z -
			2*e_z**2*C**2 + (e_x**2 + e_xy**2 -
			3*e_x*e_z)*S**2)*(-elemB - 2*e_x*e_z +
			2*e_x*e_z*C**2 + (e_x**2 - e_xy**2 + e_x*e_z)*S**2))

	# Constructing the Rotation Matrix
	ones = np.ones(len(e_x))
	zeros = np.zeros(len(e_x))

	# Populate the rotation matrix

	RotMat = np.array([[n1e1,n2e1,zeros],
					   [n1e2,n2e2,zeros],
					   [ones,ones,ones]]).T

	return n1, n2, RotMat







def solve_diel(chiL, chiR, chiZ, THETA, Bfield, verbose=False,force_numeric=False, use_old_method=False):
	'''
	Solves the wave equation to find the two propagating normal modes of the system,
	for a given magnetic field angle THETA. For the general case, use symbolic python to
	solve for the roots of n-squared.
	(Escapes this slow approach for the two analytic cases for the Voigt and Faraday geometries)

	Returns the rotation matrix to transform the coordinate system into the normal mode basis,
	and returns the two refractive index arrays.

	Inputs:

		chiL, chiR, chiZ	:	1D lists or numpy arrays, of length N, that are the frequency-dependent electric susceptibilities
		THETA				:	Float, Magnetic field angle in radians
		Bfield				:	Float, Magnitude of applied magnetic field (skips slow approach if magnetic field is very close to zero)

	Options:

		verbose			:	Boolean to output more print statements (timing reports mostly)
		force_numeric	:	If True, forces all angles to go through the numeric approach, rather than escaping for the analytic cases (THETA=0, THETA=pi/2...)
		use_old_method	:	If True, forces use of numerical method rather than analytical method.


	Outputs:
		RotMat	:	Rotation matrix to transform coordinate system, dimensions (3, 3, N)
		n1		:	First solution for refractive index, dimensions (N)
		n2		:	Second solution for refractive index, dimensions (N)

	'''

	approx_threshold = 1e-4  	##Value used do determine if THETA is close enough
								##to Voigt or Faraday geometries to use an approximation



	if verbose:
		print(('B-field angle (rad, pi rad): ',THETA, THETA/np.pi))

	stt = time.clock()

	# make chiL,R,Z arrays if not already
	chiL = np.array(chiL)
	chiR = np.array(chiR)
	chiZ = np.array(chiZ)

	#verbose=True

	#### Escape the slow loop for analytic (Faraday and Voigt) cases
	## For these analytic cases we can use array operations and it is therefore
	## much faster to compute
	if (abs(THETA%(2*np.pi) - np.pi/2) < approx_threshold) or (abs(THETA%(2*np.pi) - 3*np.pi/2) < approx_threshold) and (not force_numeric):
		# ANALYTIC SOLNS FOR VOIGT
		if verbose: print('Voigt - analytic')

		# solutions for elements of the dielectric tensor:
		ex = 0.5 * (2. + chiL + chiR)
		exy = 0.5j * (chiR - chiL)
		ez = 1.0 + chiZ

		# refractive indices to propagate
		n1 = np.sqrt(ex + exy**2/ex)
		n2 = np.sqrt(ez)

		#ev1 = [np.zeros(len(ex)),ex/exy,np.ones(len(ex))]
		#ev2 = [np.ones(len(ex)),np.zeros(len(ex)),np.zeros(len(ex))]
		#ev3 = [np.zeros(len(ex)),np.zeros(len(ex)),np.ones(len(ex))]

		#RotMat = np.array([ev1,ev2,ev3])

		ones = np.ones(len(ex))
		zeros = np.zeros(len(ex))


		# *Changed output to new array format

		RotMat = np.array([[zeros, ones, zeros],
						   [ex/exy, zeros, zeros],
						   [ones, zeros, ones]]).T

		if verbose:
			print('Shortcut:')
			print((RotMat.shape))
			print((n1.shape))
			print((n2.shape))

	elif ((abs(THETA) < approx_threshold) or ((abs(THETA - np.pi)) < approx_threshold) or abs(Bfield)<1e-2)  and (not force_numeric): ## Use Faraday geometry if Bfield is very close to zero
		# ANALYTIC SOLNS FOR FARADAY
		#if verbose:
		if verbose: print('Faraday - analytic TT')

		ex = 0.5 * (2. + chiL + chiR)
		exy = 0.5j * (chiR - chiL)
		e_z = 1.0 + chiZ

		n1 = np.sqrt(ex + 1.j*exy)
		n2 = np.sqrt(ex - 1.j*exy)

		#ev1 = np.array([-1.j*np.ones(len(ex)),np.ones(len(ex)),np.zeros(len(ex))])
		#ev2 = np.array([1.j*np.ones(len(ex)),np.ones(len(ex)),np.zeros(len(ex))])
		#ev3 = [np.zeros(len(ex)),np.zeros(len(ex)),np.ones(len(ex))]

		ones = np.ones(len(ex))
		zeros = np.zeros(len(ex))

		# *Changed output to new array format

		if (abs(THETA) < approx_threshold):
			#RotMat = np.array([ev1,ev2,ev3])

			RotMat = np.array([[-1.j*ones, 1.j*ones, zeros],
							   [ones, ones, zeros],
							   [zeros, zeros, ones]]).T

		else:
			#if anti-aligned, swap the two eigenvectors
			#RotMat = np.array([ev2,ev1,ev3])

			RotMat = np.array([[1.j*ones, -1.j*ones, zeros],
							   [ones, ones, zeros],
							   [zeros, zeros, ones]]).T

		if verbose:
			print('Shortcut:')
			print((RotMat.shape))
			print((n1.shape))
			print((n2.shape))





	elif use_old_method:
		print("using old method")
		if verbose: print('Non-analytic angle.. This will take a while...')	##### THIS IS THE ONE THAT's WRONG....
		# set up sympy symbols
		theta = Symbol('theta',real=True)
		n_sq = Symbol('n_sq')
		e_x = Symbol('e_x')
		e_xy = Symbol('e_xy')
		e_z = Symbol('e_z')

		# General form of the dielectric tensor
		DielMat = Matrix (( 	[(e_x - n_sq)*cos(theta), e_xy, e_x*sin(theta)],
									[-e_xy * cos(theta), e_x - n_sq, -e_xy*sin(theta)],
									[(n_sq - e_z)*sin(theta), 0, e_z*cos(theta)] 			))

		et1 = time.clock() - stt

		# Substitute in angle
		DielMat_sub = DielMat.subs(theta, pi*THETA/np.pi)

		et2 = time.clock() - stt

		# Find solutions for complex indices for a given angle
		solns = solve(det(DielMat_sub), n_sq)

		et3a = time.clock() - stt
		#print et3a

		# Find first refractive index
		DielMat_sub1 = DielMat_sub.subs(n_sq, solns[0])
		n1 = np.zeros(len(chiL),dtype='complex')
		n1old = np.zeros(len(chiL),dtype='complex')
		# Find second refractive index
		DielMat_sub2 = DielMat_sub.subs(n_sq, solns[1])
		n2 = np.zeros(len(chiL),dtype='complex')
		n2old = np.zeros(len(chiL),dtype='complex')

		et3b = time.clock() - stt

		Dsub1 = lambdify((e_x,e_xy,e_z), DielMat_sub1, 'numpy')
		Dsub2 = lambdify((e_x,e_xy,e_z), DielMat_sub2, 'numpy')

		nsub1 = lambdify((e_x,e_xy,e_z), solns[0], 'numpy')
		nsub2 = lambdify((e_x,e_xy,e_z), solns[1], 'numpy')

		# Initialise rotation matrix
		RotMat = np.zeros((3,3,len(chiL)),dtype='complex')

		et3c = time.clock() - stt

		# populate refractive index arrays
		n1 = np.sqrt(nsub1(0.5*(2.+chiL+chiR), 0.5j*(chiR-chiL), (1.0+chiZ)))
		n2 = np.sqrt(nsub2(0.5*(2.+chiL+chiR), 0.5j*(chiR-chiL), (1.0+chiZ)))

		et3 = time.clock() - stt

		if verbose:
			print(('setup time:', et1, et1))
			print(('solve nsq: (total/solve/sub in) ', et3a, et3a-et2, et2-et1))
			print(('get nsq arrays (tot time / populate ref. index / gen. lambdify / sub in): ', et3, et3-et3c, et3c-et3b, et3b-et3a))

		# loop over all elements of chiL,R,Z to populate eigenvectors
		# time-limiting step for arrays of length >~ 5000
		for i, (cL, cR, cZ) in enumerate(zip(chiL,chiR,chiZ)):
			#if verbose: print 'Detuning point i: ',i

			#time diagnostics
			st = time.clock()


			'''	
		## OLD and slow method::
			# Sub in values of susceptibility
			DielMat_sub1a = DielMat_sub1.subs(e_x, 0.5*(2.+cL+cR))
			DielMat_sub1a = DielMat_sub1a.subs(e_xy, 0.5j*(cR-cL))
			DielMat_sub1a = DielMat_sub1a.subs(e_z, (1.0+cZ))
			
			et1 = time.clock() - st
			
			# Evaluate and convert to numpy array
			DM = np.array(DielMat_sub1a.evalf())
			DMa = np.zeros((3,3),dtype='complex')
			for ii in range(3):
				for jj in range(3):
					DMa[ii,jj] = np.complex128(DM[ii,jj])
			
			et2 = time.clock() - st
		
			# use scipy to find eigenvector
			#ev1 = Matrix(DMa).nullspace()
			#print 'Sympy: ', ev1
			
			ev1old = nullOld(DMa).T[0]
			#ev1 = null(DMaNP).T
			
			# sub in for ref. index
			n1soln = solns[0].subs(e_x, 0.5*(2.+cL+cR))
			n1soln = n1soln.subs(e_xy, 0.5j*(cR-cL))
			n1soln = n1soln.subs(e_z, (1.0+cZ))
			
			# Populate the refractive index array
			n1old[i] = np.sqrt(np.complex128(n1soln.evalf()))
		## /OLD method
			'''

			# NEW method

			# Sub in values of susceptibility
			DMaNP = Dsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ))
			#print DMa
			ev1 = null(DMaNP).T
			# Populate the refractive index array
			#n1[i] = np.sqrt(nsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ)))


			'''
			## METHOD COMPARISON
			
			print 'SymPy:'
			print DMa
			print DMa.shape, type(DMa)
			print 'Numpy'
			print DMaNP
			print DMaNP.shape, type(DMaNP)
			
			print 'Eigenvectors ...'
			print 'Old: ', ev1old			
			print 'New: ',ev1
			'''

			#print '\n\n\n'

			#print 'scipy: ', ev1

			et3 = time.clock() - st

			et4 = time.clock() - st

			#
			## Now repeat the above for second eigenvector
			#

		## NEW
			# Sub in values of susceptibility
			DMaNP = Dsub2(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ))
			# Find null eigenvector
			ev2 = null(DMaNP).T
			# Populate the refractive index array
			#n2[i] = np.sqrt(nsub2(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ)))

			et5 = time.clock() - st

			'''
		## OLD
			# Evaluate and convert to numpy array
			DielMat_sub2a = DielMat_sub2.subs(e_x, 0.5*(2.+cL+cR))
			DielMat_sub2a = DielMat_sub2a.subs(e_xy, 0.5j*(cR-cL))
			DielMat_sub2a = DielMat_sub2a.subs(e_z, (1.0+cZ))
			
			DM = np.array(DielMat_sub2a.evalf())
			DMa = np.zeros((3,3),dtype='complex')
			for ii in range(3):
				for jj in range(3):
					DMa[ii,jj] = np.complex128(DM[ii,jj])
					
			et6 = time.clock() - st
			
			# use scipy to find eigenvector
			ev2old = nullOld(DMa).T[0]
			
			et7 = time.clock() - st
			
			# sub in for ref. index
			n2soln = solns[1].subs(e_x, 0.5*(2.+cL+cR))
			n2soln = n2soln.subs(e_xy, 0.5j*(cR-cL))
			n2soln = n2soln.subs(e_z, (1.0+cZ))
			
			# Populate the refractive index array
			n2old[i] = np.sqrt(np.complex128(n2soln.evalf()))
			'''


			# Populate the rotation matrix
			RotMat[:,:,i] = [ev1, ev2, [0,0,1]]



		et_tot = time.clock() - stt
		if verbose:
			print(('Time elapsed (non-analytic angle):', et_tot))



	else:
		if verbose: print("Using analytical method")

		#Uses analytical method
		n1, n2, RotMat = non_standard_n1_n2_RotMat(chiL, chiR, chiZ, THETA)


	#print("RotMat =", RotMat)
	#print("n1 = ", n1)
	#print("n2 = ", n2, '\n')



	if verbose: print('SD done')
	return RotMat, n1, n2


def null(A,tol=1e-6):
	ee, ev = la.eig(A)

	#for E,V in zip(ee,ev.T):
	#	print 'Eigs:',abs(E), '\t', E#, '\t', V
	#print '\n'

	z = list(zip(ee,ev.T))
	zs = sorted(z, key=lambda f: abs(f[0])) # sort by absolute value of eigenvectors
	ees, evs = list(zip(*zs))

	#for E,V in zip(ee,ev):
	#	print abs(E), '\t', E, '::', V

	if abs(ees[0]<tol):
		return evs[0].T
	else:
		print('No null eigenvector found! List of eigenvalules:')
		for E,V in zip(ee,ev.T):
			print(('Eigs:',abs(E), '\t', E, '\n\t', V))
		print('\n')
		return 0

def test_null():
	A = np.matrix([[2,3,5],[-4,2,3],[0,0,0]])
	SymA = Matrix(A)

	nv = null(A)
	nvold = nullOld(A)

	print((nv.T))
	print((nvold.T[0]))
	print((SymA.nullspace()[0].evalf()))

	print((A * nv))

def test_solveset():
	x = Symbol('x')
	A = Matrix([[x,2,x*x],[4,5,x],[x,8,9]])

	solns = solve(det(A), x)
	solns_set = list(solveset(det(A), x))

	print(solns)
	print('\n')
	print(solns_set)

	print('\n\n\n')
	print((solns[0]))
	print('\n')
	print((solns_set[0]))

	soln_sub = solns[0].subs(x, 1)
	solnset_sub = solns_set[0].subs(x, 1)

	s1 = soln_sub.evalf()
	s1set = solnset_sub.evalf()

	s2set = solns_set[1].subs(x, 1).evalf()

	print(s1)
	print(s1set)
	print(s2set)

def nullOld(A, eps=1e-14):
	""" Find the null eigenvector x of matrix A, such that Ax=0"""
	# Taken with gratitude from http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
	u, s, vh = la.svd(A)
	null_mask = (s <= eps)
	null_space = scipy.compress(null_mask, vh, axis=0)
	return scipy.transpose(null_space)


'''
def null(A, atol=1e-15, rtol=0):
	"""Compute an approximate basis for the nullspace of A.

	The algorithm used by this function is based on the singular value
	decomposition of `A`.

	Parameters
	----------
	A : ndarray
		A should be at most 2-D.  A 1-D array with length k will be treated
		as a 2-D with shape (1, k)
	atol : float
		The absolute tolerance for a zero singular value.  Singular values
		smaller than `atol` are considered to be zero.
	rtol : float
		The relative tolerance.  Singular values less than rtol*smax are
		considered to be zero, where smax is the largest singular value.

	If both `atol` and `rtol` are positive, the combined tolerance is the
	maximum of the two; that is::
		tol = max(atol, rtol * smax)
	Singular values smaller than `tol` are considered to be zero.

	Return value
	------------
	ns : ndarray
		If `A` is an array with shape (m, k), then `ns` will be an array
		with shape (k, n), where n is the estimated dimension of the
		nullspace of `A`.  The columns of `ns` are a basis for the
		nullspace; each element in numpy.dot(A, ns) will be approximately
		zero.
	"""

	A = np.atleast_2d(A)
	u, s, vh = la.svd(A)
	tol = max(atol, rtol * s[0])
	nnz = (s >= tol).sum()
	print nnz
	ns = vh[nnz:].conj().T
	return ns
'''

def main():
	""" General test method """
	from . import spectra as sp
	p_dict = {'Bfield':700,'rb85frac':1,'Btheta':88*np.pi/180,'Bphi':0*np.pi/180,'lcell':75e-3,'T':84,'Dline':'D2','Elem':'Cs'}
	chiL,chiR,chiZ = sp.calc_chi(np.linspace(-3500,3500,10),p_dict)

	#print 'ez: ',chiZ + 1 # ez / e0
	#print 'ex: ',0.5*(2+chiL+chiR) # ex / e0
	#print 'exy: ',0.5j*(chiR-chiL) # exy / e0

	RotMat, n1, n2 = solve_diel(chiL,chiR,chiZ,88*np.pi/180)
	print((RotMat.shape))

def calculation_time_analysis():
	""" Test method for looking at timing performance """
	from . import spectra as sp
	p_dict = {'Bfield':700,'rb85frac':1,'Btheta':88*np.pi/180,'Bphi':0*np.pi/180,'lcell':75e-3,'T':84,'Dline':'D2','Elem':'Cs'}
	chiL,chiR,chiZ = sp.calc_chi([-3500],p_dict)

	for angle in [0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]:
		print(('Angle (degrees): ',angle*180/np.pi))
		RotMat, n1, n2 = solve_diel(chiL,chiR,chiZ,angle)

def test_equivalence():
	""" Test numeric vs analytic solutions """

	from . import spectra as sp

	#analytic
	p_dict = {'Bfield':15000,'rb85frac':1,'Btheta':0*np.pi/180,'Bphi':0*np.pi/180,'lcell':1e-3,'T':84,'Dline':'D2','Elem':'Rb'}
	chiL1,chiR1,chiZ1 = sp.calc_chi([-18400],p_dict)
	RotMat1, n11, n21 = solve_diel(chiL1,chiR1,chiZ1,0,150,force_numeric=False)

	#numeric
	chiL2, chiR2, chiZ2 = chiL1, chiR1, chiZ1
	#chiL2,chiR2,chiZ2 = sp.calc_chi([-18400],p_dict)
	RotMat2, n12, n22 = solve_diel(chiL2,chiR2,chiZ2,0,150,force_numeric=True)

	print('RM 1')
	print(RotMat1)

	print('RM 2')
	print(RotMat2)

	print('n1_1 (analytic)')
	print(n11)
	print('n1_2')
	print(n12)
	print('n2_1 (analytic)')
	print(n21)
	print('n2_2')
	print(n22)

	print('chi1')
	print((chiL1, chiR1, chiZ1))

	print('chi2')
	print((chiL2, chiR2, chiZ2))

if __name__ == '__main__':
	test_equivalence()