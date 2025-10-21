from ase.data import atomic_masses, atomic_numbers
import ase.units as units

import json

from math import pi

import numpy as np

import time


class Cascade:

    def __init__(self, element='H', Z=None, mu=None, nmax=40):

        self.nmax = nmax
        self.element = element

        if Z is None:
            Z = atomic_numbers[element]

        self.Z = Z

        if mu is None:
            mu = self.get_reduced_mass()

        self.mu = mu


    def calculate_rates(self, gnuc_values=(100000, 2000, 2, 0.001), grefill=10, gweak=7.788e-5):

        self.grad_nlne, self.gradtot_nl = self.get_radiative_rate()
        self.gauger_nlne, self.gaugertot_nle = self.get_auger_rates()
        self.gnuc_nl = self.get_nuclear_rate(gnuc_values)
        self.grefill_e = [2*grefill, grefill, 0]
        self.gweak = gweak


    def get_auger_energy(self, n1, Ne=1):

        n2 = n1

        energy = -1

        while energy < 0:

            n2 -= 1

            if Ne == 1:
                Ze = self.Z - 1
                energy = self.Z**2 * self.mu * self.get_deltaE(n1, n2) - Ze**2 / 2 

            if Ne == 2:
                Ze = self.Z - 1 - 5/16
                energy = self.Z**2 * self.mu * self.get_deltaE(n1, n2) - (Ze**2 - (self.Z-1)**2 / 2) 


        return energy, n2


    def get_auger_rate(self, n, Ne):

        if Ne == 1:
            Ze = self.Z - 1
            energy, nf = self.get_auger_energy(n, Ne)
            y = Ze / np.sqrt(2 * energy)
            C = self.get_constant_C(n, Ze, y)

            num = 2.0**(4*n + 6) * n**(2*n + 2) * (n - 1)**(2*n + 4) * y**2 * np.exp(y*(4*np.arctan(y) - pi))
            den = (2*n - 1)**(4*n + 2) * (1 + y**2) * np.sinh(pi*y) * 2

            G = 1/3 * Ze**2/self.Z**2 * C**2 * pi/self.mu**2 * num / den / units._aut / 1e12

        if Ne == 2:
            Ze = self.Z - 1 - 5/16
            energy, nf = self.get_auger_energy(n, Ne)
            y = Ze / np.sqrt(2 * energy)
            C = self.get_constant_C(n, Ze, y)

            num = 2.0**(4*n + 6) * n**(2*n + 2) * (n - 1)**(2*n + 4) * y**2 * np.exp(y*(4*np.arctan(y) - pi))
            den = (2*n - 1)**(4*n + 2) * (1 + y**2) * np.sinh(pi*y)

            G = 1/3 * Ze**2/self.Z**2 * C**2 * pi/self.mu**2 * num / den / units._aut / 1e12

        return G, nf


    def get_auger_rates(self):

        nmax = self.nmax    
        Rif = self.get_radial_integrals()

        gauger_nlne = np.zeros((nmax,nmax+1,2,3))
        gaugertot_nle = np.zeros((nmax,nmax+1,3))

        for ni in range(2,nmax):
            for li in range(ni):
                Pauger, nf = self.get_auger_rate(ni, 1)
                gauger_nlne[ni,li,0,1] = (Rif[ni,li,nf] / Rif[ni,ni-1,ni-1])**2 * self.cg_factor(li,li-1) / self.cg_factor(ni-1,ni-2) * Pauger
                gauger_nlne[ni,li,1,1] = (Rif[nf,li+1,ni] / Rif[ni,ni-1,ni-1])**2 * self.cg_factor(li,li+1) / self.cg_factor(ni-1,ni-2) * Pauger
                gaugertot_nle[ni,li,1] = gauger_nlne[ni,li,:,1].sum()

            for li in range(ni):
                Pauger, nf = self.get_auger_rate(ni, 2)
                gauger_nlne[ni,li,0,2] = (Rif[ni,li,nf] / Rif[ni,ni-1,ni-1])**2 * self.cg_factor(li,li-1) / self.cg_factor(ni-1,ni-2) * Pauger
                gauger_nlne[ni,li,1,2] = (Rif[nf,li+1,ni] / Rif[ni,ni-1,ni-1])**2 * self.cg_factor(li,li+1) / self.cg_factor(ni-1,ni-2) * Pauger
                gaugertot_nle[ni,li,2] = gauger_nlne[ni,li,:,2].sum()

        return gauger_nlne, gaugertot_nle


    def cg_factor(self, l1, l2):
    
        cg2 = 0
        if l1 == l2 + 1:
            cg2 = l1 / (2*l1+1)
        if l1 == l2 - 1.:
            cg2 = l2 / (2*l1+1)

        return cg2


    def get_constant_C(self, n, Ze, y):

        C1 = 0
        C2 = 0

        num = (2*n + 1) * (2*n + 2) * n**2 * (n - 1)**2 * (1 + y**2)
        den = 3 * (2*n - 1)**2 * self.mu**2 * y**2 * np.exp(y * (4*np.arctan(y) - pi))

        C = 1 - Ze**2 / self.Z**2 * num / den * (C1/2 - C2/5)

        return C


    def get_deltaE(self, ni, nf):

        deltaE = 0.5 * abs(1.0 / ni**2 - 1.0 / nf**2)

        return deltaE


    def get_energies(self):

        energies_nm = np.zeros((self.nmax,self.nmax))

        for n in range(self.nmax):
            for m in range(self.ninit):
                energy = self.Z**2 * self.mu * self.get_deltaE(n+1, m+1) * units.Hartree
                energies_nm[n,m] = energy

        return energies_nm


    def get_nuclear_rate(self, gnuc_values):

        nmax = self.nmax

        gnuc_circ = np.zeros((nmax))
        gnuc_circ[:len(gnuc_values)] = gnuc_values

        gnuc_nl = np.zeros((nmax+1, nmax+1))

        for l in range(nmax-1):
            gnuc_nl[l+1,l] = gnuc_circ[l] / units.Hartree / units._aut / 1e12

            for n in range(l+1,nmax-0):
                gnuc_nl[n+1,l] = gnuc_nl[n,l] * (n / (n + 1))**(2*l + 4) * (n + l + 1) / (n-l)

        return gnuc_nl


    def get_radiative_rate(self):

        nmax = self.nmax    
        Rif = self.get_radial_integrals()

        grad_nlne = np.zeros((nmax,nmax+1,nmax,2))
        gradtot_nl = np.zeros((nmax,nmax))
    
        for ni in range(1,nmax):
            for li in range(ni):
                for nf in range(1,ni):
                    deltaE = self.get_deltaE(ni,nf)
                    grad_nlne[ni,li,nf,0] = self.Z**4 * self.mu * 4.0/3.0 * units.alpha**3 * li/(2.0*li+1.0) * Rif[ni,li,nf]**2 * deltaE**3 / units._aut / 1e12 
                    grad_nlne[ni,li,nf,1] = self.Z**4 * self.mu * 4.0/3.0 * units.alpha**3 * li/(2.0*li+1.0) * Rif[nf,li+1,ni]**2 * deltaE**3 / units._aut / 1e12
    
                gradtot_nl[ni,li] = grad_nlne[ni,li].sum(axis=-1).sum(axis=-1)
    
        return grad_nlne, gradtot_nl


    def get_radial_integrals(self):
    
        data_n = np.loadtxt('/home/smanti/CODES/cascade-model/cascade/Rif.dat')

        Rif = np.zeros((self.nmax+1,self.nmax+2,self.nmax+1))
    
        for data in data_n:
            ni,li,nf,lf,r = data
            Rif[int(ni),int(li),int(nf)] = r
    
        return Rif


    def get_reduced_mass(self, kaon_mass=493.677):

        m_electron = units._me * units._c**2 / (1e6 * units._e)
        m_kaon = kaon_mass / m_electron

        Dalton2MeV = 931.49410242
        m_nucleus = atomic_masses[self.Z] * Dalton2MeV / m_electron
        mu = m_nucleus * m_kaon / (m_nucleus + m_kaon)

        self.m_nucleus = m_nucleus

        return mu


    def run(self, max_events=1000, ninit=20, linit=None, output=False):

        self.ninit = ninit

        weakfraction = 0
        nabsorption = 0
        cascadetime = 0
        yieldstot_n = np.zeros((self.nmax))
        spectrumtot_nm = np.zeros((self.nmax,self.nmax))
        taudistribution_i = np.zeros((500))
        absorption_n = np.zeros((self.nmax))

        for count in range(max_events):
            self.run_cascade(ninit=ninit, linit=linit, output=output)
            yieldstot_n += self.yields_n / max_events
            spectrumtot_nm += self.spectrum_nm / max_events
            cascadetime += self.tau / max_events
            absorption_n += self.absorption_n / max_events
            if self.event == 'absorption':
                nabsorption += 1 / max_events
            if self.event == 'weak':
                weakfraction += 1 / max_events

        self.weakfraction = weakfraction
        self.nabsorption = nabsorption
        self.cascadetime = cascadetime
        self.yieldstot_n = yieldstot_n
        self.spectrumtot_nm = spectrumtot_nm
        self.absorption_n = absorption_n

        i = int(self.tau / self.dtau + 1)

        if i > 499:
            i = 499

        taudistribution_i[i] += 1 / max_events / self.dtau  
        self.taudistribution_i = taudistribution_i


    def run_cascade(self, ninit, linit, output=False):
    
        tau = 0
        n1 = ninit
        l1 = linit
        if l1 is None:
            r = np.random.random()
            l1 = int(n1 * np.sqrt(r))
        ne = 2
        nenew = ne
        N = 0
        nres = 20

        grad_nlne = self.grad_nlne
        gradtot_nl = self.gradtot_nl
        gauger_nlne = self.gauger_nlne
        gaugertot_nle = self.gaugertot_nle
        gnuc_nl = self.gnuc_nl
        grefill_e = self.grefill_e
        gweak = self.gweak

        yields_n = np.zeros((self.nmax))
        spectrum_nm = np.zeros((self.nmax,self.nmax))
        absorption_n = np.zeros((self.nmax))

        dtau = 1 / (gradtot_nl[n1,n1-1] + min(gaugertot_nle[n1,1,2],grefill_e[1]) + gnuc_nl[n1,n1-1] + gweak)
        self.dtau = dtau
    
        if output:
            print('begin cascade')
            print('-'*30)
        start = time.time()
        while True:
        
            if nres > 0:
                gref = grefill_e[ne]
            else:
                gref = 0
            
            gtot = gradtot_nl[n1,l1] + gaugertot_nle[n1,l1,ne] + gref + gnuc_nl[n1,l1] + gweak
          
            r1 = np.random.random()
            tau -= 1.0 / gtot * np.log(1.0 - r1)
        
            r2 = np.random.random()  
            
            if r2 < gradtot_nl[n1,l1] / gtot:
                
                event = 'radiative'
                r3 = np.random.random()
                g0 = r3 * gradtot_nl[n1,l1]
                
                gsum = 0
                nnew = 1
            
                for nnew in range(1,n1):
                    
                    gsum = gsum + grad_nlne[n1,l1,nnew,0]
                    
                    if gsum > g0:
                        n2 = nnew
                        l2 = l1 - 1
                        break
                    
                    gsum = gsum + grad_nlne[n1,l1,nnew,1]
                    
                    if gsum > g0:
                        n2 = nnew
                        l2 = l1 + 1
                        break
                
                for n in range(1, self.nmax+1):
                    if n1 == n and n2 == n - 1:
                        yields_n[n-1] += 1
                    for m in range(1, self.nmax+1):
                        if n1 == n and n2 == m:
                            spectrum_nm[n-1,m-1] += 1
 
            elif r2 < (gradtot_nl[n1,l1] + gaugertot_nle[n1,l1,ne]) / gtot:
                
                event = 'auger'
                nnew = self.get_auger_rate(n1, ne)[1]
                nenew = ne - 1
                
                r4 = np.random.random()
                g0 = r4 * gaugertot_nle[n1,l1,ne]
                
                gsum = 0
             
                while True:
                    gsum = gsum + gauger_nlne[n1,l1,0,ne]
                    
                    if gsum > g0:
                        n2 = nnew
                        l2 = l1 - 1
                        break
                        
                    gsum = gsum + gauger_nlne[n1,l1,1,ne]
                
                    if gsum > g0:
                        n2 = nnew
                        l2 = l1 + 1
                        break
                    
            elif r2 < (gradtot_nl[n1,l1] + gaugertot_nle[n1,l1,ne] + gref) / gtot:
                event = 'refill'
                
                n2 = n1
                l2 = l1
                
                nenew = ne + 1
                nres = nres - 1
    
            elif r2 < (gradtot_nl[n1,l1] + gaugertot_nle[n1,l1,ne] + gref + gnuc_nl[n1,l1]) / gtot:
                event = 'absorption'
                absorption_n[n1-1] += 1
                if output:
                    print(N,n1, l1, ne, event)
                    print('-'*30)
                    print('end cascade')
                break
            
            elif r2 < (gradtot_nl[n1,l1] + gaugertot_nle[n1,l1,ne] + gref + gnuc_nl[n1,l1] + gweak) / gtot:
                event = 'weak'
                if output:
                    print(N,n1, l1, ne, event)
                    print('-'*30)
                    print('end cascade')
                break
                
            elif n2 == 1:
                if output:
                    print('end cascade')
                break
                
            N += 1
            
            if output:
                print(N, n1, l1, ne, event)
                print('-'*30)
            
            n1 = n2
            l1 = l2
            ne = nenew
            
            if n2 == 1:
                break
        
        end = time.time()
        #print(f'Time {end-start:.2E} s')
        self.tau = tau
        self.event = event
        self.yields_n = yields_n
        self.spectrum_nm = spectrum_nm
        self.absorption_n = absorption_n
    

    def set_nuclear_rate(self, gnuc_values=(100000, 2000, 2, 0.001)):

        self.gnuc_nl = self.get_nuclear_rate(gnuc_values)


    def set_refill_rate(self, grefill=10):

        self.grefill_e = [2*grefill, grefill, 0]


    def save_results(self, filename='cascade'):

        data = {}

        data['cascadetime'] = self.cascadetime
        data['yields'] = self.yieldstot_n.tolist()
        data['spectrum'] = self.spectrumtot_nm.tolist()
        data['energies'] = self.get_energies().tolist()

        with open(f'results-{filename}.json', 'w') as outfile:
            json.dump(data, outfile, indent=3)
