import matplotlib.pylab as plt
import numpy as np
import tables as tb
import logging
import progressbar
from numba import njit
import math
import pylandau



if __name__ == '__main__':

    def theta_sigma(beam_momentum, material_budget, charge_number=1):
        '''Calculates the scattering angle sigma for multiple scattering simulation. A Gaussian distribution
        is assumed with the sigma calculated here.
    
        Parameters
        ----------
        beam_momentum : number
            In MeV
        material_budget : number
            total distance / total radiation length
        charge number: int
            charge number of scattering particles, usually 1
        '''
    
        if material_budget == 0:
            return 0
        return 13.6 / beam_momentum * charge_number * np.sqrt(material_budget) * (1 + 0.038 * np.log(material_budget))
    
    momentum = np.linspace(100., 15000, 1000)
    material_budget = (250. + 450.) * 1e-4 / 9.370 + 0.002 + 40 /3.039e4 # all values given in cm. IBL flex estimated between 0.12 - 0.2 % X0 , 40 cm of air correspond to 0.13% 
    theta = theta_sigma(momentum, material_budget)
    
    point = (50/np.sqrt(12)/np.sqrt(2)*np.sqrt(1+4*15./27)**2)
    res = np.sqrt((theta * 27e4 * 15. / 27)**2 + point**2)
    
    point2 = (50/np.sqrt(12)/np.sqrt(2)*np.sqrt(1+4*5./40)**2)
    res2 = np.sqrt((theta * 27e4 * 5. / 40)**2 + point2**2)
    
    point3 = (50/np.sqrt(12)/np.sqrt(2)*np.sqrt(1+4*2./40)**2)
    res3 = np.sqrt((theta * 27e4 * 2. / 40)**2 + point3**2)
    
    plt.plot(momentum / 1000., res, label= 'x=15, L=27')
    plt.plot(momentum / 1000., res2, label='x = 5, L=40')
    plt.plot(momentum / 1000., res3, label='x = 2, L=40')
    plt.xlabel('Momentum [GeV]')
    plt.grid()
    plt.title('Vertex resolution (mult. scattering + pointing) vs. momentum')
    plt.legend()
    plt.ylim(0,400)
    plt.ylabel('Vertex resolution [um]')

    print theta_sigma(beam_momentum=5000., material_budget=material_budget)
    print material_budget
    
    plt.show()