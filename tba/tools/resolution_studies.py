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
    
    
    def pointing_res(pitch, planes, lever_arm, pointing_distance):
        ''' calculates pointing resolution for given parameters
        
        Parameters
        ----------
        pitch: number in um, distance of single pixel centers
        planes: number of planes with a hit
        lever arm: number in um
            distance between first and last plane
        pointing_distance: number in um
            distance between first plane and plane to be pointed to by track.
        '''
            
        return pitch/np.sqrt(12)/np.sqrt(planes)*np.sqrt(1+12.*(float((planes -1))/(planes+1))*(float(pointing_distance)/lever_arm)**2)
    
    def vertex_res(theta, pointing_res,lever_arm, pointing_distance):
        ''' calculates vertex resolution from pointing res and multiple scattering angle
        
        Parameters
        ----------
        theta: angle in rad, deviation from originating track by multiple scattering
        pointing res: number in um, accuracy of telescope
        lever arm: number in um
            distance between first and last plane
        pointing_distance: number in um
            distance between first plane and plane to be pointed to by track.
        '''
        return np.sqrt((theta * lever_arm * float(pointing_distance) / lever_arm)**2 + pointing_res**2)
    
    momentum = np.linspace(100., 15000, 1000)
    material_budget17 = (250. + 450.) * 1e-4 / 9.370 + 0.002 + 27. /3.039e4 + 100e-4/1.436 # all values given in cm. IBL flex estimated between 0.12 - 0.2 % X0 , 40 cm of air correspond to 0.13% 
    theta17 = theta_sigma(momentum, material_budget17)
    
    material_budget18 = (250. + 450.) * 1e-4 / 9.370 + 0.002 + 40 /3.039e4
    theta18 = theta_sigma(momentum, material_budget18)
    
    point_res17 = pointing_res(pitch=50., planes=2, lever_arm=27e4, pointing_distance = 15e4 )
    
    point_res18 = pointing_res(pitch=50., planes=3, lever_arm=40e4, pointing_distance = 5e4 )
    
    point_res18_2 = pointing_res(pitch=50., planes=3, lever_arm=40e4, pointing_distance = 2e4 )
#     point = (50/np.sqrt(12)/np.sqrt(2)*np.sqrt(1+4*15./27)**2)
#     res = np.sqrt((theta * 27e4 * 15. / 27)**2 + point**2)
#     
#     point2 = (50/np.sqrt(12)/np.sqrt(2)*np.sqrt(1+4*5./40)**2)
#     res2 = np.sqrt((theta * 27e4 * 5. / 40)**2 + point2**2)
#     
#     point3 = 50/np.sqrt(12)/np.sqrt(3)*np.sqrt(1+6.*(2./40)**2)
#     res3 = np.sqrt((theta18 * 40e4 * 2. / 40)**2 + point3**2)
    
    plt.plot(momentum / 1000., vertex_res(theta=theta17, pointing_res=point_res17,lever_arm=27e4, pointing_distance= 15e4), label= 'r = 15 cm, L = 27 cm')
    plt.plot(momentum / 1000., vertex_res(theta=theta18, pointing_res=point_res18,lever_arm=40e4, pointing_distance= 5e4), label='r = 5 cm, L = 40 cm')
    plt.plot(momentum / 1000., vertex_res(theta=theta18, pointing_res=point_res18_2,lever_arm=40e4, pointing_distance= 2e4), label='r = 2 cm, L = 40 cm')
    
#     plt.plot(momentum / 1000., res3 )
    
    plt.xlabel('Momentum [GeV]')
    plt.grid()
    plt.title('Vertex resolution (mult. scattering + pointing) vs. momentum')
    plt.legend()
    plt.ylim(0,400)
    plt.ylabel('Vertex resolution [um]')

#     print theta_sigma(beam_momentum=5000., material_budget=material_budget18)
#     print material_budget17

    plt.savefig('vertex-res.pdf')
    plt.show()