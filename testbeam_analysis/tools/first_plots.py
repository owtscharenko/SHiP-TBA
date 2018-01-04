import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import scipy
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
import math
import ROOT

# input_filename = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/output_only_telescope/Efficiency.h5'
# input_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan_interpreted'



# with tb.open_file(input_filename + '.h5', 'r+') as in_file_h5:
#     for i in xrange(0,3):
#         data = in_file_h5.get_node(in_file_h5.root, 'DUT_%d' % i)
#         # Plot and fit result
#         plt.clf()
#         plt.plot(x, y, '.-', label='data')
#         plt.title('Histogramm')
#         plt.ylabel('Current [uA]')
#         plt.xlabel('Voltage [V]')
#         plt.grid(True)
#         plt.legend(loc=0)
#         plt.savefig(output_pdf)
#         
# print output_pdf

def cauchy(s,x,t):
    return 1/np.pi * s/(s**2 + (x-t)**2)
 
# def gauss(x, y , amplitude, mu, sigma,a,b,c, m,w):
#     gauss = y + amplitude * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))
#     line = m * x+b
#     poly = a*x + b*x**2 + c*x**3 + m
#     box = np.where(abs(x)<=w,1,0) 
#     return gauss + poly + box #+ line

def gauss(x,y,amplitude, mu, sigma,a,b,m ):
    gauss = y + amplitude * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))
    
    poly = a*x + b*x**2 + m
#     box = np.where(abs(x)<=w,1,0)
    return gauss #+ poly


def box_function(x,w):
    return np.where(abs(x)<=w,1,0)

'''
fitting gaussian to row occupancy
'''
def gauss_fit_row(input_file):
    output_pdf = input_file[:-16] + 'fit_rows.pdf'
    with tb.open_file(input_file + '.h5', 'r+') as in_file_h5:
        data1 = np.sum(in_file_h5.root.HistOcc[:], axis=1)
        data1 = np.concatenate(data1)

        y, amplitude, mu, sigma, = 1000, np.amax(data1), 145., 30.
#         m, x2, b = 0.0001, 1, 1000
        a, b, c, d, m = 2000.0, -100.0, 5.0, 0.5, 10000
        w = 30000
        rows = np.linspace(1,337,336,endpoint=False)
        xticks = np.linspace(0, 17.5, 8, endpoint=True)
        
        xtick_data = [1,50,100,150,200,250,300,350]

#         coeff, _ = curve_fit(f=gauss, xdata=rows, ydata=data1,p0=(y, amplitude, mu, sigma,a,b,c,m,w)) #
        coeff, _ = curve_fit(f=gauss, xdata=rows, ydata=data1,p0=(y, amplitude, mu, sigma,a,b,m))#,a,b,c,m,w))
#         print coeff[1]*coeff[3]*np.sqrt(2*np.pi)/np.sum(data1)
#         coeff, _ = curve_fit(f=cauchy, xdata=rows, ydata=data1,p0=(amplitude, mu))
#         print coeff
        plt.clf()
        plt.title('Occupancy in y on second plane')#('Occupancy per row on second plane')
        plt.grid()
        plt.xlabel('mm')#('row')
        plt.ylabel('#')
        plt.bar(range(1,337),data1,label='entries per row')
        plt.plot(rows,gauss(rows, *coeff), label = '$y + A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c$', color = 'crimson',linewidth = 2.5, alpha = 0.8)
        plt.xticks(xtick_data, xticks)
        plt.legend()
        ax = plt.axes()
        box = AnchoredText('mu = %.1f mm \nsigma = %.2f mm \nA = %.f ' %(coeff[2]*0.05,coeff[3]*0.05,coeff[1]), loc=2) # \na = %f \nb = %f ,coeff[4],coeff[5]
        ax.add_artist(box)
        plt.savefig(output_pdf)
#         print output_pdf
    return coeff[1]*coeff[3]*np.sqrt(2*np.pi)/np.sum(data1) # np.sqrt(coeff[1]/np.amax(data1))

'''
now for columns
'''
def gauss_fit_column(input_file):
    output_pdf2 = input_file[:-16] + 'fit_columns.pdf' 
    with tb.open_file(input_file + '.h5', 'r+') as in_file_h5:
        data = np.sum(in_file_h5.root.HistOcc[:], axis=0)
        data1 = np.sum(in_file_h5.root.HistOcc[:], axis=0)
        data1 = np.concatenate(data1)

        y, amplitude, mu, sigma = 100, np.amax(data1), 33., 30.
        a, b, c, m = 0.1, 0.001, 2, 0.001 # a*x + b*x**2 + m
        
        xticks = np.linspace(0, 20, 9, endpoint = True)
        xtick_data = [1,10,20,30,40,50,60,70,80]
        columns = np.linspace(1,81,80,endpoint=False)

        coeff, _ = curve_fit(f=gauss, xdata=columns, ydata=data1 ,p0=(y, amplitude, mu, sigma,a,b,m), bounds = (0.001,1000000)) #
#         print coeff[-4:]
#         print coeff[1],coeff[2],coeff[3]
#         print np.amax(data1)
        plt.clf()
        plt.title('Occupancy in x on second plane') #('Occupancy per column on second plane')
        plt.grid()
        plt.xlabel('mm')#('row')
        plt.ylabel('#')
        plt.bar(range(1,81),data1,label='entries per column')
        plt.plot(columns,gauss(columns, *coeff), label = 'fit', color = 'crimson',linewidth = 2.5, alpha = 0.8)
        plt.xticks(xtick_data, xticks)
        plt.legend()
        ax = plt.axes()
        box = AnchoredText('mu = %.1f mm \nsigma = %.2f mm \nA = %.f' %(coeff[2]*0.25,coeff[3]*0.25,coeff[1]), loc=2)
        ax.add_artist(box)
        plt.savefig(output_pdf2)
#         print output_pdf2    
    return coeff[1]*coeff[3]*np.sqrt(2*np.pi)/np.sum(data1)


if __name__ == "__main__":


#     files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan_interpreted',
#             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/85_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/86_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/96_module_2_ext_trigger_scan_interpreted']
    
    files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/69_module_0_ext_trigger_scan_interpreted',
             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_1/69_module_1_ext_trigger_scan_interpreted',
        
             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/69_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/70_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/71_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/72_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/85_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/86_module_2_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/96_module_2_ext_trigger_scan_interpreted',
             
             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/69_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/70_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/71_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/72_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/75_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/85_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/86_module_4_ext_trigger_scan_interpreted',
#              '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_4/96_module_4_ext_trigger_scan_interpreted',
             
             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_11/set_trigger_delay/module_0/54_module_0_ext_trigger_scan_interpreted',
             '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_11/set_trigger_delay/module_2/54_module_2_ext_trigger_scan_interpreted'
             ]
    
    
    row_coeffs = []
    column_coeffs = []

    for i in files:
        row_coeffs.append(gauss_fit_row(i))
        column_coeffs.append(gauss_fit_column(i))
        print row_coeffs[-1] # percentage of entries within gauss
#         print column_coeffs[-1]
    raise
    x = [0,0.3,1,2]
    x2 = [0,0.1,0.2]
    print row_coeffs[:4]
    print row_coeffs[8:11]

    plt.clf()
    plt.title('Beam profile vs. target thickness')
    plt.grid()
    plt.xticks(x)
    plt.xlabel('target thickness in $\lambda$')
    plt.ylabel('fraction of hits in gaussian distr.')
    plt.plot(x,row_coeffs[4:8],marker = 'o',linestyle = '-', label = 'hits on first plane within $2\sigma$')
    plt.plot(x,row_coeffs[12:16],marker = 'o',linestyle = '-', color = 'crimson', label = 'hits on second plane within $2\sigma$' )
    plt.legend()
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/beam_vs_target.pdf')

    plt.clf()
    plt.title('Beam profile vs. target thickness (emulsion runs)')
    plt.grid()
    plt.xticks(x2)
    plt.xlabel('target thickness in $\lambda$')
    plt.ylabel('fraction of hits in gaussian distr.')
    plt.plot(x2,row_coeffs[:3],marker = 'o',linestyle = '-', color = 'C0',label = 'hits on first plane within $2\sigma$')
    plt.plot(0.1,row_coeffs[3],marker = 'o',linestyle = None, color = 'C0')
    plt.plot(x2,row_coeffs[8:11],marker = 'o',linestyle = '-', color = 'crimson', label = 'hits on second plane within $2\sigma$' )
    plt.plot(0.1,row_coeffs[11],marker = 'o',linestyle = None, color = 'crimson' )
    plt.legend()
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/beam_vs_target_emulsion.pdf')



