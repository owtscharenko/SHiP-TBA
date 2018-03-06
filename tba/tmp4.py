import matplotlib.pyplot as plt
import os
import numpy as np
import tables as tb
import logging

from numba import njit,jit
from tqdm import tqdm

''' plotting FE-I4 software and hardware timestamps against each other
    overflowing hardware timestamps need to be fixed beforehand, this usually happens in the pybar_fei4_converter
'''

def open_hit_and_fixed_ts_file(input_file,meta_file):
    with tb.open_file(input_file, mode="r") as in_file, tb.open_file(meta_file,mode='r') as meta_file:
        sw = meta_file.root.meta_data[:]
        hw = in_file.root.Hits[:]

        logging.info('opening file %s' % input_file)
        logging.info('number of hits is %s' % hw.shape[0] )
    return sw, hw


def assign_hw_timestamp_corr(sw, hw):

    result = np.zeros(shape=(4, hw.shape[0]))
    
    j = 0
      
    for i in tqdm(range(1, sw.shape[0]-1)):
        try:
            while sw[i]['event_number'] >= hw[j]['event_number']:
                result[0][j] = sw[i - 1]['timestamp_start'] #+ (sw[i - 1]['timestamp_start']- sw[i - 1]['timestamp_stop']) / 2.
                result[1][j] = hw[j]['trigger_time_stamp']
                result[2][j] = sw[i - 1]['event_number']
                result[3][j] = hw[j]['event_number']
                j += 1
        except IndexError:
            break
        
    return result


def plot_hw_vs_sw_timestamps(input_file,meta_file):

    sw, hw = open_hit_and_fixed_ts_file(input_file,meta_file)
    
    title = 'Run_' + os.path.split(input_file)[1][:11] + ' software vs. hardware timestamp'

    result = assign_hw_timestamp_corr(sw, hw)

    result[0] -= result[0][0]
    result[1] *= 25 * 10**-9

    plt.clf()
    plt.plot(result[0],result[1],'.')
#     plt.plot(result[0], result[1], '.')
    
    sel = np.logical_and(result[0] >= 0, result[0] < 10)
    x, y = result[0][sel], result[1][sel]
#     x, y = result[0], result[1]
    par = np.polyfit(x, y, 1)
    yf = np.poly1d(par)
    print 'fit parameters %r ' % par
     
#     print par[0]
#     print np.all(np.diff(result[1]) >= 0)
#     print np.where(np.diff(result[2]) == 0)
    
#     plt.plot(x,y,'.')
    xp = np.linspace(plt.xlim()[0], plt.xlim()[1], 1000)
    plt.plot(xp, yf(xp),label = 'slope = %.3f'% par[0])
    plt.plot(xp,xp,label = 'slope = 1')
    plt.grid()
    plt.title(title)
    plt.xlabel('software ts [s]')
    plt.ylabel('hardware ts [s]')
    plt.xlim(-1,5)
    plt.ylim(-1,5)

    plt.legend()
    plt.show()
#     plt.savefig(input_file[:-3] + '_sw_vs_hw_timestamp_2.pdf')

#
def measure_rate(input_file,meta_file):
    with tb.open_file(input_file, mode="r") as in_file, tb.open_file(meta_file,mode='r') as meta_file:
        meta = meta_file.root.meta_data[:]
        hits = in_file.root.Hits[:]
        
        region_1 = meta['event_number'][(meta['event_number']>9662) & (meta['event_number']<=14331)]
        
        print (1/(np.mean(np.diff(hits['trigger_time_stamp']))*25*10**-9))
        print 'hw rate = %.0f Hz' % (1/(np.mean(np.diff(hits['trigger_time_stamp'][(hits['event_number']>9662) & (hits['event_number']<=14331)]))*25*10**-9))
        print 'sw rate = %.0f Hz' % (1/np.mean(np.diff(region_1-region_1[0])))


if __name__ == "__main__":
    
#     plot_hw_vs_sw_timestamps('/home/niko/git/pyBAR/pybar/test_trigger_rate_tlu/module_0/27_module_0_ext_trigger_scan_fixed.h5',
#                              '/home/niko/git/pyBAR/pybar/test_trigger_rate_tlu/module_0/27_module_0_ext_trigger_scan_interpreted.h5'
#                              )
    
    measure_rate('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/96_module_2_ext_trigger_scan_interpreted_fixed.h5',
                 '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/96_module_2_ext_trigger_scan_interpreted.h5'
                 )
