import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import tables as tb
import math
import sys

from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
from numba import jit, njit

from pybar.analysis import analysis as pyana
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar_fei4_interpreter import data_struct
# from pybar.scans.analyze_example_conversion import output_file_hits_analyzed
from numpy import choose
# from testbeam_analysis.tools.geometry_utils import cartesian_to_spherical
from testbeam_analysis.tools import geometry_utils as gu
from tqdm import tqdm
from collections import Counter

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


# def gauss_fit_row(input_file):
# output_pdf = input_file[:-16] + 'timestamp_distr.pdf'
# t_length = []
#
# with tb.open_file(input_file + '.h5', 'r+') as in_file_h5:
#     data = np.array(in_file_h5.root.meta_data[:])
#     for i in xrange(0,len(data)):
#         t_length.append(data[i][2]-data[i][1])
#
#     binning = (0.048,0.049,0.05,0.051,0.052,0.053,0.054,0.055,0.056 )
#     test = np.random.randn(10000)
#     print
#
#     plt.clf()
#     ax = plt.subplot(111)
#     plt.title('Timestamp distribution for 5M run (no target)')
#     plt.grid()
#     plt.xlabel('timestamp length [s]')
#     plt.ylabel('#')
#     plt.xlim(min(binning),max(binning))
#     ax.set_yscale('log')
#     plt.hist(t_length,bins = binning, align = 'left')
#     plt.legend()
#     print output_pdf
#     plt.savefig(output_pdf)
#     plt.show()
#     ax = plt.axes()
#     box = AnchoredText()
#     ax.add_artist(box)
#     plt.savefig(output_pdf)
#         print output_pdf

def pprint_array(array):  # Just to print the arrays in a nice way
    offsets = []
    for column_name in array.dtype.names:
        sys.stdout.write(column_name)
        sys.stdout.write('\t')
        offsets.append(column_name.count(''))
    for row in array:
        print('')
        for i, column in enumerate(row):
            sys.stdout.write(' ' * (offsets[i] / 2))
            sys.stdout.write(str(column))
            sys.stdout.write('\t')
    print('')

def gauss(x, y, amplitude, mu, sigma, a, b, m):
    gauss = y + amplitude * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))

    poly = a * x + b * x**2 + m
#     box = np.where(abs(x)<=w,1,0)
    return gauss  # + poly


def box_function(x, w):
    return np.where(abs(x) <= w, 1, 0)


def line(x, m, b):
    return m * x + b


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
        rows = np.linspace(1, 337, 336, endpoint=False)
        xticks = np.linspace(0, 17.5, 8, endpoint=True)

        xtick_data = [1, 50, 100, 150, 200, 250, 300, 350]

# coeff, _ = curve_fit(f=gauss, xdata=rows, ydata=data1,p0=(y, amplitude,
# mu, sigma,a,b,c,m,w)) #
        coeff, _ = curve_fit(f=gauss, xdata=rows, ydata=data1, p0=(
            y, amplitude, mu, sigma, a, b, m))  # ,a,b,c,m,w))
#         print coeff[1]*coeff[3]*np.sqrt(2*np.pi)/np.sum(data1)
#         coeff, _ = curve_fit(f=cauchy, xdata=rows, ydata=data1,p0=(amplitude, mu))
#         print coeff
        plt.clf()
        # ('Occupancy per row on second plane')
        plt.title('Occupancy in y on second plane')
        plt.grid()
        plt.xlabel('mm')  # ('row')
        plt.ylabel('#')
        plt.bar(range(1, 337), data1, label='entries per row')
        plt.plot(rows, gauss(rows, *coeff),
                 label='$y + A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c$', color='crimson', linewidth=2.5, alpha=0.8)
        plt.xticks(xtick_data, xticks)
        plt.legend()
        ax = plt.axes()
        box = AnchoredText('mu = %.1f mm \nsigma = %.2f mm \nA = %.f ' % (
            coeff[2] * 0.05, coeff[3] * 0.05, coeff[1]), loc=2)  # \na = %f \nb = %f ,coeff[4],coeff[5]
        ax.add_artist(box)
        plt.savefig(output_pdf)
#         print output_pdf
    # np.sqrt(coeff[1]/np.amax(data1))
    return coeff[1] * coeff[3] * np.sqrt(2 * np.pi) / np.sum(data1)


def plot_clustersize_per_event(data_files, raw_data_convert=True, logy=False):

    for data_file in data_files:

        if raw_data_convert:
            analyze_raw_data = AnalyzeRawData(
                raw_data_file=data_file + '.h5', create_pdf=True)
            # self.dut['TLU']['DATA_FORMAT']
            analyze_raw_data.trigger_data_format = 2
            analyze_raw_data.create_source_scan_hist = True
            analyze_raw_data.create_cluster_size_hist = True
            analyze_raw_data.create_cluster_tot_hist = True
            analyze_raw_data.align_at_trigger = True
            analyze_raw_data.create_cluster_table = True
            analyze_raw_data.create_empty_event_hits = True
            analyze_raw_data.interpreter.set_warning_output(False)
            analyze_raw_data.interpret_word_table()
            analyze_raw_data.interpreter.print_summary()
            analyze_raw_data.plot_histograms()

        timestamp, n_cluster = pyana.analyse_n_cluster_per_event(scan_base=[data_file],
                                                                 combine_n_readouts=500,
                                                                 time_line_absolute=False,
                                                                 percentage=False,
                                                                 output_pdf=None,
                                                                 output_file=data_file + '_n_cluster_per_event_histo.h5'
                                                                 )

        n_cluster = np.array(n_cluster)
        plot_n_cluster = n_cluster[:, 1:5]

        cluster_above_3 = n_cluster[:, 4:].sum(axis=1)

        plot_n_cluster[:, 3] = cluster_above_3
        title = os.path.split(data_file)[1][:-26]

        plt.clf()
        plt.title('Run' + title +
                  ' - Number of cluster per event as a function of time')
        lineObjects = plt.semilogy(
            timestamp, plot_n_cluster, linestyle='None', marker='o', markersize=3)
        plt.xlabel('time [min.]')

        plt.grid()
        plt.legend(lineObjects, ('1 cluster, mean = %.3f' % np.mean(plot_n_cluster[:, 0], axis=0),
                                 '2 cluster, mean = %.3f' % np.mean(
                                     plot_n_cluster[:, 1], axis=0),
                                 '3 cluster, mean = %.3f' % np.mean(
                                     plot_n_cluster[:, 2], axis=0),
                                 '4 or more cluster\nmean = %.3f' % np.mean(
                                     plot_n_cluster[:, 3], axis=0)
                                 )
                   )
        plt.savefig(data_file + '_n_cluster_per_event.pdf')

        cluster_sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.clf()
        plt.title('Run' + title + ' - Number of cluster per event')

        if logy:
            plt.yscale('log')
            lineObjects = plt.plot(cluster_sizes, n_cluster.sum(
                axis=0), linestyle='None', marker='o', markersize=3)
        else:
            lineObjects = plt.plot(
                timestamp, plot_n_cluster, linestyle='None', marker='o', markersize=1)

        plt.xlabel('cluster size')
        plt.grid()
        plt.savefig(data_file + '_n_cluster_cluster_size_logscale.pdf')


def slice_per_spill(data_file):
    '''
    look for irregularities in spills to compare with emulsion data from PIX1
    data_file: interpreted scan file with hit data
    '''

#     output_pdf = data_file[:-16] + 'sliced.pdf'
    with tb.open_file(data_file + '.h5', 'r+') as in_file_h5:
        data = in_file_h5.root.meta_data
        hits = in_file_h5.root.Hits

        hit_array = []
        for row in hits.iterrows():
            # row[0] is event number, row[5] is column, row[6] is row
            hit_array.append([row[0], row[5], row[6]])
        hit_array = np.array(hit_array)

    # sel_events = hit_array[(hit_array[:,0]>=1569272) &
    # (hit_array[:,0]<1800000)]# 1585412
        sel_events = hit_array#[(hit_array[:, 0] >= 1) & (hit_array[:, 0] < 16000)]  # 120792

    #     time = meta_data[meta_data[:,0] == sel_events[:,0]]
        print 'start event: %s' % sel_events[0, 0]
        print 'stop event: %s' % sel_events[-1, 0]
        print 'number of analyzed events = %s' % len(sel_events)

        meta_data = []
        for row in data.iterrows():
            if row[0] >= sel_events[0, 0] and row[0] <= sel_events[-1, 0]:
                meta_data.append([row[0], row[1] - data[0][1], row[2]])
    # meta_data.append([row[0], row[1]-1506634273, row[2]]) # row[0] is event
    # number, row[1] is timestamp start, row[2] is timestamp stop
        meta_data = np.array(meta_data)

        timestamps = []
        print len(meta_data)
        for i in tqdm(range(0, len(meta_data) - 1)):
            for b in range(0, len(sel_events)):
                if sel_events[b, 0] >= meta_data[i, 0] and sel_events[b, 0] < meta_data[i + 1, 0]:
                    timestamps.append(
                        [meta_data[i, 2], sel_events[b, 0], sel_events[b, 1], sel_events[b, 2]])

        timestamps = np.array(timestamps)

        plt.clf()
        plt.grid()
        plt.title('Event number vs. scan time')
        plt.xlabel('time [s]')
        plt.ylabel('event number [#]')
#         plt.xlim(65, 95)
        plt.plot(meta_data[:, 1], meta_data[:, 0],
                 linestyle='None', marker='o', markersize=1)

        plt.savefig(data_file[:-11] + 'sliced.pdf')

    #     plt.clf()
    #     plt.grid()
    #     plt.plot(sel_events[:,2],sel_events[:,0],linestyle='None',marker='o',markersize=0.2)
    #     plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/75_row_per_eventnumber.pdf')

        plt.clf()
        plt.grid()
        plt.plot(timestamps[:, 0] - data[0][1], timestamps[:, 3], linestyle='None',
                 marker='o', markersize=0.5, markevery=100)  # 1506634273
        plt.savefig(data_file[:-11] + 'row_vs_time.pdf')


def plane_movement_speed(data_file):

    logging.info('calculating Plane movement speed')

    with tb.open_file(data_file, 'r+') as in_file_h5:
        hit_data = np.array(in_file_h5.root.Hits[:])
        meta_data = np.array(in_file_h5.root.meta_data[:])

        hit, meta, corr = [], [], []
        for row in meta_data:
            meta.append((row[0], row[1]))
#         for i in xrange (0,len(meta_data)):
#             meta.append((meta_data[i][0],meta_data[i][1]))
        for row in hit_data:
            hit.append((row[0], row[5], row[6]))
        for i in xrange(0, len(hit)):
            for a in xrange(1, len(meta) - 1):
                # and meta[a][0]>14400 and meta[a][0]<15500
                if meta[a][0] <= hit[i][0] and hit[i][0] < meta[a + 1][0] and meta[a][0] > 14400 and meta[a][0] < 15500:
                    corr.append((hit[i][0], hit[i][1], hit[i]
                                 [2], meta[a][1], meta[a][0]))
        corr = np.array(corr)

    ''' line fit to determine events/s '''
    timing_x = corr[:, 3] - corr[:, 3][0]

    coeff2, _ = curve_fit(f=line, xdata=timing_x, ydata=corr[:, 0], p0=(0, 0))

#     x = np.array(corr[:,0]-coeff2[1])
#     y = np.array(corr[:,1])

    ''' line fit to determine columns per eventnumber '''
    fit = corr[(corr[:, 0] > 14550.) & (corr[:, 0] < 15240.)& (corr[:, 1] > 0.) & (corr[:, 1] < 81.)]
    fit[:, 0] = fit[:, 0] - coeff2[1]
    coeff, _ = curve_fit(f=line, xdata=fit[:, 0], ydata=fit[:, 1], p0=(0, 0))  # 20*0.25,-40

    mm_per_s = np.average(coeff[0] * coeff2[0] * 0.25)
    column_per_s = np.average(coeff[0] * coeff2[0])
    logging.info('horizontal movement speed of plane = %.2f mm/s or %f column/s' %
                 (mm_per_s, column_per_s))
#     logging.info('vertical movement speed = %')

    ''' plot event number vs. relative timestamp '''

    logging.info('plotting line fits for movement speed')

    plt.clf()
    plt.grid()
    plt.title('Events per ms')
    plt.xlabel('rel. time [s]')
    plt.ylabel('Event number')
    plt.plot(timing_x, corr[:, 0], linestyle='None',
             marker='o', markevery=50, label='data')
    plt.plot(timing_x, line(timing_x, *coeff2),
             color='crimson', markevery=50, label='fit')
    plt.legend()
    ax = plt.axes()

    box = AnchoredText('$f = m \cdot x + b$\nm = %.3f\nb = %.f\nv = %f' %
                       (coeff2[0], coeff2[1], mm_per_s), loc=4)
    ax.add_artist(box)
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_evts_per_ms.pdf')

    ''' plot event number vs column '''

    plt.clf()
    plt.title('Horizontal movement of 1st plane')
    plt.xlabel('column')
    plt.ylabel('rel event number')
    plt.grid()

    plt.plot(corr[:, 1], corr[:, 0] - coeff2[1], linestyle='None',
             markersize=2.5, marker='o', label='data')
    plt.plot(line(fit[:, 0], *coeff), fit[:, 0],
             color='crimson', label='line fit')

    plt.legend()
    ax2 = plt.axes()
    mm_per_event = np.average(coeff[0] + coeff[1] / corr[:, 0]) * 0.25

    box2 = AnchoredText('$f = m \cdot x + b$ \nm = %.5f \nb = %.f \nmm per evt = %f' %
                        (coeff[0], coeff[1], mm_per_event), loc=4)
    ax2.add_artist(box2)
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_time_correlation.pdf')

    return mm_per_s, column_per_s, corr


def translate_plane_to_restframe(data_file, v_x, v_y):
    output_file = data_file + '_translation'

    with tb.open_file(data_file + '.h5', 'r+') as in_file_h5:
        hit_data = in_file_h5.root.Hits
        meta_data = in_file_h5.root.meta_data[:]

        hit, meta, corr = [], [], []
        for row in meta_data:
            meta.append((row[0], row[1]))
        for row in hit_data:
            hit.append((row[0], row[5], row[6]))
        for i in xrange(0, len(hit)):
            for a in xrange(1, len(meta) - 1):
                # and meta[a][0]>14400 and meta[a][0]<15500
                if meta[a][0] <= hit[i][0] and hit[i][0] < meta[a + 1][0]:
                    corr.append((hit[i][0], hit[i][1], hit[i][2], meta[a][1], meta[a][0]))
        corr = np.array(corr)
        hit = np.array(hit)
        print corr[corr[:, 0] == hit[:, 0],3]


        hit_data.cols.column = hit_data.cols.column - v_x * corr[corr[:, 0] == hit[:, 0],3] - corr[0,3]
#         hit_data[:, 6] = hit_data[:, 6] - v_y * corr[corr[:, 0] == hit[:, 0],3] - corr[0,3]
#         hit_data[:, 5] = hit_data[:, 5] - v_x * corr[corr[:, 0] == hit[:, 0],3] - corr[0,3]
#         hit_data[:, 6] = hit_data[:, 6] - v_y * corr[corr[:, 0] == hit[:, 0],3] - corr[0,3]

        print hit_data.cols.column[1]
        raise
        with tb.open_file(output_file + '.h5', 'w') as out_file_h5:
            hit_table_description = data_struct.HitInfoTable().columns.copy()
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=hit_table_description,
                                                     title='Selected hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            hit_table_out.append(hit_data)

    logging.info('translated data saved to %s' % output_file)
    return hit_data, output_file

class Final_track_data(tb.IsDescription):
    region = tb.Int64Col(pos=0)
    event_number = tb.Int64Col(pos=1)
    sw_timestamp = tb.Float64Col(pos=2)
    x = tb.Float64Col(pos=3)
    y = tb.Float64Col(pos=4)
    z = tb.Float64Col(pos=5)
    phi = tb.Float64Col(pos=6)
    theta = tb.Float64Col(pos=7)
    track_chi_2 = tb.Float64Col(pos=8)
    xerr_dut_0 = tb.Float64Col(pos=9)
    yerr_dut_0 = tb.Float64Col(pos=9)
#     alpha = tb.Float64Col(pos=3)
#     beta = tb.Float64Col(pos=4)
#     gamma = tb.Float64Col(pos=5)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def transform_to_emu_plane(input_file, table_name ,meta_data_file, output_file=None):
    
    if not output_file:
        output_file_name = input_file[:-3] + '_reduced.h5'
    with tb.open_file(input_file, mode="r") as in_file, tb.open_file(meta_data_file, mode='r') as meta_file:
        with tb.open_file(output_file_name, mode="w") as out_file:
            meta_data = meta_file.root.meta_data[:]
           
            node = in_file.get_node(in_file.root, table_name)
            event_number = node[:]['event_number']
            x = node[:]['offset_0']
            y = node[:]['offset_1']
            z = node[:]['offset_2']
            offset_vec = np.column_stack((x,y,z))
            direction_vec = np.column_stack((node[:]['slope_0'],
                                             node[:]['slope_1'],   
                                             node[:]['slope_2']))
            
#             alpha = np.apply_along_axis(angle_between,1,direction_vec,(1,0,0))
#             beta = np.apply_along_axis(angle_between,1,direction_vec,(0,1,0))
#             gamma = np.apply_along_axis(angle_between,1,direction_vec,(0,0,1))

            '''transformation to emulsion plane, and transformation to spherical coordinate system'''
            new_coords = gu.get_line_intersections_with_plane(line_origins=offset_vec,
                                                             line_directions=direction_vec,
                                                             position_plane=np.array([0,0,-156000]),
                                                             normal_plane=np.array([0,0,1]))
            phi, theta = np.zeros_like(z),np.zeros_like(z)
            phi, theta, _ = np.apply_along_axis(gu.cartesian_to_spherical,
                                                0,
                                                direction_vec[:,0],
                                                direction_vec[:,1],
                                                direction_vec[:,2])
#             new_data = np.column_stack((event_number,x,y,z,phi,theta))
            descr = Final_track_data
            dtypes = {'names': ['region','event_number', 'sw_timestamp','x', 'y', 'z', 'phi', 'theta', 'track_chi2','xerr_dut_0','yerr_dut_0'],
                      'formats': ['int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64','float64','float64','float64']}
            new_data = np.zeros_like(node,
                   dtype=dtypes)

            timestamps = np.zeros_like(event_number,dtype=float)

            ''' assigning sw timestamps to all tracks, based on events'''
                
            j = 0
            for i in range(1, event_number.shape[0]):
                try:
                    while meta_data[i]['event_number'] >= event_number[j]:
                        timestamps[j] = meta_data[i-1]['timestamp_start'] # + (meta_data[i - 1]['timestamp_start']- meta_data[i - 1]['timestamp_stop']) / 2.
                        j += 1
                except IndexError:
                    break
            
            timestamps -= timestamps[0]

            ''' filling new array'''
            
            new_data['event_number'] = event_number
            new_data['sw_timestamp'] = timestamps
            new_data['x'] = new_coords[:,0]
            new_data['y'] = new_coords[:,1]
            new_data['z'] = new_coords[:,2]
            new_data['phi'] = phi
            new_data['theta'] = theta
            new_data['track_chi2'] = node[:]['track_chi2']
            new_data['xerr_dut_0'] = node[:]['xerr_dut_0']
            new_data['yerr_dut_0'] = node[:]['yerr_dut_0']

            new_data['region'][(new_data['event_number']>9662) & (new_data['event_number']<=14331)] = 1
            new_data['region'][(new_data['event_number']>14332) & (new_data['event_number']<=18964)] = 2
            new_data['region'][(new_data['event_number']>18964) & (new_data['event_number']<=23702)] = 3
            new_data['region'][(new_data['event_number']>23702) & (new_data['event_number']<=28411)] = 4
            new_data['region'][(new_data['event_number']>28411)] = 5
            new_tables = []

            '''finally creating and filling table for .h5 document'''
            
            new_table = out_file.create_table(where=out_file.root,
                                            name='Tracks_for_pix2',
                                            description=descr,
                                            title='Tracks fitted for DUT 0',
                                            filters=tb.Filters(complib='blosc',
                                                   complevel=5,
                                                   fletcher32=False))

            new_table.append(new_data)
            new_table.flush()

            logging.info('New data file with %s tables written to %s'% (len(new_tables)+1 ,output_file_name))
    return output_file


def open_hit_and_fixed_ts_file(input_file,meta_file):
    with tb.open_file(input_file, mode="r") as in_file, tb.open_file(meta_file,mode='r') as meta_file:
        sw = meta_file.root.meta_data[:]
        hw = in_file.root.Hits[:]
        
        plot_data_array = np.zeros(shape=(hw.shape[0],3))
        logging.info('opening file %s' % input_file)
        logging.info('number of hits is %s' % hw.shape[0] )
    return sw, hw, plot_data_array

        
@jit(nopython=True, parallel=True)
def assign_hw_timestamp(sw,hw,plot_data_array):
    
    j = 0
      
    for i in range(0, sw.shape[0]-1):
        for j in range(0, hw.shape[0]):
            if sw[i]['event_number'] <= hw[j]['event_number']:
#                 plot_data.append([sw[i]['timestamp_start'],hw[b]['trigger_time_stamp'],sw[i]['event_number'], hw[b]['event_number']])
                plot_data_array[j][0] = sw[i]['timestamp_stop']
                plot_data_array[j][1] = hw[j]['trigger_time_stamp']
                plot_data_array[j][2] = sw[i]['event_number']
                plot_data_array[j][3] = hw[j]['event_number']
                
    plot_data_array[:,0] = plot_data_array[:,0] - plot_data_array[:,0][0]
    plot_data_array[:,1] = plot_data_array[:,1] *25 / 1e9
        
    return plot_data_array


def assign_hw_timestamp_corr(sw, hw):

    result = np.zeros(shape=(4, hw.shape[0]))
    
    j = 0
      
    for i in range(1, sw.shape[0]-1):
        try:
            while sw[i]['event_number'] >= hw[j]['event_number']:
                result[0][j] = sw[i - 1]['timestamp_start'] + (sw[i - 1]['timestamp_start']- sw[i - 1]['timestamp_stop']) / 2.
                result[1][j] = hw[j]['trigger_time_stamp']
                result[2][j] = sw[i - 1]['event_number']
                result[3][j] = hw[j]['event_number']
                j += 1
        except IndexError:
            break
        
    return result

def plot_xy(x,y,title,xlabel, ylabel,xrange,yrange,output_file = None):

    n_hits = len(y)
    plt.clf()
    plt.title(title)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#     plt.xlim(xrange)
#     plt.ylim(yrange)
    plt.plot(x, y,label = 'number of hits = %r' % n_hits ,linestyle = 'None', marker='o', markersize = 1.5)

    plt.legend()
    if not output_file:
        plt.show()
    else:
        plt.savefig(output_file)


def plot_hw_vs_sw_timestamps(input_file,meta_file):
    
    output_file = input_file[:-3] + '_sw_vs_hw_timestamp.pdf'
    sw, hw, plot_data_array = open_hit_and_fixed_ts_file(input_file,meta_file)
    
    title = 'Run_' + os.path.split(input_file)[1][:11] + ' software vs. hardware timestamp'
    
#     plot_data = assign_hw_timestamp(sw,hw,plot_data_array)
#     plot_xy(plot_data[:,0],plot_data[:,1],
#             title = title, xlabel = 'software ts [s]',
#             ylabel = 'hardware ts [s]', 
#             xrange = (0,100),yrange=(0,12),
#             output_file = output_file)

    result = assign_hw_timestamp_corr(sw, hw)
    
    histogram_delta_t(result,output_file = input_file[:-38] + '_delta_t_histo.pdf')
    
    result[0] -= (1506623943.279761 +50. )#(result[0][0] + 50.)
    result[1] *= 25e-9
    plt.clf()
    plt.plot(result[0], result[1], '.')
    sel = np.logical_and(result[0] > 1, result[0] < 6)
    x, y = result[0][sel], result[1][sel]
    xp = np.linspace(plt.xlim()[0], plt.xlim()[1], 1000)
     
#     par = np.polyfit(x, y, 1)
#     yf = np.poly1d(par)    
#     print par[0]
#     print np.all(np.diff(result[1]) >= 0)
#     plt.plot(xp, yf(xp),label = 'slope = %.3f'% par[0])
#     plt.plot(xp,xp,label = 'slope = 1')
    plt.grid()
    plt.title(title)
    plt.xlabel('software ts [s]')
    plt.ylabel('hardware ts [s]')

    plt.legend()    
    plt.savefig(input_file[:-3] + '_sw_vs_hw_timestamp.pdf')
    
   
def histogram_delta_t(result,output_file=None):
    
    delta_t = np.diff(result[1])
    
    histo,edges = np.histogram(delta_t,bins = 200)
    
    small = delta_t[np.where((delta_t <= 2048) & (delta_t > 0))]
    print 'entries with 0 < delta_t < 2048:'
    print np.count_nonzero(small)
    large = delta_t[np.where(delta_t >= 2**15)]
    print 'entries with delta t larger 2^15 = 32786:'
    print np.count_nonzero(large)
    
    plt.clf()
    plt.grid()
    plt.title('delta t for consecutive events')
    plt.xlabel('delta_t [25ns]')
    plt.ylabel('counts')
    plt.plot(edges[1:-1],histo[1:])
    if output_file == None:
        plt.show()
    else:
        plt.savefig(output_file)
        logging.info('histogram saved to %s' %output_file)
    
    
    
if __name__ == "__main__":

    n_cluster_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/75_module_2_ext_trigger_scan',
                       '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/85_module_2_ext_trigger_scan',
                       #                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/94_module_2_ext_trigger_scan',
                       '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/86_module_2_ext_trigger_scan',
                       '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/96_module_2_ext_trigger_scan']


#     slice_per_spill('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/69_module_2_ext_trigger_scan_interpreted')

    logging.getLogger().setLevel(logging.DEBUG)

#     pyana.analyze_event_rate(scan_base=['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan'],
#                             combine_n_readouts=8,
#                             time_line_absolute = False,
#                             output_pdf = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_event-rate.pdf')
#
#     print pyana.analyze_beam_spot(scan_base=['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan'],
#                             combine_n_readouts=10000,
#                             chunk_size = 1000000,
# output_pdf =
# '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/75_module_2_beam_spot.pdf')

    data_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_interpreted'
#     v = plane_movement_speed(data_file + '.h5')
#     translate_plane_to_restframe(data_file, v[1], v_y=0)
#     print angle_between((-0.0044927,-0.03259097,0.999458675),(0,0,1)) #*180/np.pi
#     slice_per_spill('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/matching_moving_module_0/pix2/converted_files/70_plane_1_module_2_ext_trigger_scan_interpreted')
    meta_data_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/matching_moving_module_0/pix2/converted_files/70_plane_1_module_2_ext_trigger_scan_interpreted.h5'
    input_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/matching_moving_module_0/pix2/output/Tracks_aligned.h5'
    
    transform_to_emu_plane(input_file, table_name = 'Tracks_DUT_0', meta_data_file = meta_data_file)
    raise
    pix1_files_fixed = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/69_module_2_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/69_module_4_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/54_module_0_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/54_module_2_ext_trigger_scan_interpreted_fixed.h5']
    pix1_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/69_module_2_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/69_module_4_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/54_module_0_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX1/54_module_2_ext_trigger_scan_interpreted.h5'] 

    pix2_files_fixed = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/70_module_2_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/70_module_4_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/55_module_0_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/55_module_2_ext_trigger_scan_interpreted_fixed.h5']
    pix2_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/70_module_2_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/70_module_4_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/55_module_0_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX2/55_module_2_ext_trigger_scan_interpreted.h5']
    
    pix3_files_fixed = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/71_module_2_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/71_module_4_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/56_module_0_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/56_module_2_ext_trigger_scan_interpreted_fixed.h5']
    pix3_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/71_module_2_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/71_module_4_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/56_module_0_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX3/56_module_2_ext_trigger_scan_interpreted.h5']
    
    pix4_files_fixed = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_2_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_4_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_0_ext_trigger_scan_interpreted_fixed.h5',
                        '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_2_ext_trigger_scan_interpreted_fixed.h5']
    pix4_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_2_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_4_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_0_ext_trigger_scan_interpreted.h5',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_2_ext_trigger_scan_interpreted.h5']
    

    for i,file in enumerate(pix2_files_fixed) :
        plot_hw_vs_sw_timestamps(file,pix2_files[i])

