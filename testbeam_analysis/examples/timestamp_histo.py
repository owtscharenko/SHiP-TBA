import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
import math
import csv
from pybar.analysis import analysis as pyana
import logging
import os
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from scipy.interpolate._interpolate import block_average_above_dddd




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

def gauss(x,y,amplitude, mu, sigma,a,b,m ):
    gauss = y + amplitude * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))
    
    poly = a*x + b*x**2 + m
#     box = np.where(abs(x)<=w,1,0)
    return gauss #+ poly


def box_function(x,w):
    return np.where(abs(x)<=w,1,0)

def line(x,m,b):
    return m*x+b

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
   


def plot_clustersize_per_event(data_files, raw_data_convert=True,logy=False):
    
    for data_file in data_files:

        if raw_data_convert: 
            analyze_raw_data = AnalyzeRawData(raw_data_file=data_file + '.h5', create_pdf=True)
            analyze_raw_data.trigger_data_format = 2 # self.dut['TLU']['DATA_FORMAT']
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
                                                                    time_line_absolute = False,
                                                                    percentage = False,
                                                                    output_pdf = None,
                                                                    output_file = data_file + '_n_cluster_per_event_histo.h5'
                                                                    )
 
        n_cluster = np.array(n_cluster)
        plot_n_cluster = n_cluster[:,1:5]

        cluster_above_3 = n_cluster[:,4:].sum(axis=1)

        plot_n_cluster[:,3] = cluster_above_3
        title = os.path.split(data_file)[1][:-26]

        plt.clf()
        plt.title('Run' + title + ' - Number of cluster per event as a function of time')
        lineObjects = plt.semilogy(timestamp, plot_n_cluster,linestyle='None', marker='o',markersize=3)
        plt.xlabel('time [min.]')
        
        plt.grid()
        plt.legend(lineObjects, ('1 cluster, mean = %.3f' % np.mean(plot_n_cluster[:,0], axis=0),
                                 '2 cluster, mean = %.3f' % np.mean(plot_n_cluster[:,1], axis=0) ,
                                 '3 cluster, mean = %.3f' % np.mean(plot_n_cluster[:,2], axis=0),
                                 '4 or more cluster\nmean = %.3f' % np.mean(plot_n_cluster[:,3], axis=0)
                                 )
                    )
        plt.savefig(data_file + '_n_cluster_per_event.pdf')
     
        cluster_sizes = [0,1,2,3,4,5,6,7,8,9]
        plt.clf()
        plt.title('Run' + title + ' - Number of cluster per event')
        
        if logy:
            plt.yscale('log')
            lineObjects = plt.plot(cluster_sizes, n_cluster.sum(axis=0),linestyle='None',marker='o',markersize=3) #
        else:
            lineObjects = plt.plot(timestamp, plot_n_cluster,linestyle='None', marker='o',markersize=1)

        plt.xlabel('cluster size')
        plt.grid()
        plt.savefig(data_file + '_n_cluster_cluster_size_logscale.pdf')


def slice_per_spill(data_file):
    
    '''
    look for irregularities in spills to compare with emulsion data from PIX1
    data_file: interpreted scan file with hit data
    '''
    
    input_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_2/69_module_2_ext_trigger_scan_interpreted'
#     output_pdf = data_file[:-16] + 'sliced.pdf' 
    with tb.open_file(data_file + '.h5', 'r+') as in_file_h5:
        data = in_file_h5.root.meta_data
        hits = in_file_h5.root.Hits

        hit_array = []
        for row in hits.iterrows():
            hit_array.append([row[0],row[5],row[6]]) # row[0] is event number, row[5] is column, row[6] is row
        hit_array = np.array(hit_array)
    #     print hit_array[:,0]
        
    #     sel_events = hit_array[(hit_array[:,0]>=1569272) & (hit_array[:,0]<1800000)]# 1585412
        sel_events = hit_array[(hit_array[:,0]>=1) & (hit_array[:,0]<10000)]# 1585412
    #     print len(plot_data)
    #     time = meta_data[meta_data[:,0] == sel_events[:,0]]
        print 'start event: %s' % sel_events[0,0]
        print 'stop event: %s' % sel_events[-1,0]
        print 'number of analyzed events = %s' %len(sel_events)
        
        meta_data = []
        for row in data.iterrows():
            if row[0] >= sel_events[0,0] and row[0] <= sel_events[-1,0]:
                meta_data.append([row[0], row[1]-data[0][1], row[2]])
    #             meta_data.append([row[0], row[1]-1506634273, row[2]]) # row[0] is event number, row[1] is timestamp start, row[2] is timestamp stop
        meta_data = np.array(meta_data)
    #     print len(meta_data)
    
        timestamps = []
        print len(meta_data)
        for i in range(0,len(meta_data)-1):
            for b in range(0,len(sel_events)):
                if sel_events[b,0]>= meta_data[i,0] and sel_events[b,0] < meta_data[i+1,0]:
                    timestamps.append([meta_data[i,2],sel_events[b,0],sel_events[b,1],sel_events[b,2]])
            print i
        timestamps = np.array(timestamps)            
        
        plt.clf()
        plt.grid()
        plt.title('Event number vs. scan time')
        plt.xlabel('time [s]')
        plt.ylabel('event number [#]')
        plt.xlim(65,95)
        plt.plot(meta_data[:,1],meta_data[:,0],linestyle='None',marker='o',markersize=1)
    
        plt.savefig(data_file[:-11]+ 'sliced.pdf')
    
        
    #     plt.clf()
    #     plt.grid()
    #     plt.plot(sel_events[:,2],sel_events[:,0],linestyle='None',marker='o',markersize=0.2)
    #     plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/75_row_per_eventnumber.pdf')
        
        plt.clf()
        plt.grid()
        plt.plot(timestamps[:,0]-data[0][1],timestamps[:,3],linestyle='None',marker='o', markersize=0.5,markevery=100) #1506634273
        plt.savefig(data_file[:-11]+ 'row_vs_time.pdf')
    

def beam_spot_vs_time(input_hits_file):
    
    pyana.analyze_beam_spot(scan_base=[input_hits_file],
                            combine_n_readouts=1,
                            chunk_size = 1000,
                            plot_occupancy_hists = True,
                            output_pdf =input_hits_file + '_beam_spot.pdf',
                            output_file = input_hits_file + '_beam_spot.h5')


def plot_beam_spot_vs_time(input_file,indices):
    with tb.open_file(input_file, mode="r+") as in_file_h5:
        data = in_file_h5.root.Beamspot[:]
        timestamp = data['time_stamp'][indices[0]:indices[1]]
        column = data['x'][indices[0]:indices[1]]
        row = data['y'][indices[0]:indices[1]]

        plt.clf()
        plt.grid()
        plt.xlim(0,80)
        plt.ylim(100,250)
        plt.xlabel('column')
        plt.ylabel('row')
        plt.title('Beam position vs. time')
        plt.plot(column, row)
        plt.savefig(input_file[:-3] + '_actual_beam_spot.pdf')
        
    return timestamp, column, row
    

def plane_movement_speed(data_file):
    
    with tb.open_file(data_file, 'r+') as in_file_h5:
        hit_data = np.array(in_file_h5.root.Hits[:])
        meta_data = np.array(in_file_h5.root.meta_data[:])
    
        hit, meta,corr = [],[],[]
        for i in xrange (0,len(meta_data)):
            meta.append((meta_data[i][0],meta_data[i][1]))
        for i in xrange (0,len(hit_data)):
            hit.append((hit_data[i][0],hit_data[i][5]))
        for i in xrange (0,len(hit)):
            for a in xrange(1,len(meta)-1):
                if meta[a][0] < hit[i][0] and hit[i][0] < meta[a+1][0] and meta[a][0]>14400 and meta[a][0]<15500: # and meta[a][0]>14400 and meta[a][0]<15500
                    corr.append((hit[i][0],hit[i][1],meta[a][1],meta[a][0]))
        corr = np.array(corr)
        
    timing_x = corr[:,2]-corr[:,2][0]

    coeff2, _ = curve_fit(f=line, xdata=timing_x, ydata=corr[:,0],p0=(0,0))
    
#     x = np.array(corr[:,0]-coeff2[1])
#     y = np.array(corr[:,1])    

    fit = corr[(corr[:,0] > 14550.) & (corr[:,0]< 15240.) & (corr[:,1] > 0.)& (corr[:,1] < 81.)]
    fit[:,0] = fit[:,0] - coeff2[1]
    coeff, _ = curve_fit(f=line, xdata=fit[:,0], ydata=fit[:,1],p0=(0,0)) # 20*0.25,-40
    
    mm_per_s = np.average(coeff[0]*coeff2[0]*0.25)
    col_per_s = np.average(coeff[0]*coeff2[0])
    
    plt.clf()
    plt.grid()
    plt.title('Events per ms')
    plt.plot(timing_x,corr[:,0],linestyle='None',marker='o',markevery=50,label='data')
    plt.plot(timing_x,line(timing_x, *coeff2),color='crimson',markevery=50,label = 'fit')
    plt.legend()
    ax = plt.axes()

    box = AnchoredText('$f = m \cdot x + b$\nm = %.3f\nb = %.f\nv = %f' %(coeff2[0],coeff2[1],mm_per_s), loc= 4)
    ax.add_artist(box)
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_evts_per_ms.pdf')

    plt.clf()
    plt.title('Horizontal movement of 1st plane')
    plt.xlabel('column')
    plt.ylabel('rel event number')
    plt.grid()

    plt.plot(corr[:,1], corr[:,0]-coeff2[1], linestyle='None',markersize=2.5,marker='o', label = 'data')
    plt.plot(line(fit[:,0],*coeff),fit[:,0],color='crimson',label = 'line fit')
#     plt.plot(dummy,line(dummy,*coeff),color='crimson',label = 'line fit')
    plt.legend()
    ax = plt.axes()
    mm_per_event = np.average(coeff[0]+coeff[1]/corr[:,0])*0.25
#     print mm_per_event
    box = AnchoredText('$f = m \cdot x + b$ \nm = %.5f \nb = %.f \nmm per evt = %f' %(coeff[0],coeff[1],mm_per_event), loc= 4)
    ax.add_artist(box)
    plt.savefig('/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_time_correlation.pdf')

    
    return mm_per_s, col_per_s





if __name__ == "__main__":

    n_cluster_files=['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/files_vadim/75_module_2_ext_trigger_scan',
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
    input_file = [ '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_1/67_module_1_ext_trigger_scan',
                  '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan']
    
    indices = [(391,407),(402,421)]
    
#     beam_spot_vs_time(input_file)
    t ,x, y = [],[],[]
    for i, file in enumerate(input_file):
        th,xh,yh = plot_beam_spot_vs_time(file + '_beam_spot.h5',indices=indices[i])
        t.append(th)
        x.append(xh)
        y.append(yh)
    
    print np.vstack((t,x,y)).shape
    
    
    plt.clf()
    plt.plot(x,y)
    plt.show()
#     print pyana.analyze_beam_spot(scan_base=['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan'],
#                             combine_n_readouts=100,
#                             chunk_size = 1000,
#                             output_pdf = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_beam_spot.pdf')
    
#     data_file = '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_interpreted'
#     x_speed = plane_movement_speed(data_file + '.h5')
#     print 'speed in x direction = %f mm/s \n= %f col/s' % x_speed
