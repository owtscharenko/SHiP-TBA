"""This script prepares FE-I4 test beam raw data recorded by pyBAR to be analyzed by the simple python test beam analysis.
An installation of pyBAR is required: https://silab-redmine.physik.uni-bonn.de/projects/pybar
- This script does for each DUT in parallel
  - Create a hit tables from the raw data
  - Align the hit table event number to the trigger number to be able to correlate hits in time
  - Rename and select hit info needed for further analysis.
"""

import logging
import numpy as np
from numba import njit
import tables as tb
from multiprocessing import Pool

from pybar.analysis import analysis_utils
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar_fei4_interpreter import data_struct


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def analyze_raw_data(input_file, trigger_data_format):  # FE-I4 raw data analysis
    '''Std. raw data analysis of FE-I4 data. A hit table is created for further analysis.

    Parameters
    ----------
    input_file : pytables file
    output_file_hits : pytables file
    '''
    with AnalyzeRawData(raw_data_file=input_file, create_pdf=True) as analyze_raw_data:
        #analyze_raw_data.align_at_trigger_number = True  # if trigger number is at the beginning of each event activate this for event alignment
        analyze_raw_data.use_trigger_time_stamp = False  # the trigger number is a time stamp
        analyze_raw_data.trigger_data_format = trigger_data_format
        analyze_raw_data.use_tdc_word = False
        analyze_raw_data.create_hit_table = True
        analyze_raw_data.create_meta_event_index = True
        analyze_raw_data.create_trigger_error_hist = True
        analyze_raw_data.create_rel_bcid_hist = True
        analyze_raw_data.create_error_hist = True
        analyze_raw_data.create_service_record_hist = True
        analyze_raw_data.create_occupancy_hist = True
        analyze_raw_data.create_tot_hist = False
#         analyze_raw_data.n_bcid = 16
#         analyze_raw_data.max_tot_value = 13
        analyze_raw_data.interpreter.create_empty_event_hits(False)
#         analyze_raw_data.interpreter.set_debug_output(False)
#         analyze_raw_data.interpreter.set_info_output(False)
        analyze_raw_data.interpreter.set_warning_output(False)
#         analyze_raw_data.interpreter.debug_events(0, 1, True)
        analyze_raw_data.interpret_word_table()
        analyze_raw_data.interpreter.print_summary()
        analyze_raw_data.plot_histograms()


def process_dut(raw_data_file, trigger_data_format=0, do_corrections=False,transpose=False):
    ''' Process and format raw data.
    Parameters
    ----------
    raw_data_file : string
        file with raw data
    fix_trigger_number : bool
        activate trigger number fixing

    Returns
    -------
    Tuple: (event_number, trigger_number, hits['trigger_number'])
    or None if not fix_trigger_number
    '''

    if do_corrections is True:
        fix_trigger_number, fix_event_number = False, True
    else:
        fix_trigger_number, fix_event_number = False, False

    analyze_raw_data(raw_data_file, trigger_data_format=trigger_data_format)
    fix_time_stamp(hits_file_in=raw_data_file[:-3] + '_interpreted.h5',
                   hits_file_out=raw_data_file[:-3] + '_interpreted_fixed.h5')
    # raise for test
    ret_value = align_events(raw_data_file[:-3] + '_interpreted_fixed.h5', raw_data_file[:-3] + '_event_aligned.h5', fix_trigger_number=fix_trigger_number, fix_event_number=fix_event_number)
    format_hit_table(raw_data_file[:-3] + '_event_aligned.h5', raw_data_file[:-3] + '_aligned.h5',transpose)


def align_events(input_file, output_file, fix_event_number=True, fix_trigger_number=True, chunk_size=20000000):
    ''' Selects only hits from good events and checks the distance between event number and trigger number for each hit.
    If the FE data allowed a successful event recognition the distance is always constant (besides the fact that the trigger number overflows).
    Otherwise the event number is corrected by the trigger number. How often an inconsistency occurs is counted as well as the number of events that had to be corrected.
    Remark: Only one event analyzed wrong shifts all event numbers leading to no correlation! But usually data does not have to be corrected.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    chunk_size :  int
        How many events are read at once into RAM for correction.
    '''
    logging.info('Align events to trigger number in %s' % input_file)

    with tb.open_file(input_file, 'r') as in_file_h5:
        hit_table = in_file_h5.root.Hits
        jumps = []  # variable to determine the jumps in the event-number to trigger-number offset
        n_fixed_hits = 0  # events that were fixed
    
        with tb.open_file(output_file, 'w') as out_file_h5:
            hit_table_description = data_struct.HitInfoTable().columns.copy()
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=hit_table_description, title='Selected hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False), chunkshape=(chunk_size,))
 
            # Correct hit event number
            for hits, _ in analysis_utils.data_aligned_at_events(hit_table, chunk_size=chunk_size):
 
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. This data cannot be used like this!')
 
                if fix_trigger_number is True:
                    selection = np.logical_or((hits['trigger_status'] & 0b00000001) == 0b00000001,
                                              (hits['event_status'] & 0b0000000000000010) == 0b0000000000000010)
                    selected_te_hits = np.where(selection)[0]  # select both events with and without hit that have trigger error flag set
 
#                     assert selected_te_hits[0] > 0
                    tmp_trigger_number = hits['trigger_number'].astype(np.int32)
 
                    # save trigger and event number for plotting correlation between trigger number and event number
                    event_number, trigger_number = hits['event_number'].copy(), hits['trigger_number'].copy()
 
                    hits['trigger_number'][0] = 0
 
                    offset = (hits['trigger_number'][selected_te_hits] - hits['trigger_number'][selected_te_hits - 1] - hits['event_number'][selected_te_hits] + hits['event_number'][selected_te_hits - 1]).astype(np.int32)  # save jumps in trigger number
                    offset_tot = np.cumsum(offset)
 
                    offset_tot[offset_tot > 32768] = np.mod(offset_tot[offset_tot > 32768], 32768)
                    offset_tot[offset_tot < -32768] = np.mod(offset_tot[offset_tot < -32768], 32768)
 
                    for start_hit_index in range(len(selected_te_hits)):
                        start_hit = selected_te_hits[start_hit_index]
                        stop_hit = selected_te_hits[start_hit_index + 1] if start_hit_index < (len(selected_te_hits) - 1) else None
                        tmp_trigger_number[start_hit:stop_hit] -= offset_tot[start_hit_index]
 
                    tmp_trigger_number[tmp_trigger_number >= 32768] = np.mod(tmp_trigger_number[tmp_trigger_number >= 32768], 32768)
                    tmp_trigger_number[tmp_trigger_number < 0] = 32768 - np.mod(np.abs(tmp_trigger_number[tmp_trigger_number < 0]), 32768)
 
                    hits['trigger_number'] = tmp_trigger_number
 
                selected_hits = hits[(hits['event_status'] & 0b0000100000000000) == 0b0000000000000000]  # select not empty events
 
                if fix_event_number is True:
                    selector = (selected_hits['event_number'] != (np.divide(selected_hits['event_number'] + 1, 32768) * 32768 + selected_hits['trigger_number'] - 1))
                    n_fixed_hits += np.count_nonzero(selector)
                    selector = selected_hits['event_number'] > selected_hits['trigger_number']
                    selected_hits['event_number'] = np.divide(selected_hits['event_number'] + 1, 32768) * 32768 + selected_hits['trigger_number'] - 1
                    selected_hits['event_number'][selector] = np.divide(selected_hits['event_number'][selector] + 1, 32768) * 32768 + 32768 + selected_hits['trigger_number'][selector] - 1
 
#                 FIX FOR DIAMOND:
#                 selected_hits['event_number'] -= 1  # FIX FOR DIAMOND EVENT OFFSET
 
                hit_table_out.append(selected_hits)

        jumps = np.unique(np.array(jumps))
        logging.info('Corrected %d inconsistencies in the event number. %d hits corrected.' % (jumps[jumps != 0].shape[0], n_fixed_hits))

        if fix_trigger_number is True:
            return (output_file, event_number, trigger_number, hits['trigger_number'])


def format_hit_table(input_file, output_file, transpose):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    '''

    logging.info('Format hit table in %s', input_file)
    with tb.open_file(input_file, 'r') as in_file_h5:
        hits = in_file_h5.root.Hits[:]
        hits_formatted = np.zeros((hits.shape[0], ), dtype=[('event_number', np.int64), ('trigger_time_stamp',np.uint64),('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
        with tb.open_file(output_file, 'w') as out_file_h5:
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=hits_formatted.dtype, title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hits_formatted['event_number'] = hits['event_number']
            hits_formatted['trigger_time_stamp'] = hits['trigger_time_stamp']
            hits_formatted['frame'] = hits['relative_BCID']
            if transpose:
                hits_formatted['row'] = hits['column']
                hits_formatted['column'] = hits['row']
            else:
                hits_formatted['column'] = hits['column']
                hits_formatted['row'] = hits['row']
            hits_formatted['charge'] = hits['tot']
            if not np.all(np.diff(hits_formatted['event_number']) >= 0):
                raise RuntimeError('The event number does not always increase. This data cannot be used like this!')
            hit_table_out.append(hits_formatted)

def fix_time_stamp(hits_file_in, hits_file_out):
    ''' Fix overflow of time stamp.
     
    Assume that you do not miss a full event
    '''
    with tb.open_file(hits_file_in) as in_file:
        node = in_file.root.Hits
         
        hits = node[:]
        with tb.open_file(hits_file_out, 'w') as out_file:
            hits_out = out_file.create_table(out_file.root, name=node.name,
                                             description=node.dtype,
                                             title=node.title,
                                             filters=tb.Filters(
                                                 complib='blosc',
                                                 complevel=5,
                                                 fletcher32=False))
            fix_hits(hits)
            hits_out.append(hits)
    
    
@njit
def add_offset(hits, index, offset):
    for i in range(index, hits.shape[0]):
        hits[i]['trigger_time_stamp'] += offset
 
 
@njit
def fix_hits(hits):
    ''' Event number overflow at 2^15 fix '''
 
    old_hit = hits[0]
    for i, hit in enumerate(hits):
        if hit['trigger_time_stamp'] < old_hit['trigger_time_stamp']:
            add_offset(hits,
                       index=i,
                       offset=2**15)
        old_hit = hit


if __name__ == "__main__":

    # Input raw data file names
#     raw_data_files = [r'C:\Users\DavidLP\Desktop\TB\RUN_20\Raw_Data\138_proto_9_ext_trigger_scan.h5',  # the first DUT is the master reference DUT
#                       r'C:\Users\DavidLP\Desktop\TB\RUN_20\Raw_Data\64_lfcmos_3_ext_trigger_scan.h5',
#                       r'C:\Users\DavidLP\Desktop\TB\RUN_20\Raw_Data\102_scc_167_ext_trigger_scan.h5'
#                       ]

    '''
    long rung, 5M trigger, for alignment
    '''
#     raw_data_files = [r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_10/set_trigger_delay/module_2/75_module_2_ext_trigger_scan.h5',  # the first DUT is the master reference DUT
#                       r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_10/set_trigger_delay/module_4/75_module_4_ext_trigger_scan.h5',
#                       r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_11/set_trigger_delay/module_0/60_module_0_ext_trigger_scan.h5',
#                       r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_11/set_trigger_delay/module_2/60_module_2_ext_trigger_scan.h5']
    '''
    test run
    '''
#     raw_data_files1 = [r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan.h5',
#                       r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/board_10/set_trigger_delay/module_1/67_module_1_ext_trigger_scan.h5',
#                       ]
# 
#     for i, raw_data_file in enumerate(raw_data_files1):
#         transpose = False
#         process_dut(raw_data_file, trigger_data_format=2)
    print 'finished first plane'
    
    raw_data_files = ['/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_2_ext_trigger_scan.h5',
                      '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/72_module_4_ext_trigger_scan.h5',
                      '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_0_ext_trigger_scan.h5',
                      '/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/PIX4/57_module_2_ext_trigger_scan.h5'
                        ]
    
    for i, raw_data_file in enumerate(raw_data_files):
        if i&1 == 0:
            process_dut(raw_data_file, trigger_data_format=2,transpose=False)
        else :
            process_dut(raw_data_file, trigger_data_format=2,transpose=True)

    print 'finished telescope'
    
#     # Do separate DUT data processing in parallel. The output is a formatted hit table.
#     pool = Pool()
#     results = pool.map(process_dut, raw_data_files)
#     pool.close()
#     pool.join()

#     process_dut(raw_data_files[0])
