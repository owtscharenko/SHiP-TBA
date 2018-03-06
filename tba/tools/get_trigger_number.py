import numpy as np
import matplotlib.pyplot as plt

import tables as tb
from numba import njit, jit
from tba.tmp4 import plot_hw_vs_sw_timestamps


class HitInfoTable(tb.IsDescription):
    event_number = tb.Int64Col(pos=0)
    trigger_number = tb.UInt32Col(pos=1)
    trigger_time_stamp = tb.UInt64Col(pos=2)
    relative_BCID = tb.UInt8Col(pos=3)
    LVL1ID = tb.UInt16Col(pos=4)
    column = tb.UInt8Col(pos=5)
    row = tb.UInt16Col(pos=6)
    tot = tb.UInt8Col(pos=7)
    BCID = tb.UInt16Col(pos=8)
    TDC = tb.UInt16Col(pos=9)
    TDC_time_stamp = tb.UInt8Col(pos=10)
    trigger_status = tb.UInt8Col(pos=11)
    service_record = tb.UInt32Col(pos=12)
    event_status = tb.UInt16Col(pos=13)

def fix_time_stamp(hits_file_in, hits_file_out):
    ''' Fix overflow of time stamp.
     
    Assume that you do not miss a full event
    '''
    with tb.open_file(hits_file_in) as in_file:
        node = in_file.root.Hits
         
        hits = node[:]
        hits = hits.astype([('event_number', '<i8'), ('trigger_number', '<u4'), 
                            ('trigger_time_stamp', '<u8'), ('relative_BCID', 'u1'), 
                            ('LVL1ID', '<u2'), ('column', 'u1'), ('row', '<u2'), 
                            ('tot', 'u1'), ('BCID', '<u2'), ('TDC', '<u2'), 
                            ('TDC_time_stamp', 'u1'), ('trigger_status', 'u1'), 
                            ('service_record', '<u4'), ('event_status', '<u2')])
        
        with tb.open_file(hits_file_out, 'w') as out_file:
            hits_out = out_file.create_table(out_file.root, name=node.name,
                                             description=HitInfoTable,
                                             title=node.title,
                                             filters=tb.Filters(
                                                 complib='blosc',
                                                 complevel=5,
                                                 fletcher32=False))
            fix_hits(hits)
            hits_out.append(hits)

 
@njit
def fix_hits(hits):
    ''' Event number overflow at 2^15 fix '''
 
    old_trigger_time_stamp = hits[0]['trigger_time_stamp']
    
    n_overflows = 0
    
    for i, hit in enumerate(hits):
        if hit['trigger_time_stamp'] < old_trigger_time_stamp:
            old_trigger_time_stamp = hit['trigger_time_stamp']
            n_overflows += 1

        old_trigger_time_stamp = hit['trigger_time_stamp']
        hit['trigger_time_stamp'] += n_overflows * 2**15

        

def get_trigger_time_stamp(input_file):
    with tb.open_file(input_file) as file:
        raw_data = file.root.raw_data[:]
        output_file = input_file[:-3] + '_hw_ts_extracted.h5'
    
    combined = True
    # trigger = raw_data[raw_data & 0x80000000 == 0x80000000] & 0x0000FFFF # trigger number
    # print len(raw_data[raw_data & 0xFF000000 == 0x01000000])
    
    end = len(raw_data)
    raw_i = 0
    fe_hits = ([], np.uint32)
    trigger_number = ([], np.uint32)
    all_but_m26 = ([], np.uint32)
    triggers = []  # np.ndarray(shape=(1000, 1), dtype=np.uint32)
    
    class ClusterInfoTable(tb.IsDescription):
        event_number = tb.Int64Col(pos=0)
        trigger_time_stamp = tb.Int64Col(pos=1)
    
    with tb.open_file(output_file, mode="w") as out_file:
        descr = ClusterInfoTable
        dtypes = {'names': [ 'event_number', 'trigger_time_stamp'],
    			  'formats': ['int64', 'int64']}
    
        while raw_i < end:
            raw_d = raw_data[raw_i]
            if (raw_d & 0x80000000 == 0x80000000) | (raw_d & 0xFF000000 == 0x01000000):
                all_but_m26 = np.append(all_but_m26, np.array([raw_d], np.uint32))  # tlu + fei4
            if (raw_d & 0x80000000 == 0x80000000):
                trigger = raw_d & 0x0000FFFF  # trigger number
                ts = (raw_d & 0x7FFF0000) >> 16 if combined else (raw_d & 0x7FFFFFFF)
                triggers.append((trigger, ts))
                print trigger
            if(0xFF0000 & raw_d == 0x00EA0000) | (0xFF0000 & raw_d == 0x00EF0000) | (0xFF0000 & raw_d == 0x00EC0000) | (0xFF0000 & raw_d == 0x00E90000):  # only fei4, but exclude data records
                pass
            elif(raw_d & 0xFF000000 == 0x01000000):  # only data records (hits)
                fe_hits = np.append(fe_hits, np.array([raw_d], np.uint32))
                trigger_number = np.append(trigger_number, np.array([trigger], np.uint32))      
                # fe = raw_data[raw_data & 0xFF000000 == 0x01000000]
                raw_i += 1

        triggers = np.array(triggers)

        new_table = out_file.create_table(where=out_file.root,
    									name='Hits',
    									description=descr,
    									title='Cluster with timestamps',
    									filters=tb.Filters(complib='blosc',
    										   complevel=5,
    										   fletcher32=False))
    
        new_table.append(triggers)
        new_table.flush()
    print 'done'
    

scan_file = "/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/75_module_2_ext_trigger_scan.h5"
# 
# get_trigger_time_stamp(scan_file)
# 
# fix_time_stamp(scan_file[:-3] + "_hw_ts_extracted.h5", scan_file[:-3] + "_fixed_2.h5")

plot_hw_vs_sw_timestamps(scan_file[:-3] + "_interpreted_fixed.h5", scan_file[:-3] + "_interpreted.h5")

