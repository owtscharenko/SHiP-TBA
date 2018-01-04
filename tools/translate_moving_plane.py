''' All functions acting on the hits of one DUT are listed here'''
import zlib

import tables as tb
import numpy as np
from tqdm import tqdm
from numba import njit
from pixel_clusterizer import clusterizer

from testbeam_analysis import hit_analysis


@njit
def fix_time_stamps(hits, offset):
    old_hit = hits[0]
    for i, hit in enumerate(hits):
        if hit['trigger_time_stamp'] < old_hit['trigger_time_stamp']:
            add_offset(hits,
                       index=i,
                       offset=2**15 + offset)
        old_hit = hit


@njit
def add_offset(hits, index, offset):
    for i in range(index, hits.shape[0]):
        hits[i]['trigger_time_stamp'] += offset


def correct_time_stamp(input_hits_file, chunk_size=1000000):
    '''Generating pixel mask from the hit table.

    Parameters
    ----------
    input_hits_file : string
        File name of the hits table.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''

    with tb.open_file(input_hits_file) as in_file:
        with tb.open_file(input_hits_file[:-3] + '_new.h5', 'w') as out_file:
            hits_node = in_file.root.Hits
            dcr = hits_node.dtype.descr
            dcr[2] = (dcr[2][0], '<u8')

            table_out = out_file.create_table(out_file.root,
                                              description=np.dtype(dcr),
                                              title=hits_node.title,
                                              name=hits_node.name,
                                              filters=hits_node.filters)

            offset = 0

            for i in tqdm(range(0, hits_node.shape[0], chunk_size)):
                hits = hits_node[i:i + chunk_size]

                hits = hits.astype(dcr, copy=True)
                fix_time_stamps(hits, offset=offset)

                offset = hits[-1]['trigger_time_stamp']
                table_out.append(hits)
                
@njit
def translate_column(clusters, x_speed):
    offset = 10258
    for i, cluster in enumerate(clusters):
            cluster['mean_column'] = cluster['mean_column'] - x_speed * (cluster['trigger_time_stamp']-offset)*25e-9
        
        
        
def correct_for_movement(input_cluster_file, velocity, chunk_size=1000000):
    
    output_cluster_file = input_cluster_file[:-3] + '_translated.h5'
     
    with tb.open_file(input_cluster_file) as in_file:
        cluster_node = in_file.root.Cluster
        
        for i in tqdm(range(0, cluster_node.shape[0], chunk_size)):
            clusters = cluster_node[i:i+ chunk_size]
            
            
            
    return output_cluster_file
                
def cluster_hits_niko(hit_file):
    with tb.open_file(hit_file) as in_file:
        hits = in_file.root.Hits[:]

        # Initialize clusterizer object
        cls = clusterizer.HitClusterizer()

        cls.set_hit_fields({'relative_BCID': 'frame',
                                    'tot': 'charge'})

        cls.set_hit_dtype([('column', np.uint8),
                           ('tot', np.uint8)])

        # All cluster settings are listed here with their std. values

        # Main functions
        cluster_hits, clusters = cls.cluster_hits(hits)  # cluster hits
    

if __name__ == '__main__':
    input_hits_file=r'/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/board_10/set_trigger_delay/module_0/67_module_0_ext_trigger_scan_interpreted.h5'
    correct_time_stamp(input_hits_file)
#     with tb.open_file(input_hits_file) as in_file:
#         print in_file.root.Hits[:]['trigger_time_stamp']

#     cluster_hits_niko(input_hits_file)
    
    hit_analysis.cluster_hits_niko(input_hits_file=input_hits_file[:-3] + '_new.h5',
                                  min_hit_charge=0,
                                  max_hit_charge=13,
                                  column_cluster_distance=1,
                                  row_cluster_distance=2,
                                  frame_cluster_distance=2)
    
    correct_for_movement(input_hits_file[-3] + '_clustered.h5',velocity=(105.3,0))