''' All functions acting on the hits of one DUT are listed here'''
import numpy as np
import tables as tb

from testbeam_analysis import analysis_functions
from testbeam_analysis.cpp import data_struct


events = np.arange(10, dtype=np.int64)
cluster = np.zeros((events.shape[0],), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))


cluster = np.ascontiguousarray(cluster)
events = np.ascontiguousarray(events)

mapped_cluster = np.zeros((events.shape[0],), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
mapped_cluster['mean_column'] = np.nan
mapped_cluster['mean_row'] = np.nan
mapped_cluster['charge'] = np.nan
mapped_cluster = np.ascontiguousarray(mapped_cluster)

analysis_functions.map_cluster(events, cluster, mapped_cluster)

print 'OK'