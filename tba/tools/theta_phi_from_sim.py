import numpy as np
import tables as tb

from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.result_analysis import histogram_track_angle
from testbeam_analysis.cpp import data_struct

import matplotlib.pyplot as plt

with tb.open_file('/home/niko/git/testbeam_analysis/testbeam_analysis/examples/simulation/Tracks_prealigned.h5') as in_file_h5:
    x = in_file_h5.root.Tracks_DUT_0[:]['slope_0']
    y = in_file_h5.root.Tracks_DUT_0[:]['slope_1']
    z = in_file_h5.root.Tracks_DUT_0[:]['slope_2']
    phi, theta, r = geometry_utils.cartesian_to_spherical(x, y, z)
    print np.allclose(r, 1)
     
#     phi = phi[np.logical_and(theta<0.001, theta>-0.001)]
#     plt.ylim(0,0.01)
#     plt.ylim(0,0.01)
    plt.xlabel('phi')
    plt.ylabel('theta')
    plt.plot(phi, theta, 'o', markersize=1, alpha=1)#, markerprop={'fillcolor': 'b'})
#     plt.hist(phi, range=(0, 2*np.pi), bins=100)
#     plt.hist(theta, range=(0, 0.05), bins=100)
#     plt.show()
    plt.savefig('/home/niko/git/testbeam_analysis/testbeam_analysis/examples/simulation/theta-phi-350GeV-without-material-50x50-zoom.png')
    
# histogram_track_angle(input_tracks_file='/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/matching_moving_module_0/pix2/output/Tracks_aligned.h5',
#                       input_alignment_file='/media/data/SHiP/SHiP-testbeam-September17/testbeam-analysis/tba_improvements_branch/matching_moving_module_0/5M_run/output/Alignment.h5')
