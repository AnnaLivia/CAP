# MATLAB routines for PC-SOS-SDP

This folder contains the MATLAB cutting-plane routines called by the C++
lower-bound implementation (`solve_cluster_cp` and
`solve_cluster_cp_inherit`). They are distributed from the public
[PC-SOS-SDP repository](https://github.com/antoniosudoso/pc-sos-sdp) without
modification.

To run CAP, download [SDPNAL+](https://blog.nus.edu.sg/mattohkc/softwares/sdpnalplus/)
and copy this `clustering_matlab/` folder into the SDPNAL+ root directory.
Set `SDP_SOLVER_FOLDER` in `../anticlustering_c++/config.txt` to that root.
The MATLAB engine automatically adds the directory and its subdirectories to
the MATLAB path.

The upstream PC-SOS-SDP code is released under the GNU General Public License
v3.0; its license text is included in [`LICENSE`](LICENSE).
