import numpy as np
import struct
from math import isclose
import warnings

def read_flow_file(filename: str):
    """
    Adapted from the ``readFlowFile`` Matlab code produced by Deqing Sun.
    The following is their function doc string:

     readFlowFile read a flow file FILENAME into 2-band image IMG

       According to the c++ source code of Daniel Scharstein
       Contact: schar@middlebury.edu

       Author: Deqing Sun, Department of Computer Science, Brown University
       Contact: dqsun@cs.brown.edu
       $Date: 2007-10-31 16:45:40 (Wed, 31 Oct 2006) $

     Copyright 2007, Deqing Sun.

                             All Rights Reserved

     Permission to use, copy, modify, and distribute this software and its
     documentation for any purpose other than its incorporation into a
     commercial product is hereby granted without fee, provided that the
     above copyright notice appear in all copies and that both that
     copyright notice and this permission notice appear in supporting
     documentation, and that the name of the author and Brown University not be used in
     advertising or publicity pertaining to distribution of the software
     without specific, written prior permission.

     THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
     PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
     ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
     WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
     ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

    """
    TAG_FLOAT = 202021.25; # check for this when READING the file

    assert filename, 'Filename must not be empty'
    assert '.flo' in filename, 'Filename must include extension (.flo)'

    with open(filename, 'rb') as f:
        tag = struct.unpack('f', f.read(4))[0]
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]

        assert isclose(tag, TAG_FLOAT), \
                'wrong tag (possibly due to big-endian machine?)'
        assert 1 < width < 99999, 'illegal width %d'.format(width)
        assert 1 < height < 99999, 'illegal height %d'.format(height)

        nBands = 2

        # arrange into matrix form
        tmp = f.read()
        arr = np.zeros(len(tmp) // 4, dtype='float32')
        for i in range(arr.size):
            arr[i] = struct.unpack('f', tmp[(4*i):(4*(i+1))])[0]
        arr = arr.reshape((width * nBands, height), order='F')
        arr = arr.T
        indxr = np.arange(1, width+1) * nBands
        indxr -= 1
        im = np.stack((arr[:, indxr-1], arr[:, indxr]), axis=2)
    return im

 
def flow_to_color(flow, max_flow=None):
    """
    Adapted from the ``flowToColor`` Matlab code produced by Deqing Sun.
    The following is their function doc string:

      flowToColor(flow, maxFlow) flowToColor color codes flow field, normalize
      based on specified value,

      flowToColor(flow) flowToColor color codes flow field, normalize
      based on maximum flow present otherwise

       According to the c++ source code of Daniel Scharstein
       Contact: schar@middlebury.edu

       Author: Deqing Sun, Department of Computer Science, Brown University
       Contact: dqsun@cs.brown.edu
       $Date: 2007-10-31 18:33:30 (Wed, 31 Oct 2006) $

     Copyright 2007, Deqing Sun.

                             All Rights Reserved

     Permission to use, copy, modify, and distribute this software and its
     documentation for any purpose other than its incorporation into a
     commercial product is hereby granted without fee, provided that the
     above copyright notice appear in all copies and that both that
     copyright notice and this permission notice appear in supporting
     documentation, and that the name of the author and Brown University not be used in
     advertising or publicity pertaining to distribution of the software
     without specific, written prior permission.

     THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
     PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
     ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
     WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
     ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
    """
 
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    height, width, nBands = flow.shape

    assert nBands == 2, 'image must have two bands'

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    with warnings.catch_warnings():
        # Suppress expected RuntimeWarnings
        warnings.simplefilter("ignore")
        # fix unknown flow
        idxUnknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) >
                                                        UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, np.nanmax(u))
    minu = min(minu, np.nanmin(u))

    maxv = max(maxv, np.nanmax(v))
    minv = min(minv, np.nanmin(v))

    rad = (u**2+v**2)**0.5
    maxrad = max(maxrad, np.nanmax(rad))

    print('max flow: {:.4f} flow range: u = {:.3f} .. {:.3f}; v = {:.3f} .. '
          '{:.3f}'.format(maxrad, minu, maxu, minv, maxv))

    if max_flow is not None:
        if max_flow > 0:
            maxrad = max_flow


    u = u/(maxrad+(2**(-52)))
    v = v/(maxrad+(2**(-52)))

    # compute color

    img = compute_color(u, v)

    # unknown flow
    IDX = np.stack((idxUnknown, idxUnknown, idxUnknown), axis=2)
    img[IDX] = 0
    return img
    

def compute_color(u, v):
    """
    Adapted from the ``computeColor`` Matlab code produced by Deqing Sun.
    The following is their function doc string:
       computeColor color codes flow field U, V

       According to the c++ source code of Daniel Scharstein
       Contact: schar@middlebury.edu

       Author: Deqing Sun, Department of Computer Science, Brown University
       Contact: dqsun@cs.brown.edu
       $Date: 2007-10-31 21:20:30 (Wed, 31 Oct 2006) $

     Copyright 2007, Deqing Sun.

                             All Rights Reserved

     Permission to use, copy, modify, and distribute this software and its
     documentation for any purpose other than its incorporation into a
     commercial product is hereby granted without fee, provided that the
     above copyright notice appear in all copies and that both that
     copyright notice and this permission notice appear in supporting
     documentation, and that the name of the author and Brown University not be used in
     advertising or publicity pertaining to distribution of the software
     without specific, written prior permission.

     THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
     PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
     ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
     WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
     ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
    """

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = (u**2+v**2) ** 0.5

    a = np.arctan2(-v, -u) / np.pi

    fk = ((a+1) / 2) * (ncols-1)  # -1~1 maped to 0~ncols-1

    k0 = np.floor(fk).astype(np.uint32)                 # 1, 2, ..., ncols

    k1 = k0+1
    k1[k1==ncols] = 1

    f = fk - k0

    img = np.zeros((*u.shape, 3), dtype=np.uint8)

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx] * (1-col[idx])  # increase saturation with radius

        col[~idx] = col[~idx] * 0.75             # out of range

        img[:,:, i] = np.floor(255*col*(1-nanIdx)).astype(np.uint8)

    return img


def make_colorwheel():
    # color encoding scheme
    # adapted from the color circle idea described at
    # http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = 255*np.arange(RY) // RY
    col = col+RY

    #YG
    colorwheel[col:(col+YG), 0] = 255 - (255*np.arange(YG) // YG)
    colorwheel[col:(col+YG), 1] = 255
    col = col+YG

    #GC
    colorwheel[col:(col+GC), 1] = 255
    colorwheel[col:(col+GC), 2] = 255*np.arange(GC) // GC
    col = col+GC

    #CB
    colorwheel[col:(col+CB), 1] = 255 - (255*np.arange(CB) // CB)
    colorwheel[col:(col+CB), 2] = 255
    col = col+CB

    #BM
    colorwheel[col:(col+BM), 2] = 255
    colorwheel[col:(col+BM), 0] = 255*np.arange(BM) // BM
    col = col+BM

    #MR
    colorwheel[col:(col+MR), 2] = 255 - (255*np.arange(MR) // MR)
    colorwheel[col:(col+MR), 0] = 255

    return colorwheel