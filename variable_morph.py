# -*- coding: utf-8 -*-
"""
This class encapsulates an object for performing spatially-varying morphological operations on an image. 
"""
# MIT License

# Copyright Lambert Wixson(c) 2018.  Downloadable from https://github.com/lambertwx/variable_morph

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import skimage.morphology as skmorph
from typing import Dict, NewType

ImageND = NewType('ImageND', np.ndarray)

#%%
class VariableMorpher(object):
    """
    This class encapsulates an object for performing spatially-varying morphological operations on an image.  
    This allows you to perform operations using different structuring elements (a.k.a. neighborhoods) depending 
    on the position of a pixel.  
    
    At present, it's designed to use different neighborhoods based on a pixel's row.  I.e., it can use a different 
    neighborhood in different horizontal bands of the image.  This is good for 'surveillance camera' scenes with 
    perspective so that you can apply less erosion on rows at the top of the image since they are usually 
    further away.
    """
    def __init__(self):
        # This maps a row to a structuring element.  This element will be used on All rows that are
        # strictly less than this row and >= the next lowest row in the dictionary.
        self.diBands = {}
        self.isSetup = False
    
    def addBand(self, maxrow : int, radius : int, shape : str):
        """ 
        Use this function to define a band in which a specific structuring element will be used. 
        It will be used on All rows that are strictly less than maxrow and >= the next smallest row in the dictionary.
        """
        assert maxrow >= 0
        assert radius >= 1
        assert shape in {'square', 'diamond'}
        
        self.isSetup = False
        self.diBands[maxrow] = {}
        self.diBands[maxrow]['shape'] = shape
        self.diBands[maxrow]['radius'] = radius
        if shape == 'square':
            self.diBands[maxrow]['selem'] = skmorph.square(radius * 2 + 1)
        else:
            self.diBands[maxrow]['selem'] = skmorph.diamond(radius)
 
       
    def setup(self, shape):
        """
        Allocates temporary buffers.  These will be re-used on subsequent calls to this object's binary_erosion() function.
        
        @param shape: The size of the images on which we'll be operating.
        """
        lastrow = 0
        for row in sorted(self.diBands.keys()):
            this = self.diBands[row]
            rowmin = lastrow
            rowmin = max(0, rowmin-this['radius'])
            rowmax = min(shape[0], row+this['radius'])
            this['buf'] = np.empty((rowmax-rowmin, shape[1]), dtype=bool)
            lastrow = row
            
        self.isSetup = True
        
    @staticmethod
    def erode_in_band(img : ImageND, rowstart : int, rowend : int, out : ImageND, radius : int , 
                      selem : ImageND, bandbuf: ImageND):
        """
        Applies binary erosion to the rows between rowmin and rowmax using the specified erosion mask.
        
        @param rowstart: the start row of this band (inclusive)
        
        @param rowend: the end row of this band (exclusive)
        """
        assert rowstart >= 0
        assert rowend <= img.shape[0]
        assert rowstart < rowend
        assert img.shape == out.shape
        assert img.dtype == bool
        assert radius >= 1
        
        # We'll be slicing the image and need to ensure that at the top and bottom borders we include the part that runs 
        # off the border.  Otherwise the morphology operations will treat those as being filled with zeros.
        rowmin = max(0, rowstart-radius)
        rowmax = min(img.shape[0], rowend+radius)
        
        # Extract a slice of the image.  Note this if I'm understanding things correctly, this is NOT a new image.
        inslice = img[rowmin:rowmax,:]
        
        skmorph.binary_erosion(inslice, selem, out=bandbuf )
        
        # Now copy the relevant portion out of bandbuf into out
        bufmin = 0 if rowmin == 0 else radius
        bufmax = bandbuf.shape[0] if rowmax == img.shape[0] else bandbuf.shape[0]-radius
        bandslice = bandbuf[bufmin:bufmax]
        assert (rowend-rowstart) == (bufmax-bufmin)
        
        outslice = out[rowstart:rowend,:]
        np.copyto(outslice, bandslice)
        

    def binary_erosion(self, img):
        """
        Does binary erosion, using the bands defined by previous calls to addBand().  Before calling this function, 
        you must call setup() to allocate buffers.
        """
        assert self.isSetup, "You must call setup() before you can call this function."
        
        out = np.empty(img.shape, dtype=bool)
        
        # Iterate through keys in in increasing order, i.e. working our way down in bands from the top of the image.
        lastrow = 0
                     
        for row in sorted(self.diBands.keys()):
            this = self.diBands[row]
            self.erode_in_band(img, lastrow, row, out, this['radius'], this['selem'], this['buf'])
            lastrow = row
        return out
    
#%% Regression tests
if False:
    a = np.zeros((15,10), dtype='uint8')
    a[6:,2:7] = 1
    a[0:,3:6] = 1
    m = VariableMorpher()
    m.addBand(8, radius=1, shape='diamond')
    m.addBand(a.shape[0], radius=2, shape='square')
    m.setup(a.shape)
    assert m.diBands[8]['buf'].shape == (9,10)
    assert m.diBands[15]['buf'].shape == (9,10)
    result = m.binary_erosion(a.astype(bool)).astype('uint8')
    intended = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 1
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 3
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
                         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # 6
                         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # 7
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 8
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 10
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 12
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 14
        dtype='uint8')
    assert np.alltrue(result == intended)
