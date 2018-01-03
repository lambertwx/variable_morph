# variable_morph
A class for performing spatially-varying morphological operations on an image.  This is useful for surveillance camera scenes with perspective so that you can apply less erosion on rows at the top of the image since they are further away.

It allows you to perform operations using different structuring elements (a.k.a. neighborhoods) depending on the position of a pixel.  At present, it's designed to use different neighborhoods based on a pixel's row.  I.e., it can use a different neighborhood in different horizontal bands of the image.

Example of how to use:
    a = np.zeros((15,10), dtype='uint8')
    a[6:,2:7] = 1
    a[0:,3:6] = 1
    m = VariableMorpher()
    m.addBand(8, radius=1, shape='diamond')
    m.addBand(a.shape[0], radius=2, shape='square')
    m.setup(a.shape)
    result = m.binary_erosion(a.astype(bool))
    