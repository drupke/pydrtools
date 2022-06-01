'''
author: David S. N. Rupke
created: 2021dec02
modified:
notes:
    Copied from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
'''
import numpy as np
def find_nearest(array, value):
    '''Find the index of the array that is nearest the specified value.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
