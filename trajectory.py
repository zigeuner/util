"""
"""

import plotutil

import butil 
DF = butil.DF


class Trajectory(DF):
    """
    """
    
    @property
    def _constructor(self):
        return Trajectory
    
    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        """
        super(DF, self).__init__(data=data, index=index, columns=columns, 
                                 **kwargs)
        self.index.name = 'time'
    
    """
    def __add__(self, other): 
        if self.varids == other.varids:
            times = self.times + other.times  # list addition
            varids = self.varids
            values = np.vstack((self.values, other.values))
        elif self.times == other.times:
            times = self.times
            varids = self.varids + other.varids
            values = np.hstack((self.values, other.values))      
        else:
            raise ValueError("...")
        return Trajectory(dat=values, times=times, varids=varids)
    """

    @property
    def times(self):
        return self.index.tolist()

    
    @property
    def varids(self):
        return self.columns.tolist()
    
    
    def plot(self, plotvarids=None, ax=None, legends=None, **kwargs):
        """
        """
        if plotvarids is not None:
            values = self.loc[:, plotvarids].values.T
        else:
            values = self.values.T
        
        if legends == True:
            legends = self.varids  # FIXME *
            
        reload(plotutil)
        plotutil.plot(self.times, values, ax=ax, legends=legends, **kwargs)
    
        
    

        