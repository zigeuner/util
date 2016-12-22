"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import libens
reload(libens)


class ParameterEnsemble(libens.MatrixEnsemble):
    def plot_quantiles(self, ps=[0.025,0.25,0.5,0.75,0.975], normalize=False,
                       log10=False, initvals=None, sort_by_range=False, 
                       dpi=300, figwidth=10, rowheight=4, nvar_per_row=20,
                       ylabel=False, figtitle='', filepath=''):
        """
        """
        # get quantiles
        qs = self.get_quantiles(ps=ps, normalize=normalize, log=log10)
        
        # initialize some variables
        medians = np.array(np.median(self, axis=0)).flatten()
        colvarids = self.colvarids
        nvar = len(colvarids)
        if nvar < nvar_per_row:
            nvar_per_row = nvar
        nrow = np.int(np.ceil(nvar/ nvar_per_row))
        nvar_last_row = nvar % nvar_per_row
        idxs = np.arange(nvar)
        
        if sort_by_range:
            ranges = qs[:,:,-1] - qs[:,:,0]
            
        fig = plt.figure(figsize=(figwidth, rowheight*nrow), dpi=dpi)
        # http://matplotlib.org/users/gridspec.html
        # http://stackoverflow.com/questions/5083763/\
        # python-matplotlib-change-the-relative-size-of-a-subplot
        width_ratios = [nvar_last_row, nvar_per_row - nvar_last_row - 2]
        print width_ratios
        gs = gridspec.GridSpec(nrow, 2, width_ratios=width_ratios)
        
        for i in range(nrow):
            # initialize the indices of the variables in the row
            if len(idxs) >= nvar_per_row:
                idxs_row = idxs[:nvar_per_row]
                idxs = idxs[nvar_per_row:]  # shorten idxs
            else:
                idxs_row = idxs
            if i < nrow - 1:  # all but the last row
                ax = plt.subplot(gs[i, :])
            else:  # the last row
                ax = plt.subplot(gs[i, 0])
            
            # bar plot
            heights_row = qs[0, idxs_row, -1] -qs[0, idxs_row, 0]
            bottoms_row = qs[0, idxs_row, 0]
            ax.bar(np.arange(len(idxs_row))-0.4, heights_row, 
                   bottom=bottoms_row, width=0.8)
            
            # label initial values
            if initvals is not None:
                initvals_row = np.array(initvals)[idxs_row]
                if normalize:
                    medians_row = medians[idxs_row]
                    initvals_row = initvals_row / medians_row
                if log10:
                    initvals_row = np.log10(initvals_row)
                ax.plot(np.arange(len(idxs_row)), initvals_row, 'or', 
                        markersize=5)
            
            # label quantiles
            for j in range(len(idxs_row)):
                for k in range(len(ps)):
                    q_var = qs[0, i*nvar_per_row+j, k]
                    ax.plot([j-0.4, j+0.4], [q_var, q_var], '-r', linewidth=1)

            ax.set_xlim(-1, len(idxs_row))
            if normalize:
                ax.plot([-1, len(idxs_row)], [2,2], 'g--')
                ax.plot([-1, len(idxs_row)], [-2,-2], 'g--')
                ax.set_ylim(-3, 3)
            #ax.yaxis.set_label_position('left')
            #ax2 = ax.twinx()
            #ax2.yaxis.set_label_position('right')
            if ylabel:
                ylabel = str(ps) + ' quantiles'
                if normalize:
                    ylabel = ylabel + '\n normalized by medians'
                if log:
                    ylabel = '$log_{10}$(' + ylabel + ')'
                ax.set_ylabel(ylabel, fontsize=10)
            
            # yticklabels
            yticks = ax.get_yticks()
            yticklabels = ['%.4g' % np.power(10, ytick) for ytick in yticks]
            ax.set_yticklabels(yticklabels, fontsize=12)
            
            # xticklabels
            colvarids_row = np.array(colvarids)[idxs_row]
            xticklabels_row = colvarids_row
            if normalize:
                xticklabels_row = [colvarids_row[j] + '\n(' + '%.1E' %\
                    medians_row[j] + ')' for j in range(len(idxs_row))]
            plt.xticks(np.arange(len(idxs_row)), xticklabels_row, 
                       rotation='vertical', size=9)
            
        plt.subplots_adjust(top=0.975, bottom=0.1, left=0.1, right=0.95, 
                            hspace=0.35)
        plt.title(figtitle)
        plt.savefig(filepath, dpi=dpi)
        plt.close()