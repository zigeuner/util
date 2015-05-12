"""
Some utility plot functions.
"""

from __future__ import division
import colorsys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.table import Table
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _get_colors(num_colors):
    np.random.seed(0)
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot(trajs_x=None, trajs_y=None, funcs=None, trajs_err=None, offset=0,
         figsize=(8,6), log10=False, fmts=None, 
         xlabel='', ylabel='', xticks=None, yticks=None, xticklabels=None, yticklabels=None, 
         xmin=None, xmax=None, ymin=None, ymax=None, ytickformat=None, 
         legends=None, legendloc='upper right',  
         figtitle='', filepath=''):
    """
    Input:
        Ys: multiple ys, e.g., [y1s, y2s, y3s]
        
        offset: an offset for traj_x so that error bars do not overlap
    """
    ## get the number of trajs to plot
    if trajs_y is not None:
        nvar = len(trajs_y)
    else:
        nvar = len(funcs)

    ## make as many traj_x as nvar
    if trajs_x is None:
        trajs_x = [range(len(traj_y)) for traj_y in trajs_y]
    else:
        trajs_x = list(trajs_x)
        if not hasattr(trajs_x[0], '__iter__'):  # a single traj of x
            trajs_x = [trajs_x] * nvar
    
    ## get trajs_y 
    if funcs:
        trajs_y = [[funcs[i](x) for x in trajs_x[i]] for i in range(nvar)]
    #if log10:
    #    trajs_y = np.log10(trajs_y)
    
    ## plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    #if len(trajs_y) <= 7:
    colors = ax._get_lines.color_cycle  # the color cycle, an iterator
    #colors = iter(['b','g','r','c','m','y','k','#FFA500','#FFC0CB','#A52A2A'])  # orange pink brown
    #else:
    #    #colors = iter(_get_colors(len(trajs_y)))
    #    colors = ['#f10000', '#00ff00', '#009775', '#003594', '#4d0004', '#ffb728', 
    #              '#009500', '#e5ffff', '#7e43ff', '#817a00', '#002800', '#00ffff', 
    #              '#ffceff', '#ffff00', '#000900', '#00a1ff', '#e3177a']
    #    colors = iter(colors)
    for i in range(nvar):
        if fmts:
            fmt = fmts[i]
        else:
            fmt = colors.next()  #{'linestyle':'-', 'color':colors.next()}
        traj_x = np.array(trajs_x[i]) + offset * i
        try:
            ax.errorbar(traj_x, trajs_y[i], yerr=trajs_err[i], fmt=fmt, lw=2, mec='none')
        except (TypeError, IndexError):  # either no trajs_err or run out
            ax.plot(traj_x, trajs_y[i], fmt, lw=2, mec='none')
        if ytickformat == 'sci':
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            ax.ticklabel_format(style='sci', axis='y')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
        
    if xticklabels is not None:
        ax.set_xticks(trajs_x[0])
        ax.set_xticklabels(xticklabels, fontsize=10, rotation='vertical')
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=16)
    
    if xmin is not None:  ax.set_xlim(left=xmin)
    if xmax is not None:  ax.set_xlim(right=xmax)
    if ymin is not None:  ax.set_ylim(bottom=ymin)
    if ymax is not None:  ax.set_ylim(top=ymax)     
    #if log10:
    #    yticks = ax.get_yticks()
    #    ax.set_yticklabels(10**yticks)
    if legends is not None:
        if hasattr(legendloc, '__iter__'):
            ax.legend(legends, prop={'size': 12}, bbox_to_anchor=legendloc)
        else:
            ax.legend(legends, prop={'size': 12}, loc=legendloc)
            
    plt.subplots_adjust(bottom=0.17, left=0.2)
    plt.title(figtitle)
    plt.savefig(filepath)
    plt.close()


def plot_mat_heatmap(mat, figsize=None, xlabel=None, ylabel=None, 
                     xticklabels=None, yticklabels=None, origin='lower', 
                     interpolation='nearest', xticksize=5, yticksize=5,
                     figtitle='', filepath='', **kwargs):
    """
    
    Input:
        mat: the matrix of data to be plotted
        xlabel: column variable (x)
        ylabel: row variable (y)
        xticklabels: column tick labels (xs) 
        yticklabels: row tick labels (ys)
        
        origin: y axis going up or down (default going up, which needs flipping
                the matrix and rlabels)
        
    
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = ax.matshow(mat, interpolation=interpolation, origin=origin, 
                         **kwargs)
    
    # Got the following code from here:
    # http://stackoverflow.com/questions/18195758/
    # set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(heatmap, cax=cax)
    cbar.ax.tick_params(labelsize=8.5)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
        #ax.xaxis.set_label_coords(x=0.5, y=-0.1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
        
    if xticklabels is not None:
        ax.xaxis.tick_bottom()
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(xticklabels, rotation='vertical', fontsize=xticksize)
    
    if yticklabels is not None:
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(yticklabels, fontsize=yticksize)

    #ax.plot([-0.5, mat.shape[1]-0.5], [-0.5, mat.shape[1]-0.5], 'r', lw=2)
    
    plt.subplots_adjust(bottom=0.15, left=0.15, top=None) #, top=0.95, left=0.05, right=0.95)
    plt.savefig(filepath)
    plt.close()


def plot_4d_heatmap(df, origin='lower', interpolation='nearest', filepath='',
                    minorlabelsize=None, majorlabelsize=None, **kwargs):
    """
    
    Input:
        df: pandas dataframe, both row and column double indexed 
            (hence 4d in total) 
    """
    nrow = len(df.index.levels[0])
    ncol = len(df.columns.levels[0])
    xmajorlevels = df.columns.levels[0]
    if origin == 'lower': 
        ymajorlevels = df.index.levels[0][::-1]
    else: 
        ymajorlevels = df.index.levels[0]
    mals = majorlabelsize 
    mils = minorlabelsize 
    
    minval = df.min().min()
    maxval = df.max().max()
         
    fig = plt.figure()
    for i in range(nrow):
        for j in range(ncol):
            dat_i = df.xs(ymajorlevels[i], level=0)
            dat_i_j = dat_i.xs(xmajorlevels[j], level=0, axis=1)

            ax = fig.add_subplot(nrow, ncol, i*ncol+j+1)
            heatmap = ax.matshow(dat_i_j, interpolation=interpolation, 
                                 origin=origin, vmin=minval, vmax=maxval,
                                 **kwargs)

            ax.xaxis.set_ticks(range(dat_i_j.shape[1]))
            ax.xaxis.set_ticklabels(dat_i_j.columns)
            ax.yaxis.set_ticks(range(dat_i_j.shape[0]))
            ax.yaxis.set_ticklabels(['%.2f'%num for num in dat_i_j.index])
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False) 
            
            if i == 0:
                ax.xaxis.set_ticks_position('top')
                plt.setp(ax.get_xticklabels(), visible=True, rotation='vertical')
                ax.set_xlabel('%.e'%xmajorlevels[j], fontsize=mals)
                ax.xaxis.set_label_position('top')
            if i == nrow-1:
                ax.xaxis.set_ticks_position('bottom')
                plt.setp(ax.get_xticklabels(), visible=True, rotation='vertical')
                
                ax.set_xlabel('%.e'%xmajorlevels[j], fontsize=mals)
                ax.xaxis.set_label_position('bottom')
            if j == 0:
                ax.yaxis.set_ticks_position('left')
                plt.setp(ax.get_yticklabels(), visible=True)
                ax.set_ylabel('%.2f'%ymajorlevels[i], fontsize=mals)
                ax.yaxis.set_label_position('left')
            if j == ncol-1:
                ax.yaxis.set_ticks_position('right')
                plt.setp(ax.get_yticklabels(), visible=True)
                ax.set_ylabel('%.1f'%ymajorlevels[i], fontsize=mals)
                ax.yaxis.set_label_position('right')
                
            plt.setp(ax.get_xticklabels(), fontsize=mils)
            plt.setp(ax.get_yticklabels(), fontsize=mils)
            
            ax.xaxis.set_ticks_position('none') 
            ax.yaxis.set_ticks_position('none')
    
    plt.suptitle(df.columns.names[0], position=(0.5, 0.9))
    plt.suptitle(df.columns.names[0], position=(0.5, 0.02))
    plt.suptitle(df.index.names[0], position=(0.1, 0.5))
    plt.suptitle(df.index.names[0], position=(0.85, 0.5))
    
    
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.1, right=0.8,
                        bottom=0.15, top=0.85)
    cbar = fig.add_axes([0.88, 0.2, 0.02, 0.5])
    
    plt.colorbar(heatmap, cax=cbar)
    plt.savefig(filepath)
    plt.close()
    

"""
def plot_heatmap(self, log10=False, figtitle='', filepath=''):

        mat = self
        if log10:
            mat = np.log10(np.abs(self))
        fig = plt.figure(figsize=(6.5, 6), dpi=300)
        ax = fig.add_subplot(111)
        # http://stackoverflow.com/questions/7875688/how-can-i-create-a-standard
        # -colorbar-for-a-series-of-plots-in-python: 
        # "I prefer using matshow() or pcolor() because imshow() smoothens 
        # the matrix when displayed making interpretation harder. So unless 
        # the matrix is indeed an image, I suggest that you try the other two."
        heat = ax.matshow(mat)
        ax.tick_params(labelright=True, labelbottom=True)
        ax.set_xticks(np.arange(0, len(self.colvarids)))
        ax.set_yticks(np.arange(0, len(self.rowvarids)))
        ax.set_xticklabels(self.colvarids, rotation='vertical', fontsize=10)
        ax.set_yticklabels(self.rowvarids, fontsize=10)
        ax.set_title(figtitle, position=(0.5, 1.25))
        bar = fig.add_axes([0.85, 0.2, 0.02, 0.5])
        fig.colorbar(heat, cax=bar)
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.02)
        plt.savefig(filepath, dpi=300)
        plt.close()
"""

def plot_mat_number(mat, fmt='{:.2f}', rlabels=None, clabels=None, 
                    rlabelsize=5, clabelsize=5, figtitle='', filepath=''):
    """
    """
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])
    
    nrow, ncol = mat.shape
    width, height = 1/ncol, 1/nrow

    # add cells
    for (i,j), val in np.ndenumerate(mat):
        tb.add_cell(i, j, width, height, text=fmt.format(val), loc='center')

    # row labels
    for i, rlabel in enumerate(rlabels):
        tb.add_cell(i, -1, width, height, text=rlabel, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, clabel in enumerate(clabels):
        tb.add_cell(-1, j, width, height/2, text=clabel, loc='center', 
                     edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    
def plot_mat_spectrum(mat, spectrum='sing', log10=True, figtitle='', filepath=''):
    """
    This function plots the spectrum (defined as either singular values or
    eigenvalues of the matrix).
     
    Input:
        spectrum: 'sing' (define spectrum as the singular values) or 
                  'eig' (as the eigenvalues)
        log10: to plot the spectrum on a log0 scale 
    """
    ## compute the spectrum
    if spectrum == 'sing':
        vals = np.linalg.svd(mat)[1]
    if spectrum == 'eig':
        vals = np.linalg.eig(mat)[0]
    if log10:
        vals = np.log10(vals)
    ## plot the spectrum
    fig = plt.figure(figsize=(2.5, 8), dpi=300)
    ax = fig.add_subplot(111)
    for val in vals:
        ax.plot([0, 1], [val, val], 'k', linewidth=1)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([])
    yticks = ax.get_yticks()
    yticklabels = ['1e%d' % ytick for ytick in yticks]
    ax.set_yticklabels(yticklabels)
    plt.subplots_adjust(left=0.25, bottom=0.05, top=0.95)
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_hist(data, figsize=(8,6), cdf=None, nbin=20, normed=True, histtype='stepfilled',
              xmin=None, xmax=None, ymax=None, xlabel='', ylabel='',
              xticks=None, yticks=None, pts_mark=None, 
              legends=None, legendloc='upper right', figtitle='', filepath=''):
    """
    Input:
        cdf: cumulative distribution function
    """
    data = np.array(data)
    
    if data.ndim == 1 or data.shape[0] == 1:  # only one set of data
        data = data.reshape(1, np.prod(data.shape))
        alpha = 1
    else:  # multiple sets of data
        alpha = np.exp(-len(data)/4)
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    colors = ax._get_lines.color_cycle  # color cycle, an iterator
    for data_var in data:
        counts, bins, patches = ax.hist(data_var, bins=nbin, normed=normed, 
                                        histtype=histtype, color=colors.next(),
                                        alpha=alpha, ec='none')
    if legends is not None:
        ax.legend(legends, prop={'size': 12}, loc=legendloc)
    
    """
    if cdf:
        intervals = np.array([bins[:-1], bins[1:]]).T 
        probs = []
        for a, b in intervals:
            probs.append(cdf(b) - cdf(a))
        if normed:
            ys = probs / (intervals[:,1] - intervals[:,0])
        else:
            ys = len(nums) * np.array(probs)
        xs = np.mean(intervals, axis=1)
        ax.plot(xs, ys, 'or')
    """ 
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    if xmin is not None:  ax.set_xlim(left=xmin)
    if xmax is not None:  ax.set_xlim(right=xmax)
    if ymax is not None:  ax.set_ylim(top=ymax)
    #ax.xaxis.set_tick_params(labelsize=8)
    #ax.yaxis.set_tick_params(labelsize=8)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if pts_mark is not None:
        ax.plot(pts_mark, [0]*len(pts_mark), 'or', ms=8)
    plt.subplots_adjust(bottom=0.2)
    plt.title(figtitle)
    plt.savefig(filepath)
    plt.close()
    

def plot_ecdf(vals, xlabel='', figtitle='', filepath=''):
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(vals)
    xs = sorted(vals)
    ys = ecdf(xs)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.step(xs, ys)
    ax.set_xlabel(xlabel)
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
