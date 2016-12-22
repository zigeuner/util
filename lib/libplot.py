"""
Some utility plot functions.
"""

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np


def plot(xs, Ys=None, funcs=None, figsize=(8,6), log10=False, 
         varids=None, styles=None, xlabel='', ylabel='', 
         figtitle='', filepath=''):
    """
    Input:
        Ys: multiple ys, e.g., [y1s, y2s, y3s]
    """
    if funcs:
        Ys = [[func(x) for x in xs] for func in funcs]
    if log10:
        Ys = np.log10(Ys)
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    if styles:
        for ys, style in zip(Ys, styles):
            ax.plot(xs, ys, style)
    else:
        ax.plot(xs, np.transpose(Ys))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(1, 15)
    if log10:
        yticks = ax.get_yticks()
        ax.set_yticklabels(10**yticks)
    if varids:
        ax.legend(varids, prop={'size': 9})
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_mat_heatmap(mat, xticklabels=None, yticklabels=None, reverse_y=True,
                     xticklabelsize=5, yticklabelsize=5, xlabel=None, ylabel=None, 
                     xlabelsize=10, ylabelsize=10,
                     figsize=(10, 10), interpolation='nearest', fmt=None, 
                     figtitle='', filepath=''):
    """
    """
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    
    mat = np.transpose(mat)
    if reverse_y:
        mat = mat[::-1]
    cax = ax.matshow(mat, interpolation=interpolation)
    fig.colorbar(cax)
    
    if xticklabels is not None:
        ax.set_xticks(range(mat.shape[0]))
        ax.xaxis.set_ticks_position('bottom')
        if fmt:
            xticklabels = [fmt(xticklabel) for xticklabel in xticklabels]
        ax.set_xticklabels(xticklabels, rotation='vertical', 
                           size=xticklabelsize)
        
        
    if yticklabels is not None:
        if reverse_y:
            yticklabels = yticklabels[::-1]
        ax.set_yticks(range(mat.shape[1]))
        if fmt:
            yticklabels = [fmt(yticklabel) for yticklabel in yticklabels]
        ax.set_yticklabels(yticklabels, size=yticklabelsize)
    
    ax.set_xlabel(xlabel, size=xlabelsize)
    ax.set_ylabel(ylabel, size=ylabelsize)

    plt.suptitle(figtitle, size=20)    
    plt.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.95)
    plt.savefig(filepath, dpi=300)
    plt.close()


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


def plot_surface(mat, xs, ys, log10x=False, log10y=False, xlabel=None, ylabel=None, zlabel=None,
                 fmt=None, animation=True, step_angle=1, figtitle='', filepath=''):
    
    if log10x:
        xs = np.log10(xs)
    if log10y:
        ys = np.log10(ys)
        
    Xs, Ys = np.meshgrid(xs, ys)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xs, Ys, mat)
    
    ax.set_xticks(xs[::2])
    if log10x:
        xticklabels = np.power(10, xs[::2])
    else:
        xticklabels = xs[::2]
    xticklabels = [fmt(xticklabel) for xticklabel in xticklabels]
    ax.set_xticklabels(xticklabels, size=5)
    
    ax.set_yticks(ys[::2])
    if log10y:
        yticklabels = np.power(10, ys[::2])
    else:
        yticklabels = ys[::2]
    yticklabels = [fmt(yticklabel) for yticklabel in yticklabels]
    ax.set_yticklabels(yticklabels, size=5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(figtitle)
    
    if animation:
        for angle in np.arange(0, 360.1, step_angle):
            ax.azim = angle
            plt.savefig(filepath.rstrip('.png')+'%05.1f'%angle+'.png', dpi=300)
    else:
        plt.savefig(filepath, dpi=300)
        
    plt.close()