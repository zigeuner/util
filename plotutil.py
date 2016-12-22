"""
Some utility plot functions.

FIXME ****: add the option of composition; this would greatly increase flexibility
def plot(x, y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        plot_fig = True
    else:
        plot_fig = False
    ax.plot(x, y, **kwargs)
    if plot_fig:
        plt.savefig()
        plt.show()
        plt.close()

def hist(x, ax=None, **kwargs):
    pass

fig = plt.figure()
ax1 = fig.add_subplot(211)
plot(x, y, ax1)
ax2 = fig.add_subplot(212)
hist(x, ax2)
plt.savefig()
plt.show()
plt.close()

http://stackoverflow.com/questions/22210074/using-passed-axis-objects-in-a-matplotlib-pyplot-figure
"""

from __future__ import division
import itertools 

#import colorsys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm

from matplotlib.table import Table

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import proj3d

from matplotlib.mlab import PCA

import seaborn
# Importing seaborn is enough to set the style
#seaborn.set()


# http://matplotlib.org/users/colormaps.html
cmapnames = cm.cmap_d.keys()  # cmapids? change 'colorscheme' to 'cmapid'?


def get_axs(nrow, ncol, axslices=None, subplots_adjust=None, aspect=None, 
            **kwargs_fig):
    """
    http://matplotlib.org/users/gridspec.html
    
    Input:
        nrow and ncal: eg, Gridspec(nrow=4, ncol=4)
        axslices: eg, [(0,slice(3)), (1,slice(3)), (2,slice(3)), (3,0), (slice(4),3)]
    """
    plt.figure(**kwargs_fig)
    
    axs = []
    
    gs = gridspec.GridSpec(nrow, ncol)
    if subplots_adjust is not None:
        gs.update(**subplots_adjust)
    if axslices is None:
        axslices = list(itertools.product(range(nrow), range(ncol)))
    for axslice in axslices:
        rowslice, colslice = axslice
        if aspect is None:
            ax = plt.subplot(gs[rowslice, colslice])
        else:
            ax = plt.subplot(gs[rowslice, colslice], aspect=aspect)
        axs.append(ax)
    
    return axs


def get_ax3d(figsize=None, equal_aspect=True, **kwargs):
    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.gca(projection='3d')
    if equal_aspect:
        ax.set_aspect("equal")
    return ax
    

def get_colors(ncolor, cmapname=None, scheme=None):
    """
    Input:
        ncolor:
        scheme: DEPRECATION WARNING 'standard' or colormap name, eg, 'jet'
            (http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html)
        cmapname: a new name for scheme
    """
    if scheme is not None:
        cmapname = scheme
    
    if cmapname in [None, 'standard']:  # 'standard'
        cs0 = ['b','g','r','c','m','y','k']
        cs = cs0 * int(np.ceil(ncolor/7))
        return cs[:ncolor]
    else:
        cmap = cm.ScalarMappable(mpl.colors.Normalize(1, ncolor), 
                                 getattr(cm, cmapname))
        cs = cmap.to_rgba(range(1, ncolor+1))
        return cs


def pca0(ys, k=3, recenter=False, ):
    """
    https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml
    
    Input: 
    """
    out = PCA(np.array(ys))
    return out.Y[:, :k]


def pca(mat, k=3):
    """
    Usually / assumes sample size exceeds feature space dimension (so dense
    sampling would break this); otherwise one needs to use svd, because 
    U is needed. 
    
    Input:
        mat: m by n; covariance matrix: n by n  
        
    returns: data transformed in 2 dims/columns + regenerated original data
    
    """
    # mean center the data
    mat = np.array(mat)
    mat -= mat.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(mat, rowvar=False)
    
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    eigvals, eigvecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idxs = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,idxs]
    # sort eigenvectors according to same index
    eigvals = eigvals[idxs]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigvecs = eigvecs[:, :k]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(mat, eigvecs), eigvals, eigvecs
    

def scatter3d(xs, ys, zs, cs=None, figsize=None, 
                  xyzlabels=None, xyzlims=None, 
                  title='', filepath='', show=True, equal_axis=True, 
                  log10=False, subplots_adjust=None,
                  **kwargs_scatter):
    """
    Input:
        figsize:
        xyzlabels:
        xyzlims:
        title:
        filepath:
        show:
        equal_axis:
        log10:
        adjust:
        kwargs_scatter:
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if cs is not None:
        for x, y, z, c in zip(xs, ys, zs, cs):  # this seems to slow things down a lot 
            if not isinstance(c, str):
                c = cm.jet(c)
            ax.scatter(x, y, z, color=c, **kwargs_scatter)
    else:
        ax.scatter(xs, ys, zs, **kwargs_scatter)

    if xyzlabels is not None:
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
        
    if log10:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if xyzlims is not None:
        ax.set_xlim(xyzlims[0])
        ax.set_ylim(xyzlims[1])
        ax.set_zlim(xyzlims[2])
        
    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)    
    
    if title:
        plt.title(title, fontsize=10)
    
    if filepath:
        plt.savefig(filepath)
    
    if show:
        plt.show()
    
    plt.close()



def scatterplot(xs, ys, figsize=None, xylabels=None, xylims=None, 
                xyticks=None, xyticklabels=None,
                title='', filepath='', show=True, equal_axis=True, 
                plot_equal=False, minmax=None, log10=False, subplots_adjust=None,
                equal_scale=False, ax=None,
                **kwargs_scatter):
    """
    Input:
        equal_scale: x and y axes of equal scale (not forcing the aspect ratio)
        kwargs_scatter: kwargs of plt.scatter, whose docstring is attached 
            below for convenience.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plot_fig = True
    else:
        plot_fig = False
    
        
    if equal_scale:
        ax.set_aspect(1)
        
    ax.scatter(xs, ys, **kwargs_scatter)

    if xylabels is not None:
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        
    if xyticks is not None:
        if xyticks[0] is not None:
            ax.set_xticks(xyticks[0])
        if xyticks[1] is not None:
            ax.set_yticks(xyticks[1])
    
    if xyticklabels is not None:
        if xyticklabels[0] is not None:
            ax.set_xticklabels(xyticklabels[0])
        if xyticklabels[1] is not None:
            ax.set_yticklabels(xyticklabels[1], rotation=0)
        
    if log10:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    """
    if xlim is None:
        if log10:
            xmin, xmax = np.log10(min(xs)), np.log10(max(xs))
        else:
            xmin, xmax = min(xs), max(xs)
        xdiff = xmax - xmin
        xlim = (xmin - xdiff / 10, xmax + xdiff / 10) 
    if ylim is None:
        if log10:
            ymin, ymax = np.log10(min(ys)), np.log10(max(ys))
        else:
            ymin, ymax = min(ys), max(ys)
        ydiff = ymax - ymin
        ylim = (ymin - ydiff / 10, ymax + ydiff / 10)
    """
    if xylims is not None:
        ax.set_xlim(xylims[0])
        ax.set_ylim(xylims[1])
    
    if plot_equal:
        min_, max_ = minmax
        ax.plot([min_, max_], [min_, max_])
    
    if plot_fig:
        if subplots_adjust:
            plt.subplots_adjust(**subplots_adjust)    
        if title:
            plt.title(title, fontsize=10)
        if filepath:
            plt.savefig(filepath)
        if show:
            plt.show()
    
        plt.close()
scatterplot.__doc__ += plt.scatter.__doc__


def plot_vectorfield(xs, vs, cs=None, figsize=None, 
                     xylabels=None, xylims=None, xyticklabels=None,
                     title='', filepath='', show=True,
                     subplots_adjust=None, equal_scale=False,
                     **kwargs_quiver):
    """
    Input:
        xs: N-by-2 array
        vs: N-by-2 array
        cs: None or N-by-2 array
        equal_scale: x and y axes of equal scale (not forcing the aspect ratio)
        kwargs_scatter: kwargs of plt.scatter, whose docstring is attached 
            below for convenience.
    
    matplotlib.pyplot.quiver's doc is attached below.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if equal_scale:
        ax.set_aspect(1)
    
    X, Y = np.array(xs).T
    U, V = np.array(vs).T
    C = cs
    ax.quiver(X, Y, U, V, C, **kwargs_quiver)
    #ax.scatter(xs[0], xs[1], **kwargs_scatter)
    

    if xylabels is not None:
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
    
    if xyticklabels is not None:
        if xyticklabels[0] is not None:
            ax.set_xticklabels(xyticklabels[0])
        if xyticklabels[1] is not None:
            ax.set_yticklabels(xyticklabels[1], rotation=0)
        
    if xylims is not None:
        if xylims[0] is not None:
            ax.set_xlim(xylims[0])
        if xylims[1] is not None:
            ax.set_ylim(xylims[1])
    
    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)    
    
    if title:
        plt.title(title, fontsize=10)
    
    if filepath:
        plt.savefig(filepath)
    
    if show:
        plt.show()
    
    plt.close()
plot_vectorfield.__doc__ += plt.quiver.__doc__


def plot_vectors3d(starts, vectors, scalefactors=None, 
                   color='b', colors=None, cmapname=None,
                   figsize=None,
                   ax=None, filepath='', show=True, **kwargs):
    """
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        plot_fig = True
    else:
        plot_fig = False
    
    nvec = len(starts)
    
    if scalefactors is None:
        scalefactors = [1] * nvec
    
    vectors = (np.array(vectors).T * np.array(scalefactors)).T
    ends = np.array(starts) + vectors
    
    xs, ys, zs = ends.T
    vxs, vys, vzs = vectors.T
    
    if cmapname is not None:
        colors = get_colors(len(xs), cmapname=cmapname)
    if colors is None and cmapname is None:
        colors = [color] * len(xs)
    #import ipdb
    #ipdb.set_trace()
    for x, y, z, vx, vy, vz, c in zip(xs, ys, zs, vxs, vys, vzs, colors):
        ax.quiver(x, y, z, vx, vy, vz, colors=c, length=np.linalg.norm([vx,vy,vz]),
                  **kwargs)
    
    if plot_fig:
        if filepath:
            plt.savefig(filepath)
        if show:    
            plt.show()
        plt.close()        

plot_vectors3d.__doc__ += Axes3D.quiver.__doc__

def plot_heatmap(mat=None, xs=None, ys=None, f=None, cmap='jet',
                 xyticklabels=None, xylabels=None, subplots_adjust=None,
                 figsize=None, show=True, filepath='', **kwargs):
    """
    Input:
        kwargs: kwargs of ax.matshow, typically include:
            'interpolation', 'origin'
    """
    if mat is None:
        xss, yss = np.meshgrid(xs, ys)
        xys = zip(xss.flatten(), yss.flatten())
        try:
            zs = map(f, xys)
        except TypeError:
            f2 = lambda xy: f(*xy) 
            zs = map(f2, xys)
        mat = np.reshape(zs, xss.shape)
        
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = ax.matshow(mat, cmap=getattr(cm, cmap), **kwargs)
    
    # Got the following code from here:
    # http://stackoverflow.com/questions/18195758/
    # set-matplotlib-colorbar-size-to-match-graph
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(heatmap) #, cax=cax)
    cbar.ax.tick_params(labelsize=8.5)
    
    if xylabels is not None:
        ax.set_xlabel(xylabels[0], fontsize=None)
        ax.set_ylabel(xylabels[1], rotation=0, fontsize=None)
        #ax.xaxis.set_label_coords(x=0.5, y=-0.1)
        
    if xyticklabels is None and hasattr(mat, 'rowvarids'):
        xyticklabels = [mat.colvarids, mat.rowvarids]
    if xyticklabels is not None:
        ax.xaxis.tick_bottom()
        ax.set_xticks(range(np.shape(mat)[1]))
        ax.set_yticks(range(np.shape(mat)[0]))
        ax.set_xticklabels(xyticklabels[0], rotation=0, fontsize=16)
        ax.set_yticklabels(xyticklabels[1], rotation=0, fontsize=None)
    
    if subplots_adjust is not None:
        plt.subplots_adjust(**subplots_adjust) 
    
    plt.savefig(filepath)
    
    if show:
        plt.show()
        
    plt.close()
plot_heatmap.__doc__ += plt.imshow.__doc__


def plot_table(mat):
    pass


def plot_surface(f, theta1s, theta2s,
                 cmap=None,
                
                 pts=None, cs_pt=None, 
                 
                 
                 xyzlabels=None, xyzlims=None, 
                 xyzticks=None, xyzticklabels=None,
                 
                 figsize=None, show=True,
                 filepath='', ax=None,
                 **kwargs_surface):
    """
    Input:
        fs: a function or a list of functions that takes in a len-2 array and
            outputs a scalar (in which case the graph is plotted) or
            outputs a len-3 array (in which case the image if plotted) 
        theta1s: a list of first parameter values
        theta2s: a list of second parameter values
        
        cmap: a function mapping from p to c, a scalar representing color
        cs_f: None or a list of colors; surface colors
        
        pts: a list of (y1, y2, y3)
        cs_pt: a list of c, a scalar representing color
        
        kwargs_surface: alpha, edgecolor, linewidth, antialiased
        
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        plot_fig = True
    else:
        plot_fig = False
        
    theta1ss, theta2ss = np.meshgrid(theta1s, theta2s)
    ps = zip(np.ravel(theta1ss), np.ravel(theta2ss))
    
            
    kwargs_surface.setdefault('cstride', 1)
    kwargs_surface.setdefault('rstride', 1)
    

    ys = map(f, ps)
    
    if np.array(ys).ndim == 1:  # plot the graph of f
        ys = zip([p[0] for p in ps], [p[1] for p in ps], ys)
    
    if np.shape(ys)[1] > 3:
        ys = pca(ys, 3)[0]
        
    yss = np.reshape(np.transpose(ys), (3,)+theta1ss.shape)

    if cmap is not None:
        cs = map(cmap, ps)
        css = np.reshape(np.transpose(cs), (len(cs[0]),)+theta1ss.shape)
        ax.plot_surface(*yss, facecolors=cm.jet(css),
                        #rstride=1, cstride=1,  # the two parameters control density of coordinate curves 
                        **kwargs_surface)
    else:
        ax.plot_surface(*yss,  
                        #rstride=1, cstride=1, 
                        **kwargs_surface)
                        #shade=False, linewidth=0, antialiased=False, 
                    
    if pts is not None:
        pts = np.array(pts)
        if cs_pt is None or cs_pt is []:
            cs_pt = ['r'] * len(pts)
        elif cs_pt == [] or isinstance(cs_pt[0], str) or\
            isinstance(cs_pt[0], tuple) or\
            isinstance(cs_pt[0], np.ndarray):
            pass
        else:
            cmap = cm.ScalarMappable(mpl.colors.Normalize(min(cs_pt), max(cs_pt)), 
                                     cm.jet)
            cs_pt = cmap.to_rgba(cs_pt)
        #cs = np.divide(cs, float(np.max(cs)))  # normalize
        for pt, c in zip(pts, cs_pt):
            ax.scatter(*pt, color=c, s=10)
        ax.scatter(*pts[0], color=cs_pt[0], s=30)
    
    if xyzlabels is not None:
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
        
    if xyzlims is not None:
        ax.set_xlim(xyzlims[0])
        ax.set_ylim(xyzlims[1])
        ax.set_zlim(xyzlims[2])
        
    if xyzticks is not None:
        if xyzticks[0] is not None:
            ax.set_xticks(xyzticks[0])
        if xyzticks[1] is not None:
            ax.set_yticks(xyzticks[1])
        if xyzticks[2] is not None:
            ax.set_zticks(xyzticks[2])
        
    if xyzticklabels is not None:
        if xyzticklabels[0] is not None:
            ax.set_xticklabels(xyzticklabels[0])
        if xyzticklabels[1] is not None:
            ax.set_yticklabels(xyzticklabels[1])
        if xyzticklabels[2] is not None:
            ax.set_zticklabels(xyzticklabels[2])
    
    if plot_fig:
        if filepath:
            plt.savefig(filepath)
        if show:    
            plt.show()
        plt.close()        
    
plot_surface.__doc__ += Axes3D.plot_surface.__doc__
        

def barplot_3d(xs, ys, f=None, zss=None, dx=1, dy=1,
               c='b',
                  
                 xyzlabels=None, xyzlims=None, 
                 xyzticks=None, xyzticklabels=None,
                 
                 figsize=None, show=True,
                 filepath='',
                 **kwargs_bar):
    """
    Input:
        xs:
        ys:
        f:
        zss: if given, then xs and ys should be xss, yss
        
        kwargs_bar:
        
    """
    if f is None:
        xss, yss = xs, ys
    else:
        xss, yss = np.meshgrid(xs, ys)
        xys = zip(np.ravel(xss), np.ravel(yss))
        zs = map(f, xys)
        zss = np.reshape(np.transpose(zs), xss.shape)
        
    xpos, ypos, heights = xss.flatten(), yss.flatten(), zss.flatten()
    nbar = len(xpos)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    
    ax.bar3d(xpos, ypos, np.zeros(nbar), dx=[dx]*nbar, dy=[dx]*nbar, dz=heights,
             color=c, **kwargs_bar)
    
    if xyzlabels is not None:        
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
        
    if xyzlims is not None:
        ax.set_xlim(xyzlims[0])
        ax.set_ylim(xyzlims[1])
        ax.set_zlim(xyzlims[2])
        
    if xyzticks is not None:
        if xyzticks[0] is not None:
            ax.set_xticks(xyzticks[0])
        if xyzticks[1] is not None:
            ax.set_yticks(xyzticks[1])
        if xyzticks[2] is not None:
            ax.set_zticks(xyzticks[2])
        
    if xyzticklabels is not None:
        if xyzticklabels[0] is not None:
            ax.set_xticklabels(xyzticklabels[0])
        if xyzticklabels[1] is not None:
            ax.set_yticklabels(xyzticklabels[1])
        if xyzticklabels[2] is not None:
            ax.set_zticklabels(xyzticklabels[2])
    
    if filepath:
        plt.savefig(filepath)
    if show:    
        plt.show()
    plt.close()        
    
barplot_3d.__doc__ += Axes3D.bar3d.__doc__


def barplot(lefts, heights, widths, colors=None, cmapname='',  # cmapid?
            edgecolors=None, linewidth=0,  # edgecolors='none'?
            ax=None, subplots_adjust=None, filepath='', show=True,
            xylabels=None, xylims=None, xyticks=None, xyticklabels=None,
            legends=None, legendloc=None, 
            figtitle='', title='',  # figtile==title; FIXME ***: consolidate
            **kwargs_bar):
    """
    Input:
        widths: can be a scalar
        ax: 
        kwargs_bar: kwargs for plt.bar
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_fig = True
    else:
        plot_fig = False
    
    if cmapname:
        colors = get_colors(len(heights), cmapname)
    
    ax.bar(left=lefts, height=heights, width=widths, color=colors, 
           edgecolor=edgecolors, linewidth=linewidth, **kwargs_bar)
    
    if xylims is not None:
        if xylims[0] is not None:
            ax.set_xlim(xylims[0])
        if xylims[0] is not None:
            ax.set_ylim(xylims[1])
    
    if xylabels is not None:
        if xylabels[0] is not None:
            ax.set_xlabel(xylabels[0], fontsize=14)
        if xylabels[1] is not None:
            ax.set_ylabel(xylabels[1], fontsize=12, rotation=0)
        
    if xyticks is not None:
        if xyticks[0] is not None:
            ax.set_xticks(xyticks[0])
        if xyticks[1] is not None:
            ax.set_yticks(xyticks[1])
        
    if xyticklabels is not None:
        if xyticklabels[0] is not None:
            ax.set_xticklabels(xyticklabels[0], fontsize=12)
        if xyticklabels[1] is not None:
            ax.set_yticklabels(xyticklabels[1], fontsize=12, rotation=0)
    
    if legends is not None:
        ax.legend(legends, loc=legendloc, fontsize=12)
    
    if figtitle or title:
        title = filter(None, [figtitle, title])[0]
        ax.set_title(title, fontsize=None)
        
    if plot_fig:
        if subplots_adjust is not None:
            plt.subplots_adjust(**subplots_adjust)
        if filepath:
            plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()
    
    

def boxplot():
    pass
    

def scatterplots_pairwise(df, show=True, filepath=''):
    """
    """
    pass


def scatterplots(df1, df2, shape, figsize=None,
                 equal_axis=True, plot_equal=False,  
                 show=True, filepath='', kwargs_subplots_adjust=None, **kwargs_scatter):
    """
    Input:
        shape: a tuple
    """
    assert df1.columns.tolist() == df2.columns.tolist(), "Columns are different."
    
    fig = plt.figure(figsize=figsize)
    
    for idx, colvarid in enumerate(df1.columns):
        ax = fig.add_subplot(shape[0], shape[1], idx+1, aspect=1)
        ax.scatter(df1[colvarid], df2[colvarid], **kwargs_scatter)
        ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        ax.set_title(colvarid, fontsize=10)
        
        if plot_equal:
            xlim = ax.get_xlim()
            ylim = ax.get_xlim()
            start = min(list(xlim) + list(ylim))
            end = max(list(xlim) + list(ylim))
            ax.plot([start, end], [start, end])
    
    if kwargs_subplots_adjust:
        plt.subplots_adjust(**kwargs_subplots_adjust)
        
    if show:
        plt.show()
        
    if filepath:
        plt.savefig(filepath)


def scatterplots3d(enss, colors, xyzlabels=None, xyzlims=None, filepath='', show=True):
    """
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    for ens, color in zip(enss, colors):
        ax.scatter(ens.iloc[:,0], ens.iloc[:,1], ens.iloc[:,2], 
                   s=5, color=color, alpha=0.2, edgecolor='none')

    if xyzlabels is not None:
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
    
    if xyzlims is not None:
        ax.set_xlim(xyzlims[0])
        ax.set_ylim(xyzlims[1])
        ax.set_zlim(xyzlims[2])

    plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()




'''
def _get_colors(num_colors):
    np.random.seed(0)
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
'''

def plot(xs=None, ys=None, funcs=None, log10x=False, log10y=False, 
         figsize=None, dpi=None, 
         colors=None, colorscheme='standard', 
         linestyles=None, linewidths=None, markers=None, edgecolor='none',
         xylims=None, xylabels=None, xyticks=None, xyticklabels=None,
         legends=None, legendloc=(1,0), ax=None, 
         subplots_adjust=None, figtitle=None, title=None,  # title == figtitle, refactoring...
         show=True, filepath='', **kwargs_ax):
    """
    Input:
        xs: a sequence, or a list of sequences
        ys: a sequence, or a list of sequences of the same size as xs
        funcs: a list of functions
        colors: a list
        styles: a list 
        legendloc: 'best', 'upper right', 'lower left', etc. 
    """
    if xs is None:
        assert ys is not None, "either xs or ys has to be provided."
        xs = np.arange(len(ys[0]))  # FIXME **?
        
    ## get xs and ys
    xs = np.asarray(xs)
    if ys is not None:
        ys = np.asarray(ys)
        
    if xs.ndim == 1:
        x = xs
        if ys is not None and ys.ndim > 1:
            xs = [x] * ys.shape[0]
        if funcs is not None:
            ys = [map(func, x) for func in funcs]
            xs = [x] * len(funcs)
        
    elif xs.ndim == 2:
        if funcs is not None:
            ys = [map(func, x) for x, func in zip(xs, funcs)]
    else:
        raise ValueError("xs has more than two dimensions: %s"%str(xs))
        
    n = len(xs)  # number of graphs to be plotted on the same axis
    
    ## get colors and styles
    # FIXME ***: make it like linestyle, etc. Also: linestyles, etc. can come in as arrays!
    if colors is None:
        colors = get_colors(n, colorscheme)
    
    if not isinstance(linestyles, list):
        if linestyles is None:
            linestyle = '-'
        else:
            linestyle = linestyles
        linestyles = [linestyle] * n
    
    if not isinstance(linewidths, list):
        if linewidths is None:
            linewidth = 2
        else:
            linewidth = linewidths
        linewidths = [linewidth] * n

    if not isinstance(markers, list):
        if markers is None:
            marker = ''
        else:
            marker = markers
        markers = [marker] * n
    
    ## start plotting
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        plot_fig = True
    else:
        plot_fig = False
    
    for x, y, c, s, w, m in zip(xs, ys, colors, linestyles, linewidths, markers):
        ax.plot(x, y, color=c, linestyle=s, linewidth=w, 
                marker=m, markerfacecolor=c, markeredgecolor='none')
        
    ax.set(**kwargs_ax)
    
    if log10x:
        ax.set_xscale('log')
    if log10y:
        ax.set_yscale('log')
    
    if xylims is not None:
        ax.set_xlim(xylims[0])
        ax.set_ylim(xylims[1])
    
    if xylabels is None:
        xylabels = [ax.get_xlabel(), ax.get_ylabel()]
    ax.set_xlabel(xylabels[0], fontsize=12)
    ax.set_ylabel(xylabels[1], fontsize=12, rotation=0)
    ax.yaxis.set_label_coords(-0.5, 0.5) 
        
    if xyticks is not None:
        if xyticks[0] is not None:
            ax.set_xticks(xyticks[0])
        if xyticks[1] is not None:
            ax.set_yticks(xyticks[1])
        
    if xyticklabels is not None:
        if xyticklabels[0] is not None:
            ax.set_xticklabels(xyticklabels[0], fontsize=8)
        if xyticklabels[1] is not None:
            ax.set_yticklabels(xyticklabels[1], fontsize=8, rotation=0)
    
    # again?!
    if xylims is not None:
        ax.set_xlim(xylims[0])
        ax.set_ylim(xylims[1])
            
    if legends is not None:
        ax.legend(legends, loc=legendloc, fontsize=12)
        
    if title is not None or figtitle is not None:
        title = filter(None, [title, figtitle])[0]
        ax.set_title(title, fontsize=None)
    
    if plot_fig:
        if subplots_adjust is not None:
            plt.subplots_adjust(**subplots_adjust)
        if filepath:
            plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()
                
            
def plot3d(xs, ys, zs, figsize=None,
           cs=None, colormap=True, s=20, 
           xyzlims=None, xyzlabels=None, title='', 
           show=True, filepath='', ax=None,
           **kwargs):
    """
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        plot_fig = True
    else:
        plot_fig = False
    
    if cs is not None:
        if colormap:
            cs = cm.jet(cs)
    else:
        cs = [kwargs.get('color', kwargs.get('c', 'b'))] * len(xs)  # defaults to blue
        
    ax.plot(xs, ys, zs, **kwargs)
    
    for x, y, z, c in zip(xs, ys, zs, cs):
        ax.scatter(x, y, z, color=c, edgecolor='none', s=s)
    
    if xyzlabels is not None:
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
    
    if xyzlims is not None:
        ax.set_xlim(xyzlims[0])
        ax.set_ylim(xyzlims[1])
        ax.set_zlim(xyzlims[2])
    
    if plot_fig:
        plt.title(title)
            
        plt.savefig(filepath)
        if show: 
            plt.show()
        plt.close()

    
def plot0(trajs_x=None, trajs_y=None, funcs=None, trajs_err=None, offset=0,
         figsize=(8,6), log10=False, fmts=None, 
         xlabel='', ylabel='', xticks=None, yticks=None, xticklabels=None, yticklabels=None, 
         xmin=None, xmax=None, ymin=None, ymax=None, ytickformat=None, 
         legends=None, legendloc='upper right',  
         figtitle='', filepath='', show=True):
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
    if show:
        plt.show()
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
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(heatmap) #, cax=cax)
    cbar.ax.tick_params(labelsize=8.5)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
        #ax.xaxis.set_label_coords(x=0.5, y=-0.1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
        
    if xticklabels is not None:
        ax.xaxis.tick_bottom()
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=xticksize)
    
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

def plot_mat_number(mat, fmt='{:.2f}', figsize=None, 
                    xticklabels=None, yticklabels=None, 
                    rlabelsize=5, clabelsize=5, figtitle='', filepath='', 
                    show=True):
    """
    """
    rlabels, clabels = yticklabels, xticklabels  # FIXME *
    
    fig = plt.figure(figsize=figsize)
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
    
    
def plot_mat_spectrum(mat, figsize=None, spectrum='singval', 
                      figtitle='', filepath='', subplots_adjust=None, 
                      show=True, ax=None, **kwargs):
    """
    This function plots the spectrum (defined as either singular values or
    eigenvalues of the matrix).
     
    Input:
        spectrum: 'singval' (define spectrum as the singular values) or 
                  'eigval' (as the eigenvalues)
        log10: to plot the spectrum on a log0 scale; disabled because always True
    """
    ## compute the spectrum
    if spectrum == 'singval':
        vals = np.linalg.svd(mat, compute_uv=False)
    if spectrum == 'eigval':
        vals = np.linalg.eig(mat)[0]
    
    log10vals = np.log10(vals)
        
    #import ipdb
    #ipdb.set_trace()
    ## plot the spectrum
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plot_fig = True
    else:
        plot_fig = False
        
    for log10val in log10vals:
        ax.plot([0, 1], [log10val, log10val], **kwargs)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([])
    
    
    yticks = np.arange(np.floor(log10vals.min()), np.ceil(log10vals.max())+1)
    ax.set_yticks(yticks)
    yticklabels = ['1e%d' % ytick for ytick in yticks]
    ax.set_yticklabels(yticklabels)
    
    if plot_fig:
        if subplots_adjust:
            plt.subplots_adjust(**subplots_adjust)
        plt.title(figtitle)
        plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()



def hist0(x, figsize=None, normed=True, histtype='stepfilled',
         xlabel=None, title=None,
         show=True, filepath='', **kwargs):
    """
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.hist(x, normed=normed, histtype=histtype, **kwargs)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title: 
        ax.set_title(title)
        
    if filepath:
        plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()



def hist(data, plottype='single', figsize=None, cdf=None, 
         nbin=10, normed=True, histtype='stepfilled', cs=None,
         
         xylims=None, xyticks=None, xyticklabels=None, xylabels=None, 
         pts=None, 
         legends=None, legendloc='best', title='', show=True, filepath='',
         subplots_adjust=None,
         **kwargs_hist):
    """
    Input:
        data: 
            a 1d array: a single histogram plot; plottype=='single'
            a list of m 1d arrays: a single plot of m histograms 
                juxatopsed together; plottype='juxtapose'
            a 3d array (m by n by l): m by n subplots of 1d arrays; 
                plottype='subplots'
        plottype: 'single', 'juxtapose', 'subplots'
        cdf: cumulative distribution function
    """
    fig = plt.figure(figsize=figsize)
    
    if plottype == 'single':
        data = [[data]]
        if cs is None:
            cs = [[cs]]
        if xylims is not None:
            xylims = [[xylims]]
        if xyticks is not None:
            xyticks = [[xyticks]]
        if xyticklabels is not None:
            xyticklabels = [[xyticklabels]]
        if xylabels is not None:
            xylabels = [[xylabels]]
        if pts is not None:
            pts = [[pts]]
        
    if plottype in ['single', 'subplots']:
        m, n = np.shape(data)[:2]
        if cs is None:
            cs = [['b'] * n] * m
    
        for k, (i,j) in enumerate(np.ndindex(m,n)):
            data_ij = data[i][j]
        
            ax = fig.add_subplot(m, n, k+1)
            
            ax.hist(data_ij, bins=nbin, normed=normed, histtype=histtype, 
                    #range=xylims[i][j][0],
                    color=cs[i][j], ec='none', **kwargs_hist)
            
            #if xylims is not None:
            #    xylims_ij = xylims[i][j]
            #    ax.set_xlim(xylims_ij[0])
            #    ax.set_ylim(xylims_ij[1])
            if xyticks is not None:
                xyticks_ij = xyticks[i][j]
                ax.set_xticks(xyticks_ij[0])
                ax.set_yticks(xyticks_ij[1])
            if xyticklabels is not None:
                xyticklabels_ij = xyticklabels[i][j]
                ax.set_xticklabels(xyticklabels_ij[0])
                ax.set_yticklabels(xyticklabels_ij[1])
            if xylabels is not None:
                xylabels_ij = xylabels[i][j]
                ax.set_xlabel(xylabels_ij[0])
                ax.set_ylabel(xylabels_ij[1])
            
            if pts is not None:
                pts_ij = pts[i][j]
                ax.plot(pts_ij, [0]*len(pts), 'or', ms=8)
            
            if legends is not None:
                ax.legend(legends, prop={'size': 12}, loc=legendloc)
        
    elif plottype == 'juxtapose':
        nhist = len(data)
        alpha = np.exp(-nhist/4)
        if cs is None:
            cs = get_colors(nhist)
        
        ax = fig.add_subplot(111)
        for data_i, c in zip(data, cs):
            ax.hist(data_i, bins=nbin, normed=normed, histtype=histtype, 
                    color=c, alpha=alpha, ec='none')
        if legends is not None:
            ax.legend(legends, prop={'size': 12}, loc=legendloc)
    
    else:
        raise ValueError("Unrecognized value of 'plottype': %s"%plottype)
    
    if subplots_adjust is not None:
        plt.subplots_adjust(**subplots_adjust)
    plt.title(title)
    plt.savefig(filepath)
    if show:
        plt.show()
    plt.close()
            
    
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
    
    
def contour(func, xs, ys, levels=None, figsize=None, ax=None, 
            subplots_adjust=None, filepath='', show=True, 
            **kwargs):
    """
    Input:
        func: z = func(x, y)
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plot_fig = True
    else:
        plot_fig = False
    
    xss, yss = np.meshgrid(xs, ys)
    xys = zip(xss.ravel(), yss.ravel())
    zs = [func(x,y) for x, y in xys]
    zss = np.reshape(zs, xss.shape)
    
    ax.contour(xss, yss, zss, levels, **kwargs)
    
    if plot_fig:
        if subplots_adjust is not None:
            plt.subplots_adjust(**subplots_adjust)
        if filepath:
            plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()
    