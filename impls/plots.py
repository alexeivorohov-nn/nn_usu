import torch
import numpy as np
import matplotlib.pyplot as plt

def decision_plot(model: torch.nn.Module,
                  add_data = None,
                  data_transform = None,
                  grid_n = (100, 100),
                  classes = None,
                  cmap = None,
                  alpha = 0.2,
                  x_bounds = None,
                  figsize = (8,8),
                  legend = True,
                  ax = None
                  ):
    
    show_on_finish = False

    # Extract labels from additional data
    data_lbl_dict = {}

    # Define bounds
    x1_min, x1_max = -1.5, 1.5
    x2_min, x2_max = -1.5, 1.5

    if len(add_data) > 0:

        x1_min = []
        x1_max = []
        x2_min = []
        x2_max = []

        for key, val in add_data.items():
            x1_min.append(min(val['X'][:, 0]))
            x1_max.append(max(val['X'][:, 0]))
            x2_min.append(min(val['X'][:, 1]))
            x2_max.append(max(val['X'][:, 1]))
            data_lbl_dict[key] = torch.argmax(val['y'], dim=1)

        x1_min = min(x1_min)
        x1_max = max(x1_max)
        x2_min = min(x2_min)
        x2_max = max(x2_max)

    # Generate grid using PyTorch
    x1_ = torch.linspace(x1_min, x1_max, grid_n[0])
    x2_ = torch.linspace(x2_min, x2_max, grid_n[1]) 
    x1, x2 = torch.meshgrid(x1_, x2_, indexing='ij')

    # Tensor, that can be passed on model input
    grid = torch.stack([x1.ravel(), x2.ravel()], dim=1)

    # Get predictions
    with torch.no_grad():
        Z = model(grid)
        Z_class = torch.argmax(Z, dim=1)

    # Reshape to grid dimensions
    Z_class = Z_class.reshape(x1.shape).numpy()
    N_classes = len(np.unique(Z_class))
    
    # Create custom colormap for classes
    if cmap == None:
        vals = np.linspace(0,1,N_classes)
        cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    
    levels = (np.arange(N_classes+1) - 0.5)

    # Create figure
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        show_on_finish = True

    # Plot decision regions with solid colors
    if data_transform != None:
        pass

    ax.contourf(x1.numpy(), x2.numpy(), Z_class, 
                levels=levels, 
                alpha=alpha, 
                cmap=cmap)
 
    # Plot test points using class indices 

    if classes == None:
        classes = [f'{i}' for i in range(N_classes)]

    for key, val in add_data.items():
        for k in range(N_classes):
            mask = (data_lbl_dict[key] == k)

            try:
                mark = val['m']
            except:
                mark = ','

            try:
                msize = val['msize']
            except:
                msize = 20.0

            lbl = key + '_' + classes[k]
            ax.scatter(val['X'][mask, 0], val['X'][mask, 1],
                        marker=mark, edgecolor='w', s=msize,
                        color=cmap(k), label=lbl)
            
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    ax.grid('on')
    ax.legend()

    if show_on_finish:
        plt.show()


def plot_2d(data = None,
            data_transform = None,
            classes = None,
            cmap = None,
            alpha = 0.2,
            x_bounds = None,
            figsize = (8,8),
            legend = True,
            ax = None
            ):
    
    show_on_finish = False

    N_classes = 0
    data_lbls = {}
    x1_min = []
    x1_max = []
    x2_min = []
    x2_max = []
    for key, val in data.items():
        data_lbls[key] = torch.argmax(val['y'], dim=1)
        N_classes = max(N_classes, len(torch.unique(data_lbls[key])))
        x1_min.append(min(val['X'][:, 0]))
        x1_max.append(max(val['X'][:, 0]))
        x2_min.append(min(val['X'][:, 1]))
        x2_max.append(max(val['X'][:, 1]))

    x1_min = min(x1_min)
    x1_max = max(x1_max)
    x2_min = min(x2_min)
    x2_max = max(x2_max)


    if classes != None:
        assert len(classes) == N_classes
    else:
        classes = [f'{i}' for i in range(N_classes)]

    # Create custom colormap
    if cmap == None:
        vals = np.linspace(0,1,N_classes)
        cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))

    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        show_on_finish = True

    for key, val in data.items():
        for k in range(N_classes):
            mask = (data_lbls[key] == k)

            try:
                mark = val['m']
            except:
                mark = ','

            try:
                msize = val['msize']
            except:
                msize = 20.0

            lbl = key + '_' + classes[k]
            ax.scatter(val['X'][mask, 0], val['X'][mask, 1],
                        marker=mark, edgecolor='w', s=msize,
                        color=cmap(k), label=lbl)
            
            
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    ax.grid('on')
    ax.legend()
    
    if show_on_finish:
        plt.show()