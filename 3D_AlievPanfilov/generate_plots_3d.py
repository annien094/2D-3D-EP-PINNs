import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.animation as animation
from PIL import Image
import io

def plot_3D(data_list,dynamics, model, fig_name):
    plot_3D_cell(data_list, dynamics, model, fig_name[1:])

def plot_3D_cell(data_list, dynamics, model, fig_name):
    
    ## Unpack data
    observe_data, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a random cell to show
    cell_x = round(dynamics.max_x*0.75)
    cell_y = round(dynamics.max_y*0.75)
    cell_z = round(dynamics.max_z*0.75)
        
    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_data) if (observe_data[i][0:3]==[cell_x,cell_y,cell_z]).all()]
    observe_geomtime = observe_data[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,3]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if (observe_train[i][0:3]==[cell_x,cell_y,cell_z]).all()]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,3]
    
    ## create figure
    plt.figure()
    plt.plot(t_axis, v_GT, c='b', label='GT')
    plt.plot(t_axis, v_predict, c='r', label='Predicted')
    # If there are any trained data points for the current cell 
    if len(t_markers):
        plt.scatter(t_markers, v_trained_points, marker='x', c='black',s=6, label='Observed')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_cell_plot_3D.tiff")
    png1.close()
    return 0
