import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
#from generate_plots_1d import plot_1D
#from generate_plots_2d import plot_2D
from generate_plots_3d import plot_3D
import utils
import pinn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')    
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    parser.add_argument('-a', '--animation', dest='animation', required = False, action='store_true', help='Create and save 2D Animation')
    parser.add_argument('-rg', '--regime', dest='regime', required= True, type = str, help='regime e.g. planar, spiral')
    args = parser.parse_args()

## General Params
noise = 0 # noise factor
test_size = 0.8 # precentage of testing dat

def main(args):
    
    ## Get Dynamics Class
    dynamics = utils.system_dynamics()
    
    ## Parameters to inverse (if needed)
    params = dynamics.params_to_inverse(args.inverse)
    
    ## Generate Data 
    file_name = args.file_name
    observe_x, V, W = dynamics.generate_data(file_name, args.dim)  
    
    ## Split data to train and test
    observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=test_size)
    
    ## Add noise to training data if needed
    if args.noise:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Geometry and Time domains
    geomtime = dynamics.geometry_time(args.dim)
    ## Define Boundary Conditions
    bc = dynamics.BC_func(args.dim, geomtime)
    ## Define Initial Conditions
    ic_v, ic_w = dynamics.IC_func(observe_x, observe_train, args.regime)
    # ic = dynamics.IC_func(observe_x, observe_train, args.regime)
    
    ## Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    # input_data = [bc, ic_v, ic_w, observe_v]
    input_data = [bc, ic, observe_v]
    if args.w_input: ## If W required as an input
        observe_w = dde.PointSetBC(observe_train, w_train, component=1)
        input_data = [bc, ic_v, ic_w, observe_v, observe_w]
    
    ## Select relevant PDE (Dim, Heterogeneity) and define the Network
    model_pinn = pinn.PINN(dynamics, args.dim,args.heter, args.inverse)
    model_pinn.define_pinn(geomtime, input_data, observe_train)
            
    ## Train Network
    out_path = './saved_model'+args.model_folder_name
    model, losshistory, train_state = model_pinn.train(out_path, params)
    model.save("model")
    
    ## Plot total loss history 
    loss_train = np.sum(losshistory.loss_train, axis=1)
    loss_test = np.sum(losshistory.loss_test, axis=1)
    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss", linewidth=2)
    plt.semilogy(losshistory.steps, loss_test, label="Test loss", linewidth=2)
    plt.title("Total loss history "+args.model_folder_name)
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig(out_path + "Total loss history")

    # Plot individual loss term history
    plt.figure()
    # loss_label = ['eq_v', 'eq_w', 'IC_v', 'IC_w', 'BC', 'data']
    loss_label = ['eq_v', 'eq_w', 'IC_v', 'BC', 'data']
    plt.semilogy(losshistory.steps, losshistory.loss_train, label=loss_label, linewidth=2)
    plt.title("Individual loss history " +args.model_folder_name)
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig(out_path + "Individual loss history")

    ## Compute rMSE
    pred = model.predict(observe_test)   
    v_pred, w_pred = pred[:,0:1], pred[:,1:2]
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE for test data:", rmse_v)
    print('--------------------------')
    print("Arguments: ", args)
    
    ## Save predictions, data
    pred_all = model.predict(observe_x)
    v_pred_all, w_pred_all = pred_all[:,0:1], pred_all[:,1:2]
#    np.savetxt("train_data.dat", np.hstack((observe_train, v_train, w_train)),header="observe_train,v_train, w_train")
    np.savetxt(out_path+"vw_pred_all.dat", np.hstack((observe_x, v_pred_all, w_pred_all)),header="observe_x, v_pred, w_pred")
    
    ## Generate Figures
    data_list = [observe_x, observe_train, v_train, V]
    # if args.plot and args.dim == 1:
    #     plot_1D(data_list,dynamics, model, args.model_folder_name)
    # elif args.plot and args.dim == 2:
    #     plot_2D(data_list,dynamics, model, args.animation, args.model_folder_name)
    # elif args.plot and args.dim == 3:
    plot_3D(data_list,dynamics, model, out_path)
    return model

## Run main code
model = main(args)
