import sys
import os
import csv
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dir_path)
#dir_path = '/content/drive/MyDrive/Annie'
#sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
from deepxde.backend import tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
#from generate_plot import plot_1D  # should be changed for the new one
from generate_plots_2d import plot_2D
#from generate_plots_1d_MV import plot_1D
import utils_2D_FK as utils
import scipy.io
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = False, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
#    parser.add_argument('-d', '--dimension', dest='dim', required = False, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    parser.add_argument('-a', '--animation', dest='animation', required = False, action='store_true', help='Create and save 2D Animation')
    parser.add_argument('-ep', '--epochs', dest='epochs', required = False,type = int, help='Number of epochs to run')
    parser.add_argument('-rg', '--regime', dest='regime', type = int, help='Planar (1), spiral (2) or spiral breakup (3) regime')
    parser.add_argument('-tr', '--transformation', dest='transform', type = str, help='hardIC, sin_input, or in_n_out')
    
    args = parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.compat.v1.set_random_seed(seed)
    global random_seed
    random_seed = seed

seed=101
# set_random_seed(seed)

input_2d = 3 # network input size (2D)
num_hidden_layers_2d = 5 # number of hidden layers for NN (2D)
hidden_layer_size_2d = 60 # size of each hidden layers (2D)
output_2d = 3 # network output size (2D)
output_heter = 3 # network output size for heterogeneity case (2D)

## Training Parameters
num_domain = 30000 # number of training points within the domain
num_boundary = 1000 # number of training boundary condition points on the geometry boundary
num_test = 1000 # number of testing points within the domain
num_initial = 98 # number of points to test the initial conditions
MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
MAX_LOSS = 4 # upper limit to the initialized loss
epochs = args.epochs#100000 #100000  # number of epochs for training
lr =  0.0005 # learning rate
noise = 0.1 # noise factor
test_size = 0.8 # precentage of testing data

dim = 2
noise_introduced = False
#inverse_activated = True
inverse_string = args.inverse
inverse = args.inverse  #[inverse_string]
model_folder_name = args.model_folder_name
animation = False
heter = False
w_used = False #data
regime = args.regime

dynamics = utils.system_dynamics(args.regime)

params = dynamics.params_to_inverse(inverse)

file_name =  args.file_name
observe_x, u = dynamics.generate_data(file_name, dim)

## Split data to train and test
#observe_train, observe_test, u_train, u_test = train_test_split(observe_x, u, test_size=test_size)
# Uniformly sample - select 1 train point for every 5 points at each unique location (~20% train)
train_mask = observe_x[:, 2] % 5 == 0

u_test = u[~train_mask]
u_train = u[train_mask]

observe_test = observe_x[~train_mask]
observe_train = observe_x[train_mask]

geomtime = dynamics.geometry_time(dim)
bc = dynamics.BC_func(dim, geomtime)
ic_u, ic_v, ic_w = dynamics.IC_func(dim, regime, observe_train, u_train)

observe_u = dde.PointSetBC(observe_train, u_train, component=0)  # component says which component it is
input_data = [bc, ic_u, ic_v, ic_w, observe_u]

pde = dynamics.pde_2D

net = dde.maps.FNN([input_2d] + [hidden_layer_size_2d] * num_hidden_layers_2d + [output_2d], "tanh", "Glorot uniform")
pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = num_domain,
                            num_boundary=num_boundary,
                            num_initial = num_initial,
                            anchors=observe_train,
                            num_test=num_test)
def input_transform(t):
    return tf.concat(
        (
            t,
            tf.sin(t),
            tf.sin(2 * t),
            tf.sin(3 * t),
            tf.sin(4 * t)
        ),
        axis=1,
    )
    
def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    t = x[:, 2:3] # Input x: (x, y, t)
    x_x = x[:, 0:1]
    
    u_a = 0.95 
    v_init = 0.99
    w_init = 0.99

    u_trans = u * tf.tanh(t) + tf.cast(tf.math.less(x_x, (20+1)*0.3),tf.float32) * u_a
    # Maybe try removing tf.math? and jsut use (x < (20+1)*0.3)

    return tf.concat([u_trans,
                      v * tf.tanh(t) + v_init,
                      w * tf.tanh(t) + w_init ],
                      axis = 1)

if args.transform == 'sin_input':
    net.apply_feature_transform(input_transform)
elif args.transform == 'hardIC':
    net.apply_output_transform(output_transform)
elif args.transform == 'in_n_out':
    net.apply_feature_transform(input_transform)
    net.apply_output_transform(output_transform)

model = dde.Model(pde_data, net)

print("flag 1")
model.compile("adam", lr=lr)
print("model compiled")

## Stabalize initialization process by capping the losses
losshistory, _ = model.train(epochs=1)
num_itertions = len(losshistory.loss_train)
initial_loss = max(losshistory.loss_train[num_itertions - 1])
num_init = 0
while initial_loss>MAX_LOSS or np.isnan(initial_loss).any() or np.isinf(initial_loss).any():  # added checking for inf values
    num_init += 1
    model = dde.Model(pde_data, net)
    model.compile("adam", lr=lr)
    #model.compile("adam", lr=lr, loss_weights=loss_weights)
    losshistory, _ = model.train(epochs=1)
    initial_loss = max(losshistory.loss_train[0])
    if num_init > MAX_MODEL_INIT:
        raise ValueError('Model initialization phase exceeded the allowed limit')

print("initialisation phase finished")

out_path = './results_'+model_folder_name+'/'
if inverse_string:
    variables_file = inverse_string + ".dat"
    variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)
    checker = dde.callbacks.ModelCheckpoint("model.ckpt", save_better_only=True, period=1000)
    losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path, callbacks=[variable,checker])
else:
    checker = dde.callbacks.ModelCheckpoint("model.ckpt", save_better_only=True, period=1000)
    losshistory, train_state = model.train(epochs=epochs, model_save_path = out_path, display_every=1000, callbacks=[checker])

if args.regime == 3: # see if this helps with convergence in spiral breakup
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train(model_save_path = out_path+ "model.ckpt", display_every=1000, callbacks=[checker])

loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test = np.sum(losshistory.loss_test, axis=1)

plt.figure()
plt.semilogy(losshistory.steps, loss_train, label="Train loss", linewidth=2)
plt.semilogy(losshistory.steps, loss_test, label="Test loss", linewidth=2)
plt.title("Loss history")
plt.xlabel("# Steps")
plt.legend()
plt.savefig(out_path + "Loss history")

print("calculating RMSE")
## restore to the best state rather than final
## restore to the best state rather than final
if train_state.best_step == train_state.step:
    model.restore(out_path + 'model.ckpt-' + str(train_state.best_step), verbose=1)
else:
    model.restore(out_path + 'model.ckpt-' + str(train_state.best_step+1), verbose=1)

## Compute rMSE for testing data & all (training + testing)
u_pred_test = model.predict(observe_test)[:,0:1]  # add predict V and W and then plot them (in forward mode)
rmse_u_test = np.sqrt(np.square(u_pred_test - u_test).mean())
u_pred_train = model.predict(observe_train)[:,0:1]
rmse_u_train = np.sqrt(np.square(u_pred_train - u_train).mean())
test_err2 = np.square(u_pred_test - u_test)
train_err2 = np.square(u_pred_train - u_train)
all_err2 = np.concatenate((test_err2, train_err2))
rmse_u_all = np.sqrt( all_err2.mean() )

print('--------------------------')
print('u RMSE for test data:', rmse_u_test)
print('u RMSE for train data:', rmse_u_train)
print('u RMSE for all data:', rmse_u_all)
print('--------------------------')

data_list = [observe_x, observe_train, u_train, u, observe_test, u_test]
# if True and dim == 1:
#         plot_1D(data_list, dynamics, model, model_folder_name)
# elif True and dim == 2:
plot_2D(data_list, dynamics, model, animation, model_folder_name)

v_pred_test = model.predict(observe_test)[:,1:2]
w_pred_test = model.predict(observe_test)[:,2:3]
v_pred_train = model.predict(observe_train)[:,1:2]
w_pred_train = model.predict(observe_train)[:,2:3]

scipy.io.savemat(out_path + "uvw_test_estimates.mat", mdict={'u_pred_test': u_pred_test, 'v_pred_test': v_pred_test, 'w_pred_test': w_pred_test, 'observe_test': observe_test})
scipy.io.savemat(out_path + "uvw_train_estimates.mat", mdict={'u_pred_train': u_pred_train, 'v_pred_train': v_pred_train, 'w_pred_train': w_pred_train, 'observe_train': observe_train})
