import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import math
import pandas as pd


#########################################
#PLOT UTILITY FUNCTIONS
#########################################

# Contour plot for beale function
def contour_beales_function():
    # plot beales function
    x, y = np.meshgrid(np.arange(-4.5, 4.5, 0.2), np.arange(-4.5, 4.5, 0.2))
    fig, ax = plt.subplots(figsize=(10, 6))
    z = beales_function(np.array([x, y]), features=0, target=0)
    cax = ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap="RdYlBu_r")
    ax.plot(3, 0.5, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((-4.5, 4.5))
    ax.set_ylim((-4.5, 4.5))

    return fig, ax


# Utility to plot beales contours
def plot_surface(x, y, z, azim=-60, elev=40, dist=10, cmap="RdYlBu_r", xlimL=1, xlimH=1, ylimL=1, ylimH=1, zlimL=2,
                 zlimH=2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap': cmap,
                 'linewidth': 20, 'antialiased': True,
                 'vmin': -2, 'vmax': 2}
    ax.plot_surface(x, y, z, **plot_args)
    ax.view_init(azim=azim, elev=elev)
    ax.dist = dist
    ax.set_xlim(-xlimL, xlimH)
    ax.set_ylim(-ylimL, ylimH)
    ax.set_zlim(-zlimL, zlimH)

    plt.xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-1/2", "0", "1/2", "1"])
    plt.yticks([-1, -0.5, 0, 0.5, 1], ["-1", "-1/2", "0", "1/2", "1"])
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.set_zticklabels(["-2", "-1", "0", "1", "2"])

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_zlabel("z", fontsize=18)
    return fig, ax;


def overlay_trajectory(ax, trajectory, label, color='k', lw=2):
    xs = trajectory[:, 0]
    ys = trajectory[:, 1]
    ax.plot(xs, ys, color, label=label, lw=lw)
    ax.plot(xs[-1], ys[-1], color + '>', markersize=14)
    return ax;

#########################################
#LOSS FUNCTIONS
#########################################
def beales_function(theta, features, target):
    x = theta[0]
    y = theta[1]
    return np.square(1.5 - x + x * y) + np.square(2.25 - x + x * y * y) + np.square(2.625 - x + x * y ** 3)


def log_likelihood(theta, features, target):
    '''
    Function to compute the log likehood of theta according to data x and label y

    Input:
    theta: it's the model parameter matrix.
    features: it's the input data matrix. The shape is (N, H)
    target: the label array

    Output:
    log_g: the log likehood of theta according to data x and label y
    '''

    log_l = ((target * np.log(1 / (1 + np.exp(-features @ theta))) + (1 - target) * np.log(
        1 - 1 / (1 + np.exp(-features @ theta)))).sum()) / len(features)

    return log_l


#########################################
#UPDATING THETAS FUNCTIONS
#########################################
def vanilla_descent(theta, target, features, lr, loss, gradient, beta, n_iter, t, theta_p):
    grad = gradient(target, features, theta)
    theta = theta - lr * grad
    return (theta)


def RK_2_adaptive_gradient(theta, target, features, lr, loss, gradient, beta, n_iter, t, theta_p):
    g = gradient(target, features, theta)
    theta_hat = theta - lr * g
    g_tilde = gradient(target, features, theta_hat)
    prod = (g - g_tilde) @ g
    tentative = 2 * lr * prod / (np.linalg.norm(g - g_tilde) ** 2)

    if prod > 0:
        lr_opt = tentative
    else:
        lr_opt = lr

    if lr_opt >= lr:
        lr = beta * lr + (1 - beta) * lr_opt
    else:
        lr = (1 - beta) * lr_opt

    theta = theta - lr * g

    return (theta)


def RK4(theta, target, features, lr, loss, gradient, beta, n_iter, t, theta_p):  # theta=[x0,y0]
    c = lr * 100  # 0.05  #da capire come inizializzare c
    lr = (c / n_iter ** (1 / 5))

    k1 = lr * gradient(target, features, theta)

    theta0 = theta - (1 / 2 * lr * k1)
    k2 = lr * gradient(target, features, theta)

    theta0 -= (1 / 2 * lr * k2)
    k3 = lr * gradient(target, features, theta)

    theta0 -= (1 / 2 * lr * k3)
    k4 = lr * gradient(target, features, theta)

    gradi = (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)  # *lr
    theta = theta - (gradi)

    return theta


# https://jlmelville.github.io/mize/nesterov.html
def NAG(theta, target, features, lr, loss, gradient_f, beta, n_iter, t,
        theta_p):  # t=a quale iterazione stiamo, theta_previous=theta_{t-1}-->s_{t-1}
    mu_s = [0.999, 0.995, 0.99, 0.9, 0]
    mu_max = mu_s[-2]  # aumentando mu_max migliora al convergenza
    mu_t = min(1 - 2 ** (-1 - math.log2(t / 250 + 1)), mu_max)

    s_t_p = theta_p - lr * gradient_f(target, features, theta_p)
    s_t = theta - lr * gradient_f(target, features, theta)
    theta = s_t - mu_t * (s_t - s_t_p)
    return theta


# https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12
def NAG1(theta, target, features, lr, loss, gradient_f, beta, n_iter, t,
         theta_p):  # t=a quale iterazione stiamo, theta_previous=theta_{t-1}-->s_{t-1}
    mu_s = [0.999, 0.995, 0.99, 0.9, 0]
    mu_max = mu_s[-2]  # aumentando mu_max migliora al convergenza
    mu = min(1 - 2 ** (-1 - math.log2(t / 250 + 1)), mu_max)
    theta_a = theta - mu * p_update
    update_t = mu * p_update + lr * gradient_f(target, features, theta_a)
    theta = theta - update_t

    return theta, update_t


#########################################
#GRADIENT FUNCTIONS
#########################################

def beales_gradient(target, features, theta):
    x = theta[0]
    y = theta[1]
    grad_x = 2 * (1.5 - x + x * y) * (-1 + y) + 2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2) + 2 * (
                2.625 - x + x * y ** 3) * (-1 + y ** 3)
    grad_y = 2 * (1.5 - x + x * y) * x + 4 * (2.25 - x + x * y ** 2) * x * y + 6 * (2.625 - x + x * y ** 3) * x * y ** 2
    grad = np.array([grad_x, grad_y])

    return (grad)

def log_likelihood_gradient(target,features,theta):

  log_lik_deriv = (((target - 1 / (1 + np.exp(-features @ theta))).transpose() @ features).transpose() / len(features))
  return log_lik_deriv


#########################################
#GRADIENT DESCENT FUNCTIONS
#########################################

def gradient_descent(theta, features, target, lr, num_steps, loss, GD_type, gradient_function, beta, n_iter, t,
                     theta_p):
    loss_history = np.zeros(num_steps)
    parameter_traj = np.zeros((num_steps, theta.shape[0]))

    for step in range(num_steps):

        parameter_traj[step] = np.reshape(theta, theta.shape[0])
        loss_history[step] = loss(theta, features, target)
        temp = parameter_traj[step - 1]
        # theta=GD_type(theta, target, features, lr,loss,gradient_function,beta,n_iter,t=step, theta_p=temp)

        if GD_type == NAG1:
            if step == 0: up = 0
            theta, up = GD_type(theta, target, features, lr, loss, gradient_function, beta, n_iter, t=step, theta_p=up)
        else:
            theta = GD_type(theta, target, features, lr, loss, gradient_function, beta, n_iter, t=step, theta_p=temp)

    return theta, loss_history, parameter_traj

#########################################
#PREDICTION FUNCTIONS
#########################################
def predictions(features, theta):
    '''
    Function to compute the predictions for the input features

    Input:
    theta: it's the model parameter matrix.
    features: it's the input data matrix. The shape is (N, H)

    Output:
    preds: the predictions of the input features
    '''

    response = 1 / (1 + np.exp(-features @ theta))
    preds = np.where(response >= 0.5, 1, 0)
    return preds
