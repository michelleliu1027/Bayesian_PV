# import relevant packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error
from itertools import product
from gp_para import gp_tuning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# stop showing warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Seed for reproducibility
rng = np.random.default_rng(12345)

### Impot data for training and evaluation
# import data sheet (time:0~5000s) with temp 120, 140, 160, 180
address = 'data/degradation.xlsx'
y_experiment = np.empty(shape=[0,1])
i = 3
list_temp = (120,140,160,180)
for temp in list_temp:
    df = pd.read_excel(address,sheet_name = 'normal data',usecols = [i],names = None,nrows = 5000)
    df = df.values.tolist()
    df = np.array(df)
    y_experiment =  np.append(y_experiment,df)
    i+=1


def data_setup(NUM_OF_DATA_POINTS,y_normal):
    ################################################################################
    # Data setup with half points evenly spread in 0-999, another half in 1000-4999
    X_num = np.hstack([np.linspace(0,999,num=int(3*NUM_OF_DATA_POINTS/4),dtype=int),np.linspace(1000,4999,num=int(NUM_OF_DATA_POINTS/4),dtype=int)])

    # Data setup with1/4 in 0-199, 2/4 in 200-999, 1/4 1000-4999
    # X_num = np.hstack([ np.linspace(0,199,num=int(NUM_OF_DATA_POINTS/4),dtype=int),
    #                     np.linspace(200,999,num=int(NUM_OF_DATA_POINTS/2),dtype=int),
    #                     np.linspace(1000,4999,num=int(NUM_OF_DATA_POINTS/4),dtype=int)])

    # Data setup with 2/5 in 0-99, 2/5 in 100-999, 1/5 1000-4999
    # X_num = np.hstack([ np.linspace(0,99,num=int(NUM_OF_DATA_POINTS*2/5),dtype=int),
    #                     np.linspace(199,999,num=int(NUM_OF_DATA_POINTS*2/5),dtype=int),
    #                     np.linspace(1000,4999,num=int(NUM_OF_DATA_POINTS*1/5),dtype=int)])

    # Data setup with  points evenly spread 
    # X_num = np.linspace(0,4999,num=int(NUM_OF_DATA_POINTS),dtype=int)

    # Data setup with points spread in log space
    # X_num = np.logspace(0,np.log10(4999), num=NUM_OF_DATA_POINTS, base=10.0, dtype=int)
    # HIGHEST_NOISE_LEVEL = 1e-15
    # KERNEL_ALPHA = 8e-5

    ################################################################################

    x_5000 = np.arange(0,5000).astype(int)
    x_normal = np.empty(shape=[0,2])
    temp_list = np.array([120,140,160,180])

    for temp in temp_list:
        df_temp = np.full((5000,1),temp)
        x_normal_partial = np.hstack([x_5000.reshape(5000,1),df_temp])
        x_normal =  np.concatenate([x_normal,x_normal_partial])

    Y = y_normal[np.hstack([X_num,X_num+5000,X_num+5000*2,X_num+5000*3])]
    X = np.empty(shape=[0,2])
    for temp in temp_list:
        df_temp = np.full((int(NUM_OF_DATA_POINTS),1),temp)
        X_partial = np.hstack([X_num.reshape(int(NUM_OF_DATA_POINTS),1),df_temp])
        X = np.concatenate([X, X_partial])
    return X,Y,X_num,x_normal,temp_list

def ucb(X , gp, dim, delta):
	"""
	Calculates the GP-UCB acquisition function values
	Inputs: gp: The Gaussian process, also contains all data
		x:The point at which to evaluate the acquisition function 
	Output: acq_value: The value of the aquisition function at point x
	"""
	mean, var = gp.predict(X[:, np.newaxis], return_cov=True)
	mean = np.atleast_2d(mean).T
	var = np.atleast_2d(var).T  
	beta = 2*np.log(np.power(5000,2.1)*np.square(np.pi)/(3*delta))
	return mean - np.sqrt(beta)* np.sqrt(np.diag(var))

# Plot function to show performance of gp prediction
def plot_performance(list_DATA_POINTS,performance_parameter):
    plt.figure()
    plt.plot(list_DATA_POINTS, performance_parameter, c='blue', lw=2, zorder=3)
    plt.scatter(list_DATA_POINTS, performance_parameter, c='red', s=10, zorder=4)
    # plt.xscale('log')
    plt.tick_params(axis='y')
    plt.tick_params(axis='x')
    plt.ylabel('Performance parameter')
    plt.xlabel('Number of data points')
    # plt.ylim(0,0.00001)
    # plt.title('Performance of gp',color ='white')
    plt.tight_layout()
    plt.show()

# Plot function to show confidence bounds 
def plot(gp,X,temp_list,NUM_OF_PLOT_POINTS):
    x1 = np.linspace(X[:,0].min(), X[:,0].max(),num = NUM_OF_PLOT_POINTS)
    x2 = np.linspace(X[:,1].min(), X[:,1].max(),num = NUM_OF_PLOT_POINTS)
    x1x2 = np.array(list(product(x1, x2)))
    y_pred, std = gp.predict(x1x2, return_std=True)
    X0p, X1p = x1x2[:,0].reshape(NUM_OF_PLOT_POINTS,NUM_OF_PLOT_POINTS), x1x2[:,1].reshape(NUM_OF_PLOT_POINTS,NUM_OF_PLOT_POINTS)
    Zp = np.reshape(y_pred,(NUM_OF_PLOT_POINTS,NUM_OF_PLOT_POINTS))

    fig = plt.figure(figsize=(16,24),facecolor='white')
    ax = fig.add_subplot(3,2,1, projection='3d')
    surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax.scatter(X[:,0],X[:,1],Y)
    fig.colorbar(surf, aspect=5)

    ax = fig.add_subplot(3,2,2)
    pcm = ax.pcolormesh(X0p, X1p, Zp)
    fig.colorbar(pcm, ax=ax)

    i = 1
    j = int(X.size/8)
    k = 0
    plot_index = 3
    for temp in temp_list:
        plt.subplot(3,2,plot_index)
        plt.plot(x1,Zp[:,i],label = str(temp)+' prediction',c='red',lw=2,zorder=3)
        plt.scatter (X[j-int(X.size/8):j,0],Y[j-int(X.size/8):j],label = str(temp)+'training',c='blue',zorder=2)
        plt.legend()
        plt.title('Plot for '+str(temp))
        plt.scatter(X_5000,y_experiment[k:k+5000],label = str(temp)+' true',s=1,c='grey',zorder=1),
        plt.ylim(0.69,0.77)
        # plt.yticks(np.arange(0.6,0.8,0.01)),plt.title(str(temp)),plt.legend(),plt.tight_layout
        i += int(NUM_OF_PLOT_POINTS/3-1)
        j += int(X.size/8)
        k+=5000
        plot_index+=1
    plt.savefig('plot_comp.png')
    plt.show()

### Data setup and indexing
# Data parameters
NUM_OF_DATA_POINTS = 12
NUM_OF_PLOT_POINTS = 100
NUM_OF_EXTRA_DATA = 40
# Kernel setting and prediction
LOWEST_NOISE_LEVEL = 1e-16
HIGHEST_NOISE_LEVEL = 1e-12
KERNEL_ALPHA = 1e-15
# max mse before remove it from final performance plot
MAX_mse_allowed = 1e-5

gp = GaussianProcessRegressor(alpha=1e-06 ,kernel = 1**2 * RBF(length_scale=6.17) + WhiteKernel(noise_level=1e-16))
#2.35,0.27,4.24,6.17,17,0.32
X,Y,X_num,x_normal,temp_list = data_setup(NUM_OF_DATA_POINTS,y_experiment)
gp.fit(X,Y)
print(gp)
plot(gp,X,temp_list,NUM_OF_PLOT_POINTS)

# Set up performance parameters set
list_DATA_POINTS = np.array([])
list_mse = np.array([])
list_lml = np.array([])

y_mean, sigma = gp.predict(x_normal,  return_std=True)
# Collect mse and lml infomation
mse = mean_squared_error(y_experiment,y_mean)
lml = gp.log_marginal_likelihood(gp.kernel_.theta)

if mse < MAX_mse_allowed:
    list_DATA_POINTS = np.append(list_DATA_POINTS,X.size/8)
    list_mse = np.append(list_mse,mse)
    list_lml = np.append(list_lml,lml)
else:
    print('Excessive mse at %s points = %s' %(X.size/8,mse))

for i in range(NUM_OF_EXTRA_DATA):
    ### Ways to choose next point 
    ## 1. by picking lowest UCB value
    # y_ucb = ucb(X_,gp,0.1,5)
    # y_ucb = y_ucb.reshape(5000)
    # X_next =np.argmin(y_ucb)
    # kernel_alpha = 5e-7

    ## 2. by the biggest distance between prediction adn experiment data
    
    y_mean = y_mean.reshape(20000)
    y_distance = y_mean-y_experiment
    y_distance = y_distance.reshape(20000)
    k=0
    for temp in temp_list:
        X_next = np.array([np.argmax(y_distance[0+5000*k:4999+5000*k]),temp])
        X_next = np.array([np.argmax(y_distance[0+5000*k:4999+5000*k]),temp]).reshape(1,2)
        X = np.append(X,X_next).reshape(int(X.size/2+1),2)
        k+=1
    
    ## 3. by picking the minimum prediction value
    ## (if the value has already used in model training, go to next one til not repeated)
    # y_mean = y_mean.reshape(5000)
    # X_next = np.argmin(y_mean)
    # while X_next in X:
    #     X_next +=1
    # kernel_alpha = 1e-8
    ##

    X = X[X[:,0].argsort()]
    X = X[X[:, 1].argsort(kind='mergesort')]
    X_1d = X[:,0]
    k=0
    for temp in temp_list:
        X_1d[int(X_1d.size/4*k):int(X_1d.size/4*(k+1))] += np.full(int(X_1d.size/4*1),k*5000)
        k+=1

    Y = y_experiment[X_1d]
    # X,Y,X_num,x_normal,temp_list= data_setup(NUM_OF_DATA_POINTS,y_normal)
    gp.fit(X,Y)
    y_mean, sigma = gp.predict(x_normal,return_std=True)

    # Collect mse and lml infomation
    mse = mean_squared_error(y_experiment,y_mean)
    lml = gp.log_marginal_likelihood(gp.kernel_.theta)

    if mse < MAX_mse_allowed:
        list_DATA_POINTS = np.append(list_DATA_POINTS,X.size/8)
        list_mse = np.append(list_mse,mse)
        list_lml = np.append(list_lml,lml)
        if (X.size/8)%10 == 0:
            print('Number of data points',X.size/8)
            plot(gp,X,temp_list,NUM_OF_PLOT_POINTS)
    else:
        print('Excessive mse at %s points = %s' %(X.size/8,mse))


# plot relevant graphs
print('\nMean squared error')
plot_performance(list_DATA_POINTS,list_mse)
print('Log marginal likelihood')
plot_performance(list_DATA_POINTS,list_lml)

plot(gp,X,temp_list,NUM_OF_PLOT_POINTS)