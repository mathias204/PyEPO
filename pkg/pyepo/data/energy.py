"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PredOpt/predopt-benchmarks
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def get_energy_pandas(fname=None):
    if fname is None:
        fname = "data/prices2013.dat"

    df = pd.read_csv(fname, sep=r'\s+', quotechar='"')
    # remove unnecessary columns
    df.drop(['#DateTime', 'Holiday', 'ActualWindProduction', 'SystemLoadEP2'], axis=1, inplace=True)
    # remove columns with missing values
    df.drop(['ORKTemperature', 'ORKWindspeed'], axis=1, inplace=True)

    # missing value treatment
    # df[pd.isnull(df).any(axis=1)]
    # impute missing CO2 intensities linearly
    df.loc[df.loc[:,'CO2Intensity'] == 0, 'CO2Intensity'] = np.nan # an odity
    df.loc[:,'CO2Intensity'].interpolate(inplace=True)
    # remove remaining 3 days with missing values
    grouplength = 48
    for i in range(0, len(df), grouplength):
        day_has_nan = pd.isnull(df.loc[i:i+(grouplength-1)]).any(axis=1).any()
        if day_has_nan:
            #print("Dropping",i)
            df.drop(range(i,i+grouplength), inplace=True)
    # data is sorted by year, month, day, periodofday; don't want learning over this
    df.drop(['Day', 'Year', 'PeriodOfDay'], axis=1, inplace=True)

    # insert group identifier at beginning
    grouplength = 48
    length = int(len(df)/48) # 792
    gids = [gid for gid in range(length) for i in range(grouplength)]
    df.insert(0, 'groupID', gids)

    return df

# prep numpy arrays, Xs will contain groupID as first column
def get_energy(fname=None):
    df = get_energy_pandas(fname)

    length = df['groupID'].nunique()
    grouplength = 48

    # numpy arrays, X contains groupID as first column
    X1g = df.loc[:, df.columns != 'SMPEP2'].values
    y = df.loc[:, 'SMPEP2'].values

    # no negative values allowed...for now I just clamp these values to zero. They occur three times in the training data.
    # for i in range(len(y)):
    #     y[i] = max(y[i], 0)
    
    

    #print(len(X1g_train),len(X1g_test),len(X),len(X1g_train)+len(X1g_test))
    return X1g, y, length, grouplength


def get_data(num_groups, standardize=True, seed=0, trainTestRatio=0.8):
    x, y, length, grouplength = get_energy(fname= 'data/prices2013.dat')
    print("x.shape", x.shape)
    print("num_groups", length // grouplength)
    print("original train len", trainTestRatio * length * grouplength)
    print("length", length)

    x = x[:,1:]

    num_data = (num_groups * grouplength)

    x,y = shuffle(x,y,random_state=seed)
    x = x[:num_data]
    y = y[:num_data]

    train_len = int(trainTestRatio * num_groups)
    val_len = int(0.5 * (num_groups - train_len))

    print("train_len", train_len)
    print("val_len", val_len)

    # The splitting using grouplength to ensure clean cuts
    x_train = x[:grouplength * train_len]
    print("train", x_train.shape)
    y_train = y[:grouplength * train_len]
    
    x_val = x[grouplength * train_len : grouplength * (train_len + val_len)]
    print("val", x_val.shape)
    y_val = y[grouplength * train_len : grouplength * (train_len + val_len)]
    
    x_test = x[grouplength * (train_len + val_len):]
    y_test = y[grouplength * (train_len + val_len):]

    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    x_train = x_train.reshape(-1,48,x_train.shape[1])
    y_train = y_train.reshape(-1,48)
    x_val = x_val.reshape(-1,48,x_val.shape[1])
    y_val = y_val.reshape(-1,48)
    x_test = x_test.reshape(-1,48,x_test.shape[1])
    y_test = y_test.reshape(-1,48)

    return x_train, y_train, x_val, y_val, x_test, y_test



def get_instance_config(filename):
    with open(filename) as f:
        mylist = f.read().splitlines()
    
    q= int(mylist[0])
    nbResources = int(mylist[1])
    nbMachines =int(mylist[2])
    idle = [None]*nbMachines
    up = [None]*nbMachines
    down = [None]*nbMachines
    MC = [None]*nbMachines
    for m in range(nbMachines):
        l = mylist[2*m+3].split()
        idle[m] = int(l[1])
        up[m] = float(l[2])
        down[m] = float(l[3])
        MC[m] = list(map(int, mylist[2*(m+2)].split()))
    lines_read = 2*nbMachines + 2
    nbTasks = int(mylist[lines_read+1])
    U = [None]*nbTasks
    D=  [None]*nbTasks
    E=  [None]*nbTasks
    L=  [None]*nbTasks
    P=  [None]*nbTasks
    for f in range(nbTasks):
        l = mylist[2*f + lines_read+2].split()
        D[f] = int(l[1])
        E[f] = int(l[2])
        L[f] = int(l[3])
        P[f] = float(l[4])
        U[f] = list(map(int, mylist[2*f + lines_read+3].split()))
    return {"nbMachines":nbMachines,
                "nbTasks":nbTasks,"nbResources":nbResources,
                "MC":MC,
                "U":U,
                "D":D,
                "E":E,
                "L":L,
                "P":P,
                "idle":idle,
                "up":up,
                "down":down,
                "q":q}
