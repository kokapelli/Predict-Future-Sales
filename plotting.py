from preprocessing import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm

plt.style.use('seaborn')

# Todo make box plot with several inputs
def box_plot(df, target, feat):
    data = pd.concat([df[target], df[feat]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=df[feat], y=target, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

def scatter_plot(df, target, feat):
    sns.scatterplot(x=feat, y=target, palette='Blues', data=df)
    plt.ylabel(target, fontsize=12)
    plt.xlabel(feat, fontsize=12)
    plt.show()

def dist_plot(df, feat, log_transform=None):
    if(log_transform):
        df[feat] = np.log1p(df[feat])

    (mu, sigma) = norm.fit(df[feat])

    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(10, 7))
    
    sns.distplot(df[feat], color="b")
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel=feat)
    ax.set(title=feat +" distribution")
    plt.legend(['$\mu=$ {:.2f}\n$\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
    sns.despine(trim=True, left=True)
    plt.show()

def prob_plot(df, feat):
    stats.probplot(df[feat], plot=plt)
    plt.show()

def plot_every_numeric(df, target):
    
    numeric = get_all_numerical(df)

    fig = plt.figure()
    
    for feature in list(df[numeric]):
        scatter_plot(df, target, feature)   

def plot_every_category(df, target):
    
    categorical = get_all_categorical(df)

    fig = plt.figure()
    
    for feature in list(df[categorical]):
        box_plot(df, target, feature)

def plot_all_corr_heatmap(df, target):
    corr = df.corr()
    plt.subplots(figsize=(14,11))
    sns.heatmap(corr, cmap="twilight_shifted_r", square=True)
    plt.show()

def plot_zoom_corr_heatmap(df, target, feats):
    
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, cmap="twilight_shifted_r", vmax=.8, square=True)
    cols = corrmat.nlargest(feats, target)[target].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

def plot_thresholded_corr_heatmap(df, threshold):
    corr = df.corr()
    kot = corr[corr>=threshold]
    plt.figure(figsize=(12,8))
    sns.heatmap(kot, cmap="twilight_shifted_r")
    plt.show()