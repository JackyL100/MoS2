import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

class LorentzianModel:
    def __init__(self, x0:np.ndarray, scaling, gamma=10,  curves = 2):
        assert len(x0) == curves
        self.parameters = np.zeros((curves, 3))
        for i in range(curves):
            self.parameters[i][0] = x0[i]
            self.parameters[i][1] = gamma
            self.parameters[i][2] = scaling * 10

    def infer_curve(self, x, curve_num):
        """infer using isolated curve"""
        i = curve_num
        return (self.parameters[i][2] / math.pi) * (self.parameters[i][1] / ((x - self.parameters[i][0]) ** 2 + self.parameters[i][1] ** 2))

    def infer(self, x:np.ndarray):
        output = np.zeros(len(x))
        for i in range(len(output)):
            for curve in self.parameters:
                output[i] += (curve[2] / math.pi) * (curve[1] / ((x[i] - curve[0]) ** 2 + curve[1] ** 2))
        return output
    
    def loss(self, x, y_true):
        assert len(x) == len(y_true)
        loss = np.array([0.5 * (self.infer(x[i]) - y_true[i]) ** 2 for i in range(len(x))])
        return loss.sum()
    
    def update(self, x,y, lr):
        """Update parameters using gradient descent and MSE loss with one sample
        
        Parameters:
        x -- independent variable of data (usually first column of data, raman - (cm^-1), pl - (eV))
        y -- dependent variable of data (usually second column of data, intensity)
        parameters -- numpy array of parameters for curves used to fit data
        curve -- which curve to update; if left blank, update all
        lr -- sets how aggressively parameters are updated"""
        p_update = np.empty(self.parameters.shape)
        for curve_ in range(len(self.parameters)):
            x0 = self.parameters[curve_][0]
            gamma = self.parameters[curve_][1]
            scaling = self.parameters[curve_][2]
            p_update[curve_][0] = lr * scaling * (2 * gamma) * (self.infer(x) - y) * (x - x0) / (math.pi * ((x - x0) ** 2 + gamma ** 2) ** 2)
            p_update[curve_][1] = lr * scaling * ((1 / (math.pi * (gamma ** 2 + (x - x0) ** 2))) - ((2 * gamma ** 2) / (math.pi * (gamma ** 2 + (x - x0) ** 2) ** 2))) * (self.infer(x) - y)
            p_update[curve_][2] = lr * (self.infer_curve(x, curve_) / scaling) * (self.infer(x) - y)
            self.parameters -= p_update

    def batch_update(self,x, y, lr=0.0001):
        """Update parameters using batch gradient descent and MSE loss with all data
        
        Parameters:
        x -- list of independent variable of data (usually first column of data, raman - (cm^-1), pl - (eV))
        y -- list of dependent variable of data (usually second column of data, intensity)
        parameters -- numpy array of parameters for curves used to fit data
        lr -- sets how aggressively parameters are updated"""
        assert len(x) == len(y)
        for i in range(len(x)):
            # for j in range(len(parameters)):
            self.update(np.array([x[i]]), np.array([y[i]]), lr)

    def graphvsdata(self,x,y,title=None):
        plt.plot(x, self.infer(x), label="Lorentzian")
        plt.plot(x, y, label="Data")
        if title != None:
            plt.title(title)
        plt.show()