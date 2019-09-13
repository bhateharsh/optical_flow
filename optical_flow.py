"""         OPTICAL FLOW
Author: Harsh Bhate
"""

import csv
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import time
from utils import flow_to_color, read_flow_file

# Configuration for Plotting 
rc('text', usetex=True)
rc('xtick', labelsize=10) 
rc('ytick', labelsize=10) 
font = {'family' : 'serif',
        'size'   : 10}
rc('font', **font)


class opticalFlow:
    """Class to implement Horn Schunk Optical Flow"""

    def __init__(self, path, nos_itr = 10, alpha = 1.0):
        """Initialize Path
        """
        frame1_path = os.path.join(path, 'frame1.png')
        frame2_path = os.path.join(path, 'frame2.png')
        self.nos_itr = nos_itr
        self.alpha = alpha
        # Loading the Image
        image = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        self.frame1 = cv2.normalize(image, 
                                    None, 
                                    alpha=0, 
                                    beta=1, 
                                    norm_type=cv2.NORM_MINMAX, 
                                    dtype=cv2.CV_32F)
        image = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        self.frame2 = cv2.normalize(image, 
                                    None, 
                                    alpha=0, 
                                    beta=1, 
                                    norm_type=cv2.NORM_MINMAX, 
                                    dtype=cv2.CV_32F)
        # Reading the ground truth
        ground_truth_flow = read_flow_file(os.path.join(path, 'flow1_2.flo'))
        u_gt_orig = ground_truth_flow[:, :, 0]
        v_gt_orig = ground_truth_flow[:, :, 1]
        self.u_gt = np.where(np.isnan(u_gt_orig), 0, u_gt_orig)
        self.v_gt = np.where(np.isnan(v_gt_orig), 0, v_gt_orig)
        
    def display_frames(self):
        """Display initial frames"""
        save_path = 'results/frames.png'
        difference = np.abs(self.frame1 - self.frame2)
        # Plotting 
        plt.figure()
        plt.title(r"Plot of $p_{X|Y}(x|1)$ vs $x$")
        ax1 = plt.subplot(131)
        plt.axis('off')
        plt.imshow(self.frame1, cmap='gray', vmin=0.0, vmax=1.0)
        ax1.set_title(r'Frame 1')
        ax2 = plt.subplot(132)
        plt.axis('off')
        plt.imshow(self.frame2, cmap='gray', vmin=0.0, vmax=1.0)
        ax2.set_title(r'Frame 2')
        ax3 = plt.subplot(133)
        plt.axis('off')
        plt.imshow(difference, cmap='gray', vmin=0.0, vmax=1.0)
        ax3.set_title(r'Difference')
        fig = plt.gcf()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def gradients(self, frame1, frame2):
        """Function to compute I_x, I_y, I_t"""
        mask_x = np.array([ [-1/4, 1/4], 
                            [-1/4,1/4]])
        mask_y = np.array([ [-1/4, -1/4], 
                            [1/4,1/4]])
        mask_t_2 = np.array([   [-1/4, -1/4], 
                                [-1/4,-1/4]])
        mask_t_1 = np.array([   [1/4, 1/4], 
                                [1/4,1/4]])
        I_x = signal.convolve2d(frame1, 
                                mask_x, 
                                mode='same', 
                                boundary='symm')\
            + signal.convolve2d(frame2, 
                                mask_x, 
                                mode='same', 
                                boundary='symm')
        I_y = signal.convolve2d(frame1, 
                                mask_y, 
                                mode='same', 
                                boundary='symm')\
            + signal.convolve2d(frame2, 
                                mask_y, 
                                mode='same', 
                                boundary='symm')
        I_t = signal.convolve2d(frame1, 
                                mask_t_1, 
                                mode='same', 
                                boundary='symm')\
            + signal.convolve2d(frame2, 
                                mask_t_2, 
                                mode='same', 
                                boundary='symm')
        return [I_x, I_y, I_t]

    def laplacian(self, image):
        """Function to compute Laplacian"""
        laplacian_kernel = np.array([   [0  , 1,  0], 
                                        [1,  -4 , 1], 
                                        [0  , 1,  0]]) 
        delta_I = signal.convolve2d(image,
                                    laplacian_kernel, 
                                    mode='same', 
                                    boundary='symm')
        return delta_I
    
    def horn_schunck(self, alpha=1, nos_itr = 100, delta_t = 0.1):
        "Function to perform HS optical flow"
        start_time = time.time()
        self.alpha = alpha
        self.nos_itr = nos_itr
        # Initializing u,v 
        u = np.zeros_like(self.frame1)
        v = np.zeros_like(self.frame2)
        # Getting Gradients
        I_x, I_y, I_t = self.gradients(self.frame1, self.frame2)
        # Iterating over frames
        for i in range(self.nos_itr):
            delta_u = self.laplacian(u)
            delta_v = self.laplacian(v)
            u = u + delta_t\
                *(alpha*delta_u - u*(np.power(I_x,2)) - I_x*I_y*v - I_t*I_x)
            v = v + delta_t\
                *(alpha*delta_v - v*(np.power(I_y,2)) - I_x*I_y*u - I_t*I_y)
            # Convert Nans to zero
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        end_time = time.time()
        self.time = end_time - start_time
        return u, v

    def get_ground_truth(self):
        """Function to return ground truth"""
        return self.u_gt, self.v_gt

    def downsample_flow(self, x, y, stride=10):
        """Function to return downsampled version of signal"""
        return x[::stride, ::stride], y[::stride, ::stride]

    def plot_flow(self, u, v, stride=10):
        """Plot Stride"""
        save_path = 'results/flowplot.png'
        # Defining Meshgrid for Quiver
        m, n = self.frame1.shape
        x, y = np.meshgrid(range(n), range(m))
        x = x.astype('float64')
        y = y.astype('float64')
        # Downsampling for better visibility
        u_gt_discrete, v_gt_discrete = self.downsample_flow(self.u_gt, 
                                                            self.v_gt, 
                                                            stride=stride)
        u_discrete, v_discrete = self.downsample_flow(  u, 
                                                        v, 
                                                        stride=stride)
        x_discrete, y_discrete = self.downsample_flow(  x, 
                                                        y, 
                                                        stride=stride)        
        # Flow Matrix
        estimated_flow = np.stack((u, v), axis=2)
        gt_flow = np.stack((self.u_gt, self.v_gt), axis=2)
        # Plot the optical flow field
        plt.figure()
        ax1 = plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.imshow(self.frame2, cmap='gray')
        plt.quiver(x_discrete, y_discrete,
                   u_discrete, v_discrete, 
                   color='r')
        ax1.set_title(r'Estimated Flow')
        ax2 = plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.imshow(self.frame2, cmap='gray')
        plt.quiver(x_discrete, y_discrete,
                   u_gt_discrete, v_gt_discrete, 
                   color='r')
        ax2.set_title(r'Ground truth Flow')
        ax3 = plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.imshow(flow_to_color(estimated_flow))
        ax3.set_title(r'Estimated Color Map')
        ax4 = plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.imshow(flow_to_color(gt_flow))
        ax4.set_title(r'Ground truth Color Map')
        fig = plt.gcf()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def normalize(self, x):
        """Function to perform min-max normalization"""
        return (x -np.min(x)) / (np.max(x) - np.min(x))

    def norm(self, x, x_g, y, y_g):
        """Function to compute l-2 norm"""
        return ((x-x_g)**2 + (y-y_g)**2)**0.5

    def angle(self, x, x_g, y, y_g):
        """Function to compute angle"""
        return np.arccos(((x*x_g)+(y*y_g)+1)/(((x+y+1) * (x_g+y_g+1))**0.5))

    def benchmark_flow(self, u, v):
        """Function to benchmark Optical Flow"""
        log_file = "results/log.csv"
        # Normalizing flows
        norm_u = self.normalize(u)
        norm_v = self.normalize(v)
        norm_u_gt = self.normalize(self.u_gt)
        norm_v_gt = self.normalize(self.v_gt)
        # Computing End-Point Error and Angular Error
        EPE = self.norm(norm_u, norm_u_gt, norm_v, norm_v_gt)
        AE = self.angle(norm_u, norm_u_gt, norm_v, norm_v_gt)
        # Computing the average error
        EPE_avg = np.mean(EPE[~np.isnan(EPE)])
        AE_avg = np.mean(AE[~np.isnan(AE)])
        # Printing Error
        msg = "Itr = %d, Time(s) = %.2f, Alpha = %.2f, EPE = %.2f, AE = %.2f"\
            %(self.nos_itr, self.time, self.alpha, EPE_avg, AE_avg)
        print (msg)
        # Logging the error and display
        fields = [self.nos_itr, self.time, self.alpha, EPE_avg, AE_avg]
        with open(log_file,'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def plot_iteration_stats(self):
        """Plot iteration vs AE and Iteration vs EPE"""
        log_path = "results/log.csv"
        mod_path = "results/itrLog.csv"
        save_path = "results/itr_error.png"
        iterations = []
        timing = []
        AE = []
        EPE = []
        data = np.genfromtxt(log_path, 
                            delimiter=',', 
                            dtype=None)
        for entry in data:
            itr, time, _, epe, ae = entry
            iterations.append(itr)
            AE.append(ae)
            EPE.append(epe)
            timing.append(time)
        AE_min = min(AE)
        EPE_min = min(EPE)
        msg = "AE_min: {}, EPE_min: {}".format(AE_min, EPE_min)
        print (msg)
        plt.figure()
        ax1 = plt.subplot(131)
        plt.plot(iterations, AE, 'k')
        ax1.set_title(r'Iterations vs Angular Error')
        ax1.set_xlabel(r'Iterations')
        ax1.set_ylabel(r'Angular Error')
        ax2 = plt.subplot(132)
        plt.plot(iterations, EPE, 'k')
        ax2.set_title(r'Iterations vs End Point Error')
        ax2.set_xlabel(r'Iterations')
        ax2.set_ylabel(r'End Point Error')
        ax2 = plt.subplot(133)
        plt.plot(iterations, timing, 'k')
        ax2.set_title(r'Iterations vs Time')
        ax2.set_xlabel(r'Iterations')
        ax2.set_ylabel(r'Time(seconds)')
        fig = plt.gcf()
        plt.tight_layout()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # Copy the log file
        os.rename(log_path, mod_path)

    def plot_alpha_stats(self):
        """Plot alpha vs AE and alpha vs EPE"""
        log_path = "results/log.csv"
        mod_path = "results/alphaLog.csv"
        save_path = "results/alpha_error.png"
        alpha = []
        AE = []
        EPE = []
        data = np.genfromtxt(log_path, 
                            delimiter=',', 
                            dtype=None)
        for entry in data:
            _, _, Lambda, epe, ae = entry
            alpha.append(Lambda)
            AE.append(ae)
            EPE.append(epe)
        AE_min = min(AE)
        EPE_min = min(EPE)
        msg = "AE_min: {}, EPE_min: {}".format(AE_min, EPE_min)
        print (msg)
        plt.figure()
        ax1 = plt.subplot(121)
        plt.plot(alpha, AE, 'k')
        ax1.set_title(r'$\lambda$ vs Angular Error')
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(r'Angular Error')
        ax2 = plt.subplot(122)
        plt.plot(alpha, EPE, 'k')
        ax2.set_title(r'$\lambda$ vs End Point Error')
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel(r'End Point Error')
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # Copy the log file
        os.rename(log_path, mod_path)  