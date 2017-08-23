# Malgaonkar, Ashutosh
# 2016-10-16
# 1001-171-483
# Assignment_04

import numpy as np
import Tkinter as Tk
import matplotlib
import math
from numpy import linalg as Al
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys
import scipy.misc
import random
import os
import glob
from sklearn.utils import shuffle


class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        # Farhad Kamangar 2016_09_05
        self.samples = np.array(samples)

        if targets != None:
            self.targets = np.array(targets)
        else:
            self.targets = None


nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": 0.0,  # minimum initial weight
    "max_initial_weights": 0.0,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "gamma_rate": 0.001,  # gamma constant
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 2,
    'number_of_samples_in_each_class': 1000
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "gamma_rate": self.gamma_rate,
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.input_samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."

    def create_samples(self, sample_size):
        print "Sample size Percentage: "+str(sample_size)
        size = float(sample_size)

        self.path = "data.csv"
        self.data = data = np.loadtxt(self.path, skiprows=1, delimiter=',', dtype=np.float32)

        samples_to_process = int(math.floor(len(self.data)* (size/100)))
        self.processed_data = self.data[:samples_to_process]
        print self.processed_data

        max_val= self.processed_data.max(axis=0)
        self.processed_data = self.processed_data/max_val

        print "Samples to process: "+str(samples_to_process)
        print "final size of data processed: "+str(self.processed_data.shape)

    def adjust_weights(self, learning_rate, input_samples, output, target):
        self.neural_network.adjust_weights(learning_rate, input_samples, output, target)

    def create_weights_for_learning(self, delayed_elements):
        self.neural_network.create_weights_for_learning(delayed_elements)

    def weights_to_zero(self, delayed_elements):
        self.neural_network.weights_to_zero(delayed_elements)



class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = -2
        self.xmax = 2
        self.ymin = -2
        self.ymax = 2
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.learning_rate = self.nn_experiment.learning_rate
        self.batch_size = self.nn_experiment.batch_size
        self.sample_size_percentage = self.nn_experiment.sample_size_percentage
        self.delayed_elements = self.nn_experiment.delayed_elements
        self.no_of_iterations = self.nn_experiment.no_of_iterations
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 0.02
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.gamma_rate = 0.001
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Widrow-Hoff Learning")
        self.axes = self.figure.add_subplot(211)
        self.figure = plt.figure("Widrow-Hoff Learning")
        self.axes = self.figure.add_subplot(211)
        plt.title("Widrow-Hoff Learning")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.00001, to_=1, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="Delayed Elements")
        self.delayed_elements_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=1000, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.set(self.delayed_elements)
        self.delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.sample_size_percentage_slider_label = Tk.Label(self.sliders_frame, text="Sample Size Percentage")
        self.sample_size_percentage_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_percentage_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.sample_size_percentage_slider.set(self.sample_size_percentage)
        self.sample_size_percentage_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="Batch Size")
        self.batch_size_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0, to_=1000, resolution=10, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.iterations_slider_label = Tk.Label(self.sliders_frame, text="No of Iterations")
        self.iterations_slider_label.grid(row=0, column=4, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.iterations_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.iterations_slider_callback())
        self.iterations_slider.set(self.no_of_iterations)
        self.iterations_slider.bind("<ButtonRelease-1>", lambda event: self.iterations_slider_callback())
        self.iterations_slider.grid(row=0, column=5, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.zero_weights_button = Tk.Button(self.buttons_frame,
                                               text="Weights to Zero",
                                               bg="yellow", fg="red",
                                               command=lambda: self.randomize_weights_button_callback())

        self.zero_weights_button.grid(row=0, column=5, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())


        self.adjust_weights_button.grid(row=1, column=5, sticky=Tk.N + Tk.E + Tk.S + Tk.W)


        self.initialize()
        #self.refresh_display()

    def initialize(self):
        self.nn_experiment.create_samples(self.nn_experiment.sample_size_percentage)
        #self.nn_experiment.neural_network.randomize_weights()
        self.neighborhood_colors = plt.cm.get_cmap('Accent')
        self.sample_points_colors = plt.cm.get_cmap('Dark2')
        self.xx, self.yy = np.meshgrid(np.arange(self.xmin, self.xmax + 0.5 * self.step_size, self.step_size),
                                       np.arange(self.ymin, self.ymax + 0.5 * self.step_size, self.step_size))
        self.convert_binary_to_integer = []
        for k in range(0, self.nn_experiment.neural_network.layers[-1].number_of_neurons):
            self.convert_binary_to_integer.append(2 ** k)

    def display_samples_on_image(self):
        # Display the samples for each class
        for class_index in range(0, self.number_of_classes):
            self.axes.scatter(self.nn_experiment.data_set.samples[0, class_index * self.number_of_samples_in_each_class: \
                (class_index + 1) * self.number_of_samples_in_each_class],
                              self.nn_experiment.data_set.samples[1, class_index * self.number_of_samples_in_each_class: \
                                  (class_index + 1) * self.number_of_samples_in_each_class],
                              c=self.sample_points_colors(class_index * (1.0 / self.number_of_classes)),
                              marker=(3 + class_index, 1, 0), s=50)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def plot_error(self,mse_price, mse_vol,max_err_price, max_err_vol):
        self.axes.cla()
        self.axes.plot(mse_price, color='blue', label="MSE Price")
        self.axes.plot(mse_vol, color='red', label="MSE Volume")
        self.axes.plot(max_err_price, color='green', label='Max Price Error')
        self.axes.plot(max_err_vol, color='black',label='Max Volume Error')
        plt.title('Widrow Hoff Learning')
        plt.legend(loc='best')
        self.canvas.draw()


    def refresh_display(self):
        self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)
        self.display_neighborhoods()

    def display_neighborhoods(self):
        self.class_ids = []
        for x, y in np.stack((self.xx.ravel(), self.yy.ravel()), axis=-1):
            output = self.nn_experiment.neural_network.calculate_output(np.array([x, y]))
            self.class_ids.append(output.dot(self.convert_binary_to_integer))
        self.class_ids = np.array(self.class_ids)
        self.class_ids = self.class_ids.reshape(self.xx.shape)
        self.axes.cla()
        self.axes.pcolormesh(self.xx, self.yy, self.class_ids, cmap=self.neighborhood_colors)
        self.display_output_nodes_net_boundaries()
        self.display_samples_on_image()

    def display_output_nodes_net_boundaries(self):
        output_layer = self.nn_experiment.neural_network.layers[-1]
        for output_neuron_index in range(output_layer.number_of_neurons):
            w1 = output_layer.weights[output_neuron_index][0]
            w2 = output_layer.weights[output_neuron_index][1]
            w3 = output_layer.weights[output_neuron_index][2]
            if w1 == 0 and w2 == 0:
                data = [(0, 0), (0, 0), 'r']
            elif w1 == 0:
                data = [(self.xmin, self.xmax), (float(w3) / w2, float(w3) / w2), 'r']
            elif w2 == 0:
                data = [(float(-w3) / w1, float(-w3) / w1), (self.ymin, self.ymax), 'r']
            else:
                data = [(self.xmin, self.xmax),  # in form of (x1, x2), (y1, y2)
                        ((-w3 - float(w1 * self.xmin)) / w2,
                         (-w3 - float(w1 * self.xmax)) / w2), 'r']
            self.axes.plot(*data)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class

    def delayed_elements_slider_callback(self):
        self.delayed_elements = self.delayed_elements_slider.get()
        #print self.delayed_elements

    def sample_size_percentage_slider_callback(self):
        self.sample_size_percentage = self.sample_size_percentage_slider.get()
        self.nn_experiment.create_samples(self.sample_size_percentage)
        #print self.sample_size_percentage

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        #print self.batch_size

    def iterations_slider_callback(self):
        self.no_of_iterations = self.iterations_slider.get()
        #print self.no_of_iterations

    def number_of_classes_slider_callback(self):
        self.number_of_classes = self.number_of_classes_slider.get()
        self.nn_experiment.number_of_classes = self.number_of_classes
        self.nn_experiment.neural_network.layers[-1].number_of_neurons = self.number_of_classes
        self.nn_experiment.neural_network.randomize_weights()
        self.initialize()
        #self.refresh_display()

    def number_of_samples_slider_callback(self):
        self.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.create_samples()
        self.refresh_display()

    def create_new_samples_bottun_callback(self):
        temp_text = self.create_new_samples_bottun.config('text')[-1]
        self.create_new_samples_bottun.config(text='Please Wait')
        self.create_new_samples_bottun.update_idletasks()
        self.nn_experiment.create_samples()
        self.refresh_display()
        self.create_new_samples_bottun.config(text=temp_text)
        self.create_new_samples_bottun.update_idletasks()



    def weights_to_zero_callback(self):
        print "Zero Weights"
        self.delayed_elements = self.delayed_elements_slider.get()
        self.nn_experiment.weights_to_zero(self.delayed_elements)

    def adjust_weights_button_callback(self):
        self.mean_square_error=[]
        self.mse_price=[]
        self.mse_vol=[]
        self.max_error=[]
        self.max_error_price=[]
        self.max_error_vol=[]
        self.error_vec =[]
        self.input_samples = self.nn_experiment.processed_data
        self.no_of_iterations = self.iterations_slider.get()
        #print self.no_of_iterations
        if self.batch_size == 0:
            self.batch_size = len(self.input_samples)

        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        for itr in range(self.no_of_iterations):

            for i in range(0,len(self.input_samples),self.batch_size):
                self.batch_data = self.input_samples[i:i+self.batch_size]
                self.error_vec = []
                #print self.batch_data
                #print len(self.batch_data)
                length = int(len(self.batch_data)-self.delayed_elements-1)
                count = 0
                end_count = int(len(self.batch_data) - (self.delayed_elements)) - 1
                for j in range(0,length):
                    input_data = self.batch_data[j:j + self.delayed_elements + 1]
                    self.input_to_network = input_data.reshape(1, 2 + 2 * self.delayed_elements)
                    # self.input_to_network
                    if end_count != count:
                        target = None
                        target = self.batch_data[self.delayed_elements + count + 1]
                        self.target_vec = target.T.reshape(2, 1)
                        self.output = self.nn_experiment.neural_network.calculate_output(self.input_to_network.T)
                        self.nn_experiment.adjust_weights(self.learning_rate, self.input_to_network.T, self.output,
                                                          self.target_vec)
                        #print self.output
                        count += 1

                cnt = 0
                for k in range(0, length):
                    input_data = self.batch_data[k:k + self.delayed_elements + 1]
                    #print len(input_data)
                    self.input_to_network = input_data.reshape(1, 2 + 2 * self.delayed_elements)
                    #print self.input_to_network, len(input_data)
                    if cnt != (length -1):
                        error=None
                        self.target_vec = self.batch_data[self.delayed_elements+cnt+1]
                        self.target_vec = self.target_vec.T.reshape(2,1)
                        #print "target : "+str(self.target_vec)
                        self.output = self.nn_experiment.neural_network.calculate_output(self.input_to_network.T)
                        error = self.target_vec - self.output
                        #print "error : "+str(error)

                        cnt+=1
                    self.error_vec.append(error)

                max_err = np.max(self.error_vec,axis=0)
                mse = np.sum((np.square(self.error_vec)),axis=0)/len(self.error_vec)
                self.mse_price.append(mse[0])
                self.mse_vol.append(mse[1])
                self.max_error_price.append(max_err[0])
                self.max_error_vol.append(max_err[1])
                self.plot_error(self.mse_price,self.mse_vol, self.max_error_price, self.max_error_vol)
            #print self.mse_price,self.mse_vol
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()



















    def randomize_weights_button_callback(self):
        temp_text = self.zero_weights_button.config('text')[-1]
        self.zero_weights_button.config(text='Please Wait')
        self.zero_weights_button.update_idletasks()
        self.delayed_elements = self.delayed_elements_slider.get()
        print self.delayed_elements
        self.nn_experiment.neural_network.randomize_weights(self.delayed_elements)
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        #self.refresh_display()
        self.zero_weights_button.config(text=temp_text)
        self.zero_weights_button.update_idletasks()




neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": 0.0,  # minimum initial weight
    "max_initial_weights": 0.0,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "gamma_rate":0.001,  #gamma rate
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 2,
                              "activation_function": "linear"}],  # list of dictionaries
    'delayed_elements': 0,
    'no_of_iterations': 0,
    'sample_size_percentage': 100
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, delayed_elements, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        self.delayed_elements = delayed_elements
        for layer in self.layers:
            layer.randomize_weights(self.delayed_elements,self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(input_values)
        self.output = output
        return self.output

    def weights_to_zero(self, delayed_elements):
        #print self.layers[0].weights
        #print self.layers[0].weights.shape
        for index, value in enumerate(self.layers[0].weights):
            self.layers[0].weights[index]=0.0

        self.weights = self.layers[0].weights.T
        self.weights = self.weights[:(2 + 2 * delayed_elements)].T
        self.weights = np.vstack([self.weights, np.zeros((1, self.weights.shape[1]), float)])
        # print self.weights.shape
        return self.weights
        #print self.layers[0].weights
        #print self.layers[0].weights.shape


    def adjust_weights(self, learning_rate, input_samples, output, target):
        #print "Output : "+str(output)
        self.error =  target - output
        #print "Error : "+str(self.error)
        self.input_samples = np.vstack([input_samples, np.ones((1, input_samples.shape[1]), float)])
        #print "Input : "+str(self.input_samples)
        self.layers[0].weights = self.layers[0].weights + (2 * learning_rate * np.dot(self.error, self.input_samples.T))
        #print "Weights : "+str(self.layers[0].weights)

    def create_weights_for_learning(self, delayed_elements):

        weights = self.layers[0].weights.T
        weights = self.weights[:(2 + 2 * delayed_elements)].T
        #weights = np.vstack([self.weights, np.zeros((1, self.weights.shape[1]), float)])
        #print self.weights.shape
        self.weights = weights
        return self.weights











single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": 0.0,  # minimum initial weight
    "max_initial_weights": 0.0,  # maximum initial weight
    "number_of_inputs_to_layer": 2,  # number of input signals
    "number_of_neurons": 2,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        #self.randomize_weights()

    def randomize_weights(self, delayed_elements, min_initial_weights=None, max_initial_weights=None):
        self.delayed_elements = int(delayed_elements)
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (self.number_of_neurons, 2 * (self.delayed_elements + 1) + 1))
        print "Weights adjusted to Zero : "+str(self.weights)



        #print "weights randomized"

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        # Farhad Kamangar Sept. 4, 2016
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net

        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": 0.0,  # minimum initial weight
        "max_initial_weights": 0.0,  # maximum initial weight
        "number_of_inputs": 2,  # number of inputs to the network
        "learning_rate": 0.001,  # learning rate
        "gamma_rate": 0.001,    # gamma constant
        "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 2,
        'number_of_samples_in_each_class': 1000,
        'batch_size':0, # 0 := entire trainingset as a batch
        'delayed_elements':0,
        'no_of_iterations':0,
        'sample_size_percentage':100
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Perceptron")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
