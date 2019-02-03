# Sharma, Roopam
# 1001-559-960
# 2018-12-09
# Assignment-06-01

import tensorflow as tf
from collections import Counter
tf.reset_default_graph()

import tensorflow as tf
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import matplotlib
import pickle

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class MainWindow(tk.Tk):
    # This class creates and controls the main window frames and widgets
    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # set the properties of the row and columns in the master frame
        self.master_frame.rowconfigure(0, weight=15, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        # create the frame for placing plot and controls
        self.model_frame = tk.Frame(self.master_frame)
        # Arrange the widgets
        self.model_frame.grid(row=0, pady=(0, 40), sticky=tk.N + tk.E + tk.S + tk.W)
        # Create an object for plotting graphs in the left frame
        self.model_frame_class = ModelFrame(self, self.model_frame, debug_print_flag=self.debug_print_flag)


class ModelFrame:
    """
    This class creates the plot frame and sliders and buttons
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the tensorflow variables and placeholders
        #########################################################################
        self.X_data = []
        self.X_test_data = []
        self.y_data = []
        self.y_test_data = []
        self.error_rate = []
        self.alpha = tf.placeholder(dtype=tf.float32,name="alpha")
        self.lambda_val = tf.placeholder(dtype=tf.float32,name="lambda")
        self.f1 = tf.placeholder(dtype=tf.int32,name="num_f1")
        self.k1 = tf.placeholder(dtype=tf.int32,name="size_k1")
        self.f2 = tf.placeholder(dtype=tf.int32, name="num_f2")
        self.k2 = tf.placeholder(dtype=tf.int32, name="size_k2")
        self.sample_size = 20
        self.input_tensor = tf.placeholder(dtype=tf.float32,name="input_tensor")
        self.target_tensor = tf.placeholder(dtype=tf.int32,name="target_tensor")
        self.batch_size = tf.placeholder(dtype=tf.int32,name="batch_size")
        self.feed_dict = {self.alpha: 0.1, self.lambda_val: 0.01, self.f1: 32, self.k1: 3,self.f2: 32, self.k2: 3, self.batch_size:250}

        # CNN network
        self.cnn_w1 = tf.Variable(tf.random.uniform([self.k1,self.k1,3,self.f1],-0.001,0.001),dtype=tf.float32,validate_shape=False)
        self.cnn_b1 = tf.Variable(tf.random.uniform([self.f1], -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        self.conv_1 = tf.nn.conv2d(input=self.input_tensor,filter=self.cnn_w1,strides=[1,1,1,1],padding='SAME')
        self.conv_1 = self.conv_1+self.cnn_b1
        self.relu_1 = tf.nn.relu(self.conv_1)
        self.max_pool_1 = tf.nn.max_pool(self.relu_1,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')

        self.cnn_w2 = tf.Variable(tf.random.uniform([self.k2, self.k2, self.f1, self.f2], -0.001, 0.001), dtype=tf.float32, validate_shape=False)
        self.cnn_b2 = tf.Variable(tf.random.uniform([self.f2], -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        self.conv_2 = tf.nn.conv2d(input=self.max_pool_1,filter=self.cnn_w2,strides=[1,1,1,1],padding="SAME")
        self.conv_2 = self.conv_2+self.cnn_b2
        self.relu_2 = tf.nn.relu(self.conv_2)
        self.max_pool_2 = tf.nn.max_pool(self.relu_2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        self.cnn_w3 = tf.Variable(tf.random.uniform([3, 3, self.f2, 32], -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        self.cnn_b3 = tf.Variable(tf.random.uniform([32], -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        self.conv_3 = tf.nn.conv2d(input=self.max_pool_2, filter=self.cnn_w3, strides=[1, 1, 1, 1], padding="SAME")
        self.conv_3 = self.conv_3 + self.cnn_b3
        self.relu_3 = tf.nn.relu(self.conv_3)
        # default shape batch_size,32*32*32
        self.max_pool_3 = tf.nn.max_pool(self.relu_3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        #Fully connected neural network
        # default shape batch_size,32768
        self.nn_input = tf.reshape(self.max_pool_3,(self.batch_size, -1))
        # default shape 32768,120
        self.weights_1 = tf.Variable(tf.random.uniform([32768,1000], -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        # default shape 1,120
        self.bias_1 = tf.Variable(tf.random.uniform((1,1000), -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        # default shape 120,100
        self.weights_2 = tf.Variable(tf.random.uniform((1000, 100), -0.001, 0.001),dtype=tf.float32, validate_shape=False)
        # default shape 1,10
        self.bias_2 = tf.Variable(tf.random.uniform((1,100), -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        self.weights_3 = tf.Variable(tf.random.uniform((100, 10), -0.001, 0.001), dtype=tf.float32,validate_shape=False)
        # default shape 1,10
        self.bias_3 = tf.Variable(tf.random.uniform((1, 10), -0.001, 0.001), dtype=tf.float32, validate_shape=False)
        # default shape batch_size,120
        self.output_1 = tf.matmul(self.nn_input,self.weights_1)+ self.bias_1
        self.activation_1 = tf.nn.relu(self.output_1)
        # default shape  batch_size,10
        self.output_2 = tf.matmul(self.activation_1,self.weights_2) + self.bias_2
        self.activation_2 = tf.nn.relu(self.output_2)
        self.output_3 = tf.matmul(self.activation_2, self.weights_3) + self.bias_3
        self.activation_3 = self.output_3
        self.softmax_output = tf.nn.softmax(self.activation_3)

        self.encoded_target = tf.one_hot(self.target_tensor,10)

        self.softmax_output = tf.clip_by_value(self.softmax_output,1e-7,0.9999999)
        self.indexes = tf.math.argmax(self.softmax_output, axis=1)
        self.cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.encoded_target*tf.log(self.softmax_output)+(1-self.encoded_target)*tf.log(1-self.softmax_output),axis=1))
        self.beta1 = tf.reduce_mean(tf.pow(self.weights_1,2))
        self.beta2 = tf.reduce_mean(tf.pow(self.weights_2,2))
        self.beta3 = tf.reduce_mean(tf.pow(self.weights_3, 2))
        self.cnn_beta1 = tf.reduce_mean(tf.pow(self.cnn_w1, 2))
        self.cnn_beta2 = tf.reduce_mean(tf.pow(self.cnn_w2, 2))
        self.cnn_beta3 = tf.reduce_mean(tf.pow(self.cnn_w3, 2))

        self.cross_entropy_loss = tf.add(self.cross_entropy_loss,self.lambda_val*(self.beta1+self.beta2+
                                    self.beta3+self.cnn_beta1+self.cnn_beta2+self.cnn_beta3))

        # self.dW1,self.dW2,self.db1,self.db2 = tf.gradients(self.cross_entropy_loss,[self.weights_1,self.weights_2,self.bias_1,self.bias_2])
        # self.updated_W1 = tf.assign_sub(self.weights_1,self.alpha*self.dW1)
        # self.updated_b1 = tf.assign_sub(self.bias_1, self.alpha*self.db1)
        # self.updated_W2 = tf.assign_sub(self.weights_2, self.alpha*self.dW2)
        # self.updated_b2 = tf.assign_sub(self.bias_2, self.alpha*self.db2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.cross_entropy_loss)

        self.session = None
        self.initializer = tf.global_variables_initializer()
        self.computed_error = []
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        # row widget size
        master.rowconfigure(0, weight=1)
        # column widget size
        master.columnconfigure(0,weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=1)
        master.columnconfigure(3, weight=1)


        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column = 0,columnspan=3, sticky=tk.N + tk.S + tk.E + tk.W)
        # stretch the figure to cover full row and column
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.figure = plt.figure()
        self.axes = self.figure.gca()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, sticky=tk.N + tk.S + tk.E + tk.W)

        self.plot_frame1 = tk.Frame(self.master)
        self.plot_frame1.grid(row=0, column=3, columnspan=3, sticky=tk.N + tk.S + tk.E + tk.W)
        # stretch the figure to cover full row and column
        self.plot_frame1.grid_columnconfigure(0, weight=1)
        self.plot_frame1.grid_rowconfigure(0, weight=1)
        self.figure1 = plt.figure()
        self.axes1 = self.figure1.gca()
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plot_frame1)
        self.plot_widget1 = self.canvas1.get_tk_widget()
        self.plot_widget1.grid(row=0, sticky=tk.N + tk.S + tk.E + tk.W)

        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame1 = tk.Frame(self.master)
        self.controls_frame2 = tk.Frame(self.master)

        self.controls_frame1.grid(row=1, column=0,columnspan=7, sticky=tk.N + tk.E + tk.S + tk.W)
        #self.controls_frame2.grid(row=1, column=3,columnspan=3, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        self.controls_frame1.columnconfigure(0, weight=1)
        self.controls_frame1.columnconfigure(1, weight=1)
        self.controls_frame1.columnconfigure(2, weight=1)
        self.controls_frame1.columnconfigure(3, weight=1)
        self.controls_frame1.columnconfigure(4, weight=1)
        self.controls_frame2.columnconfigure(0, weight=1)


        self.alpha_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="Alpha (Learning Rate)",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.feed_dict[self.alpha])
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0,columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.lambda_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="Lambda (Weight Regularization)",
                                     command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.feed_dict[self.lambda_val])
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=1, column=0,columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.f1_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000", highlightcolor="#00FFFF",
                                      label="F1(First layer Filter)",
                                      command=lambda event: self.f1_slider_callback())
        self.f1_slider.set(self.feed_dict[self.f1])
        self.f1_slider.bind("<ButtonRelease-1>", lambda event: self.f1_slider_callback())
        self.f1_slider.grid(row=0, column=2,columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.k1_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_= 3, to_=7, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="K1(First layer kernel)",
                                     command=lambda event: self.k1_slider_callback())
        self.k1_slider.set(self.feed_dict[self.k1])
        self.k1_slider.bind("<ButtonRelease-1>", lambda event: self.k1_slider_callback())
        self.k1_slider.grid(row=0, column=4,columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.sample_size_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000", highlightcolor="#00FFFF",
                                      label="Training Sample Size(percentage)",
                                      command=lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.set(self.sample_size)
        self.sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.grid(row=2, column=0,columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.adjust_weights_button = tk.Button(self.controls_frame1, text="Adjust Weights (Train)",
                                               command=self.adjust_weights)
        self.adjust_weights_button.grid(row=2, column=2,columnspan=1, sticky=tk.S )

        self.reset_weights_button = tk.Button(self.controls_frame1, text="Reset Weights", command=self.reset_weights)
        self.reset_weights_button.grid(row=3, column=2,columnspan=1, sticky=tk.N+tk.S )
        self.label = tk.Label(self.controls_frame1, text="Reset weights after changing the sliders for filters and kernels",
                                                  justify="left")
        self.label.grid(row=4, column=2, sticky=tk.N)

        self.f2_slider = tk.Scale(self.controls_frame1, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                  from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="F2(Second layer Filter)",
                                  command=lambda event: self.f2_slider_callback())
        self.f2_slider.set(self.feed_dict[self.f2])
        self.f2_slider.bind("<ButtonRelease-1>", lambda event: self.f2_slider_callback())
        self.f2_slider.grid(row=1,column=2,columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.k2_slider = tk.Scale(self.controls_frame1, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                  from_=3, to_=7, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="K2(Second layer kernel)",
                                  command=lambda event: self.k2_slider_callback())
        self.k2_slider.set(self.feed_dict[self.k2])
        self.k2_slider.bind("<ButtonRelease-1>", lambda event: self.k2_slider_callback())
        self.k2_slider.grid(row=1,column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        self.read_data()
        #self.read_small_data()
        self.initialize_tensorflow()
        self.test_data()

    def adjust_weights(self):
        data_size = 50000
        epochs = 5
        for j in range(epochs):
            print("Epoch :", j + 1," of", epochs)
            for i in range(0,int(data_size*self.sample_size/100),self.feed_dict[self.batch_size]):
                self.feed_dict[self.input_tensor] = self.X_data[i:i+self.feed_dict[self.batch_size]].transpose(0,2,3,1)/255
                self.feed_dict[self.target_tensor] = self.y_data[i:i+self.feed_dict[self.batch_size]]
                res = self.session.run([self.cross_entropy_loss,self.max_pool_3,self.indexes,self.encoded_target,self.nn_input,self.optimizer,self.activation_1,self.activation_2,self.activation_3,self.softmax_output,self.weights_1,self.weights_2,self.weights_3],feed_dict=self.feed_dict)
                err = res[0]
                if i%self.feed_dict[self.batch_size]==0:
                    print("Batch",i/self.feed_dict[self.batch_size]+1,"of ",int(data_size*self.sample_size/100)/self.feed_dict[self.batch_size])
                print("Batch Cross entropy loss : ",err)
            self.test_data()
            self.axes.cla()
            self.axes.plot(self.error_rate)
            self.canvas.draw()
        print(self.computed_error)

    def test_data(self):
        accuracy = 0
        data_size = 10000
        err = 0
        y = []
        for i in range(0, data_size, self.feed_dict[self.batch_size]):
            self.feed_dict[self.input_tensor] = self.X_test_data[i:i + self.feed_dict[self.batch_size]].transpose(0, 2,
                                                                                                                  3,
                                                                                                                  1) / 255
            self.feed_dict[self.target_tensor] = self.y_test_data[i:i + self.feed_dict[self.batch_size]]
            res = self.session.run([self.softmax_output, self.indexes, self.cross_entropy_loss],
                                   feed_dict=self.feed_dict)
            accuracy += np.sum(np.equal(res[1], self.y_test_data[i:i + self.feed_dict[self.batch_size]]))
            print("Test Batch",i/self.feed_dict[self.batch_size]+1, "of 40.0")
            #print("Test batch: ", Counter(res[1]))
            #print("Actual", Counter(self.y_test_data[i:i + self.feed_dict[self.batch_size]]))
            err += res[2]
            y.extend(res[1])
        matrix = np.zeros((10, 10))
        c = 0
        for i, j in zip(self.y_test_data, y):
            matrix[i][j] += 1
            c += 1
        self.display_numpy_array_as_table(matrix/100)
        self.computed_error.append(err)
        accuracy = accuracy * 100 / data_size
        print("Accuracy :", accuracy)
        self.error_rate.append(100 - accuracy)

    def plot_error(self):
        self.axes.cla()
        self.axes.plot(np.linspace(1,len(self.computed_error),len(self.computed_error)),self.computed_error)
        self.axes.set_xlabel("No of iterations")
        self.axes.set_ylabel("Cross Entropy Loss")
        self.axes.set_title("Cross Entropy Loss vs No of Iterations")
        self.canvas.draw()

    def reset_weights(self):
        self.session.close()
        self.session = tf.Session()
        self.axes.cla()
        self.axes1.cla()
        self.computed_error= []
        self.error_rate = []
        self.initialize_tensorflow()
        self.test_data()

    # this function initializes the weights and bias variable to -0.001 to 0.001
    def initialize_tensorflow(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(self.initializer, feed_dict=self.feed_dict)

    def display_numpy_array_as_table(self,input_array):
        # This function displays a 1d or 2d numpy array (matrix).
        # Farhad Kamangar Sept. 2018
        if input_array.ndim == 1:
            num_of_columns, = input_array.shape
            temp_matrix = input_array.reshape((1, num_of_columns))
        elif input_array.ndim > 2:
            print("Input matrix dimension is greater than 2. Can not display as table")
            return
        else:
            temp_matrix = input_array
        number_of_rows, num_of_columns = temp_matrix.shape
        self.axes1.cla()
        tb = self.axes1.table(cellText=np.round(temp_matrix, 2), loc=(0, 0), cellLoc='center')
        for cell in tb.properties()['child_artists']:
            cell.set_height(1 / number_of_rows)
            cell.set_width(1 / num_of_columns)

        plt.xticks(np.arange(10),('0','1','2','3','4','5','6','7','8','9'))
        plt.yticks(np.arange(10), ('9','8','7','6','5','4','3','2','1','0'))
        self.axes1.set_xlabel("Predicted")
        self.axes1.set_ylabel("Actual")
        self.axes1.xaxis.set_label_position('top')
        self.axes1.xaxis.tick_top()
        #plt.show()
        self.canvas1.draw()

    def read_data(self):
        for i in range(1,6):
            with open("Data/data_batch_"+str(i), 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            if len(self.X_data)==0:
                self.X_data = data_dict[b"data"].reshape([10000,3,32,32])
                self.y_data = data_dict[b"labels"]
            else:
                self.X_data = np.append(self.X_data,data_dict[b"data"].reshape([10000,3,32,32]),axis=0)
                self.y_data = np.append(self.y_data,data_dict[b"labels"])
        with open("Data/test_batch", 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        self.X_test_data = data_dict[b"data"].reshape([10000,3,32,32])
        self.y_test_data = data_dict[b"labels"]

    def read_small_data(self):
        with open("Data/data_batch_1", 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            self.X_data = data_dict[b"data"].reshape([10000,3,32,32])
            self.y_data = np.array(data_dict[b"labels"])
            self.X_data = self.X_data[self.y_data<3]
            self.y_data = self.y_data[self.y_data<3]
        with open("Data/test_batch", 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            self.X_test_data = data_dict[b"data"].reshape([10000,3,32,32])
            self.y_test_data = np.array(data_dict[b"labels"])
            self.X_test_data = self.X_test_data[self.y_test_data<3]
            self.y_test_data = self.y_test_data[self.y_test_data<3]

    # callback method
    def alpha_slider_callback(self):
        self.feed_dict[self.alpha] = np.float(self.alpha_slider.get())

    # callback method
    def lambda_slider_callback(self):
        self.feed_dict[self.lambda_val] = np.float(self.lambda_slider.get())

    # callback method
    def f1_slider_callback(self):
        self.feed_dict[self.f1] = np.int(self.f1_slider.get())
        self.axes.cla()

    def f2_slider_callback(self):
        self.feed_dict[self.f2] = np.int(self.f2_slider.get())
        self.axes.cla()

    # callback method
    def sample_size_slider_callback(self):
        self.sample_size = np.int(self.sample_size_slider.get())
        self.axes.cla()

    # callback method
    def k1_slider_callback(self):
        if self.k1_slider.get()%2==0:
            val = self.k1_slider.get()
            if self.feed_dict[self.k1]<self.k1_slider.get():
                val = min(self.k1_slider.get()+1,7)
            elif self.feed_dict[self.k1]>self.k1_slider.get():
                val = max(self.k1_slider.get()-1,3)
            self.k1_slider.set(val)
            self.feed_dict[self.k1] = val
        self.axes.cla()

    def k2_slider_callback(self):
        if self.k2_slider.get()%2==0:
            val = self.k2_slider.get()
            if self.feed_dict[self.k2]<self.k2_slider.get():
                val = min(self.k2_slider.get()+1,7)
            elif self.feed_dict[self.k2]>self.k2_slider.get():
                val = max(self.k2_slider.get()-1,3)
            self.k2_slider.set(val)
            self.feed_dict[self.k2] = val
        self.axes.cla()

# confirm function
def close_window_callback(root):
    root.destroy()

main_window = MainWindow(debug_print_flag=False)
# main_window.geometry("500x500")
# set the state of the main window
main_window.wm_state('normal')
# window title
main_window.title('Assignment_06 --  Sharma')
# window size
main_window.minsize(600, 600)
# function for closing the window and calls confirm function before closing
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
# to keep window running
main_window.mainloop()
