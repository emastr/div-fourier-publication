import numpy as np
import framework.windfield as wf
import framework.tools as tools
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K

class MLPWindfield(wf.Windfield):
    """
    See mlp_windfield.py. Essentially the same code but with a quick hack to enable multithreading
    """

    def __init__(self, epochs = 600, learning_rate = 0.001, layers = [20,20,20,2], activation = 'relu',
                 l2 = 0.01, decay = 0.0, dropout_rate = 0.0, batch_size = 32, elevation = True, gamma = 0.01, N = 10, altitude_data=None):

        self.MLP = None
        self.scaler_data = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.elevation = elevation
        self.lookup_data = altitude_data
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.layers = layers
        self.regularization = l2
        self.decay = decay
        self.test_data = None
        self.N = N
        self.gamma = gamma


    def fit(self, calibration_data: wf.WindDataFrame):
        ############
        frame = self.lookup_data
        frame = frame[['x', 'y', 'altitude']].drop_duplicates()
        l1 = len(frame.index)

        x_ref = np.array(frame['x']).reshape((l1, 1))
        y_ref = np.array(frame['y']).reshape((l1, 1))

        def lookup(x, y):
            # Copied code from nn_windfield
            l2 = len(x)
            x_m = np.array(x).reshape((1, l2))
            y_m = np.array(y).reshape((1, l2))

            d_x = np.repeat(x_ref, l2, axis=1) - np.repeat(x_m, l1, axis=0)
            d_y = np.repeat(y_ref, l2, axis=1) - np.repeat(y_m, l1, axis=0)

            distance = d_x ** 2.0 + d_y ** 2.0

            nearest = np.argmin(distance, axis=0)
            return frame.iloc[nearest]['altitude'].values

        self.altitude_lookup = lookup
        #############
        # Unhash the code-snippet below to clear previous session (might help if things start slowing down).
        tf.keras.backend.clear_session()

        #Encode data:
        x1 = calibration_data.x  # tools.get_x(calibration_data).values
        x2 = calibration_data.y  # tools.get_y(calibration_data).values
        h  = calibration_data.altitude # tools.get_y(calibration_data).values
        u1 = calibration_data.u  # tools.get_u(calibration_data).values
        u2 = calibration_data.v  # tools.get_v(calibration_data).values

        if self.elevation is True:
            data_x = np.hstack((x1[:, None], x2[:, None], h[:, None]))

        else:
            data_x = np.hstack((x1[:, None], x2[:, None]))
        data_v = np.hstack((u1[:, None], u2[:, None]))

        # Apply data normalization and add extra grid points:
        self.scaler_data = StandardScaler()
        self.scaler_data.fit(data_x)
        normalized_data_x = self.scaler_data.transform(data_x)

        # Create grid points:
        delta = 10
        p1 = tf.convert_to_tensor(self.scaler_data.transform(np.hstack((np.array([x1 - delta, x2]).T, h[:,None]))), dtype = tf.float32)
        p2 = tf.convert_to_tensor(self.scaler_data.transform(np.hstack((np.array([x1, x2 + delta]).T, h[:,None]))), dtype = tf.float32)
        p3 = tf.convert_to_tensor(self.scaler_data.transform(np.hstack((np.array([x1 + delta, x2]).T, h[:,None]))), dtype = tf.float32)
        p4 = tf.convert_to_tensor(self.scaler_data.transform(np.hstack((np.array([x1, x2 - delta]).T, h[:,None]))), dtype = tf.float32)
        data_v = np.hstack((data_v, p1, p2, p3, p4))


        # Create the network:
        #Learning rate schedule:
        def schedule(epoch):
            if epoch <= 10:
                return 0.1
            if epoch > 10:
                return 0.001

        def div_metric(model):
            def metric(data, y_pred):
                p1 = data[:,2:5]
                p2 = data[:,5:8]
                p3 = data[:,8:11]
                p4 = data[:,11:14]
                y_pred_p1 = model(p1)
                y_pred_p2 = model(p2)
                y_pred_p3 = model(p3)
                y_pred_p4 = model(p4)

                # Step 2: Calculate the partial derivatives with a three-point centered difference.
                scale_x = self.scaler_data.scale_[0] #scale-factor for x
                scale_y = self.scaler_data.scale_[1] #scale-factor for y

                dudx = (y_pred_p1[:, 0] - y_pred_p3[:, 0]) / (p1[:,0] - p3[:,0]) # <- pj = transformed data
                dvdy = (y_pred_p2[:, 1] - y_pred_p4[:, 1]) / (p2[:,1] - p4[:,1]) # <- pj = transformed data

                # Step 3: Calculate the divergence.
                divergence =  ( dudx / scale_x + dvdy / scale_y ) * np.mean([scale_x, scale_y])

                return K.mean(K.abs(divergence))
            return metric

        #Custom divergence loss:
        def div_loss(gamma, model):
            """Wrapper function for the divergence loss."""

            def loss(data, y_pred):
                # TODO: Try using points other than the training data points for the divergence calculation.
                """Punish non-zero divergence. Each input of size N (batch size) is expanded into
                4 sets of surrounding points, p1, p2, p3, and p4, where the elements pj_i (j = 1,2,3,4 and
                i = 1,2,3...N) are defined according to:

                         y
                         |                                      p2_i
                         |
                         |                               p1_i   P_i   p3_i
                         |
                         ------------- x                        p4_i


                The sets p1, p2, p3, and p4 are used in estimating the divergence for each point P_i. The partial
                derivatives are calculated with a three-point centered difference.

                The extra points are "smuggled" into the loss function in the data argument:

                data.head() =
                      <  y_true  > <------ p1 -------> <------- p2 ------> <------- p3 ------> <--------p4 ------>
                      |  u  |  v  |  x1  |  y1  |  h  |  x2  |  y2  |  h  |  x3  |  y3  |  h  |  x4  |  y4  |  h  |
                """
                y_true = data[:,:2]
                p1 = data[:,2:5]
                p2 = data[:,5:8]
                p3 = data[:,8:11]
                p4 = data[:,11:14]

                ### Calculate divergence using model predictions:

                # Step 1: Use the model to calculate predicted wind field in the surrounding points p1, p2, p3 and p4.
                y_pred_p1 = model(p1)
                y_pred_p2 = model(p2)
                y_pred_p3 = model(p3)
                y_pred_p4 = model(p4)

                # Step 2: Calculate the partial derivatives with a three-point centered difference.
                scale_x = self.scaler_data.scale_[0] #scale-factor for x
                scale_y = self.scaler_data.scale_[1] #scale-factor for y

                dudx = (y_pred_p1[:, 0] - y_pred_p3[:, 0]) / (p1[:,0] - p3[:,0]) # <- pj = transformed data
                dvdy = (y_pred_p2[:, 1] - y_pred_p4[:, 1]) / (p2[:,1] - p4[:,1]) # <- pj = transformed data

                # Step 3: Calculate the divergence.
                divergence =  ( dudx / scale_x + dvdy / scale_y ) * np.mean([scale_x, scale_y])
                #tf.print(K.mean(K.abs(divergence)))

                # Step 4: Calculate and return total loss.
                return K.mean(K.square(y_true - y_pred)) + gamma*K.mean(K.square(divergence))
            return loss


        # Compile the network:
        tf.random.set_seed(133)
        #lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule)
        self.MLP = tf.keras.Sequential()
        for i in range(len(self.layers)-1):
            self.MLP.add(tf.keras.layers.Dense(self.layers[i],
                                               activation=self.activation,
                                               kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
                                               ))

        self.MLP.add(tf.keras.layers.Dense(self.layers[-1]))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-08,
                                                  amsgrad=False,
                                                  name="Adam")

        #self.MLP.compile(loss='mse', optimizer=self.optimizer, metrics=['mae', 'mse'])
        self.MLP.compile(loss=div_loss(self.gamma, self.MLP), optimizer=self.optimizer, metrics=[div_metric(self.MLP)])

        #Train the network:
        if self.test_data is not None:
            x1 = self.test_data.x  # tools.get_x(calibration_data).values
            x2 = self.test_data.y  # tools.get_y(calibration_data).values
            h  = self.test_data.altitude  # tools.get_y(calibration_data).values
            u1 = self.test_data.u  # tools.get_u(calibration_data).values
            u2 = self.test_data.v  # tools.get_v(calibration_data).values
            if self.elevation is True:
                val_x = np.hstack((x1[:, None], x2[:, None], h[:, None]))
                normalized_val_x = self.scaler_data.transform(val_x)
            else:
                val_x = np.hstack((x1[:, None], x2[:, None]))
                normalized_val_x = self.scaler_data.transform(val_x)

            # Calculate generalization error for the zeroth epoch:
            train_history = []
            div_history = []

            test_history = [np.nan for k in range(self.epochs)]
            pred_wind = self.MLP.predict(normalized_val_x)
            u_est = pred_wind[:, 0]
            v_est = pred_wind[:, 1]
            error_u = u1 - u_est
            error_v = u2 - v_est
            ase = np.mean(error_u ** 2 + error_v ** 2)
            test_history[0] = ase

            for i in range(1,self.N+1):
                history = self.MLP.fit(normalized_data_x, data_v, epochs=int(self.epochs/self.N), batch_size = self.batch_size, verbose=0)

                # Calculate generalization error:
                pred_wind = self.MLP.predict(normalized_val_x)
                u_est = pred_wind[:,0]
                v_est = pred_wind[:,1]
                error_u = u1 - u_est
                error_v = u2 - v_est
                ase = np.mean(error_u ** 2 + error_v ** 2)
                test_history[(int(self.epochs/self.N))*i-1] = ase
                train_history = train_history + history.history['loss']
                div_history = div_history + history.history['metric']
            if self.N == 0:
                history = self.MLP.fit(normalized_data_x, data_v, epochs=self.epochs, batch_size=self.batch_size,
                                       verbose=0)
                test_history = [np.nan for i in range(len(history.history['loss']))]
                #print("Loss: ", np.sqrt(2 * history.history['loss'][-1]))
                return history.history['loss'], test_history, history.history['metric']
            else:
                #print("Loss: ", np.sqrt(2 * history.history['loss'][-1]))
                #plt.plot(div_history)
                #plt.show()
                return train_history, test_history, div_history
        else:
            history = self.MLP.fit(normalized_data_x, data_v, epochs = self.epochs, batch_size=self.batch_size, verbose=0)
            #print("Loss: ",np.sqrt(2*history.history['loss'][-1]))

    def predict(self, x, y) -> wf.WindDataFrame:
        ############
        frame = self.lookup_data
        frame = frame[['x', 'y', 'altitude']].drop_duplicates()
        l1 = len(frame.index)

        x_ref = np.array(frame['x']).reshape((l1, 1))
        y_ref = np.array(frame['y']).reshape((l1, 1))

        def lookup(x, y):
            # Copied code from nn_windfield
            l2 = len(x)
            x_m = np.array(x).reshape((1, l2))
            y_m = np.array(y).reshape((1, l2))

            d_x = np.repeat(x_ref, l2, axis=1) - np.repeat(x_m, l1, axis=0)
            d_y = np.repeat(y_ref, l2, axis=1) - np.repeat(y_m, l1, axis=0)

            distance = d_x ** 2.0 + d_y ** 2.0

            nearest = np.argmin(distance, axis=0)
            return frame.iloc[nearest]['altitude'].values

        self.altitude_lookup = lookup
        #############
        if self.elevation is True:
            h = self.altitude_lookup(x,y)
            coordinates = np.hstack((x[:, None], y[:, None], h[:, None]))
        else:
            coordinates = np.hstack((x[:, None], y[:, None]))
        predicted_wind_vectors = self.MLP.predict(self.scaler_data.transform(coordinates))

        u = predicted_wind_vectors[:,0]
        v = predicted_wind_vectors[:,1]
        return tools.create_wind_data_frame(x, y, np.real(u),np.real(v))


    @classmethod #('borrowed' from the random_forest module)
    def make_altitude_lookup(cls, frame):
        frame = frame[['x', 'altitude']].drop_duplicates().set_index('x')
        def lookup(x, y):
            return frame.loc[x]['altitude'].values
        return lookup