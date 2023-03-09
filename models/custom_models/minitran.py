import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_addons as tfa

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        #print(type(patch))
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class minitran:

    def __init__(self, hparam_dict, save_dir):
        init_count = 0
        self.verbosity = 0
        self.reload_best = True
        self.save_last = True

        ### transformer-specific parameters
        ### DEFAULT
        ### patch_size -- size of patches to extract from the input
        self.patch_size = 16
        ### number of attention heads
        self.n_heads = 8
        ### p
        self.projection_dim = 64
        self.t_units = [self.projection_dim*2, self.projection_dim]
        self.t_layers = 8
        self.mlp_units = [800, 800]
        self.drop_rate = 0.1

        self.crdict = dict()
        self.dropmode = "none"
        self.dropout = []
        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
                init_count += 1
            elif key == "save_location":
                self.saveloc = hparam_dict[key]
                init_count += 1
            elif key == "input_size":
                self.imgsize = hparam_dict[key]
                init_count += 1
            elif key == "save_checkpoints":
                self.savechecks = hparam_dict[key]
                init_count += 1
            elif key == "train_metric":
                train_metric = hparam_dict[key]
                init_count += 1
            elif key == "epochs":
                self.n_epochs = hparam_dict[key]
                init_count += 1
            elif key == "use_best":
                self.reload_best = hparam_dict[key]
            elif key == "save_last_epoch":
                self.save_last = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
            elif key == "avg_channel":
                self.avg_channel = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]
            elif key == "patch_size":
                self.patch_size = hparam_dict[key]
                self.n_patches = int((self.imgsize[0] / self.patch_size) ** 2)
            elif key == "heads":
                self.n_heads = hparam_dict[key]
            elif key == "projection_dim":
                self.projection_dim = hparam_dict[key]
            elif key == "transformer_unit":
                self.t_units = hparam_dict[key]
            elif key == "transformer_layers":
                self.t_layers = hparam_dict[key]
            elif key == "mlp_unit":
                self.mlp_units = hparam_dict[key]
            elif key == "drop_rate":
                self.drop_rate = hparam_dict[key]

            ### number of patches in the input -- (image size // patch size) ** 2
            self.n_patches = int((self.imgsize[0] / self.patch_size) ** 2)

            if self.dropmode == "keep":
                self.keeplen = len(self.dropout)
            elif self.dropmode == "drop":
                self.keeplen = self.imgsize[2] - len(self.dropout)
            else:
                self.keeplen = self.imgsize[2]
            self.imgsize = list(self.imgsize)
            self.imgsize[2] = self.keeplen
            self.imgsize = tuple(self.imgsize)
            print("***IMGSIZE", self.imgsize)
            self.metricset = ["mean_squared_error", "mean_absolute_error"]
            if train_metric not in self.metricset:
                self.tmetric = "mean_squared_error"
            else:
                self.tmetric = train_metric

            self.save_dir = save_dir

            inputs = layers.Input(shape=[self.imgsize[2]])

            ### create transformer block layers
            for _ in range(self.t_layers):
                ### layer normalization (1)
                x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)

                ### multi-head attention layer
                attention_output = layers.MultiHeadAttention(num_heads=self.n_heads,
                                                             key_dim=self.projection_dim,
                                                             dropout=0.1)(x1, x1)
                ### skip connection (1)
                x2 = layers.Add()([attention_output, encoded_patches])
                ### layer normalization (2)
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                ### MLP layer
                x3 = mlp(x3, hidden_units=self.t_units, dropout_rate=0.1)
                ### skip connection (2)
                encoded_patches = layers.Add()([x3, x2])

            ### create a (batch size, proj dim) tensor
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.5)(representation)
            ### append the MLP
            features = mlp(representation, hidden_units=self.mlp_units, dropout_rate=0.5)
            ### create regression outputs
            out_ = layers.Dense(1)(features)
            ### bundle everything together in a Keras model
            self.model = keras.Model(inputs=inputs, outputs=out_)

            print(self.model.summary())
            self.model.compile(loss=self.tmetric, metrics=self.metricset,
                               optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                               )
            self.callbacks = []
            if self.savechecks:
                callback = ModelCheckpoint(self.save_dir + "/checkpoint.h5",
                                           monitor="val_mean_squared_error",
                                           verbose=2,
                                           mode="min",
                                           save_best_only=True,
                                           save_freq="epoch",
                                           save_weights_only=True)
                self.callbacks.append(callback)

    def dtransform(self, data):
        if self.avg_channel:
            if self.transformer == "mean_itr":
                return self.mean_itr(data, nchannels=self.keeplen)
        else:
            return data

    def mean_itr(self, data, nchannels):
        ret_vals = np.zeros((data.shape[0], nchannels))
        n_in_c = data.shape[1] // nchannels
        for i in range(nchannels):
            ret_vals[:, i] = np.mean(data[:, [(ii * nchannels) + i for ii in range(n_in_c)]], axis=1)
        return ret_vals

    def drop_set(self, dchannels):
        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = dchannels - len(self.dropout)
        else:
            self.keeplen = dchannels

    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
            data.set_flatten(True)
            if self.dropmode == "keep":
                data.set_keeps(self.dropout)
                data.set_drops(data.keeps_to_drops())
            elif self.dropmode == "drop":
                data.set_drops(self.dropout)
                keepsl = []
                for i in range(data.nchannels):
                    if i not in self.dropout:
                        keepsl.append(i)
                data.set_keeps(keepsl)
            else:
                data.set_drops([])
                data.set_keeps(data.drops_to_keeps())
        else:
            data.set_flatten(self.crdict[name][0])
            data.set_keeps(self.crdict[name][1])
            data.set_drops(self.crdict[name][2])

    def aggregate(self, data):
        self.change_restore(data, "c", "agg")
        keep_mu = 1
        y_agg = np.zeros(data.get_n_samples())
        if not self.avg_channel:
            keep_mu = data.dims[0] * data.dims[1]
        agg = np.zeros((data.get_n_samples(), self.keeplen * keep_mu))
        for i in range(len(data)):
            tx, ty = data[i]
            agg[i * data.batch_size: min(len(agg), (i + 1) * data.batch_size), :] = self.dtransform(tx)
            y_agg[i * data.batch_size: min(len(agg), (i + 1) * data.batch_size)] = ty
        print(agg.shape)
        self.change_restore(data, "r", "agg")
        return agg, y_agg

    def train(self, train_data, validation_data):
        self.drop_set(train_data.nchannels)
        npx, npy = self.aggregate(train_data)
        self.model.fit(train_data, callbacks=self.callbacks, epochs=self.n_epochs, validation_data=validation_data,
                       verbose=2, batch_size=12)  # self.verbosity)
        if self.save_last:
            self.model.save_weights(self.save_dir + "/last_epoch.h5")
        if self.reload_best and self.savechecks:
            self.model.load_weights(self.save_dir + "/checkpoint.h5")
        self.change_restore(train_data, "r", "train")
        self.change_restore(validation_data, "r", "val")

    def predict(self, x_predict, typein="simg"):
        npx, _ = self.aggregate(x_predict)
        #self.change_restore(x_predict, "c", "predict")
        if typein == "simg":
            dumb_out = []
            #og_ret = x_predict.return_format
            #x_predict.set_return("x")
            dumb_out = self.model(npx)
            ret_y = np.array(dumb_out).reshape(-1).flatten()
            #self.model(x_predict)
            #x_predict.set_return(og_ret)
        return ret_y