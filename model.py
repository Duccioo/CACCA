# model.py


import numpy as np
import tensorflow as tf

class ConditionalAutoencoder():
    def __init__(self,
                 vocab_size,
                 args
                  ):

        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.num_prop = args.num_prop
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        
        self._create_network()


    def _create_network(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size, None])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, None])
        self.C = tf.placeholder(tf.float32, [self.batch_size, self.num_prop])
        self.L = tf.placeholder(tf.int32, [self.batch_size])
        
        decoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        encoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        
        with tf.variable_scope('decode'):
            decode_cell=[]
            for i in decoded_rnn_size[:]:
                decode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell(decode_cell)
        
        with tf.variable_scope('encode'):
            encode_cell=[]
            for i in encoded_rnn_size[:]:
                encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)
        
        self.weights = {}
        self.biases = {}
        # MODIFICA: Rimosso il placeholder per l'errore 'eps' non più necessario in un AE standard.
        # self.eps = { ... }

        self.weights['softmax'] = tf.get_variable("softmaxw", initializer=tf.random_uniform(shape=[decoded_rnn_size[-1], self.vocab_size], minval = -0.1, maxval = 0.1))       
        self.biases['softmax'] =  tf.get_variable("softmaxb", initializer=tf.zeros(shape=[self.vocab_size]))
        
        # MODIFICA: Sostituiti i pesi per 'mean' e 'log_sigma' con un singolo set di pesi per l'output latente deterministico.
        self.weights['out_latent'] = tf.get_variable("outlatentw", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.unit_size, self.latent_size])
        self.biases['out_latent'] = tf.get_variable("outlatentb", initializer=tf.zeros_initializer(), shape=[self.latent_size])

        # Rimozione dei pesi non più necessari
        # self.weights['out_mean'] = ...
        # self.weights['out_log_sigma'] = ...
        # self.biases['out_mean'] = ...
        # self.biases['out_log_sigma'] = ...

        self.embedding_encode = tf.get_variable(name = 'encode_embedding', shape = [self.latent_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        self.embedding_decode = tf.get_variable(name = 'decode_embedding', shape = [self.latent_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        
        # MODIFICA: L'encoder ora restituisce solo il vettore latente.
        self.latent_vector = self.encode()

        self.decoded, decoded_logits = self.decode(self.latent_vector)

        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)
        
        self.reconstr_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=decoded_logits, targets=self.Y, weights=weights))
        
        # MODIFICA: Rimossa la latent_loss (KL divergence).
        # self.latent_loss = self.cal_latent_loss(self.mean, self.log_sigma)

        # MODIFICA: La loss totale è solo la loss di ricostruzione.
        self.loss = self.reconstr_loss 
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.loss)
        
        self.mol_pred = tf.argmax(self.decoded, axis=2)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=None)
        print ("Network Ready")

    def encode(self): 
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(X)[1], 1])
        inp = tf.concat([X, C], axis=-1)
        _, state = tf.nn.dynamic_rnn(self.encoder, inp, dtype=tf.float32, scope = 'encode', sequence_length = self.L)
        c,h = state[-1]

        # MODIFICA: L'encoder ora è deterministico.
        # Calcola un singolo vettore latente tramite una trasformazione lineare dello stato nascosto dell'RNN.
        latent_vector = tf.matmul(h, self.weights['out_latent']) + self.biases['out_latent']
        
        # Rimossa la logica VAE per calcolare mean, log_sigma e campionare.
        return latent_vector

    def decode(self, Z):
        # NESSUNA MODIFICA NECESSARIA: Il decoder funziona allo stesso modo,
        # prende in input un vettore latente Z e le condizioni C.
        seq_length=tf.shape(self.X)[1]
        new_Z = tf.tile(tf.expand_dims(Z, 1), [1, seq_length, 1])
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(self.X)[1], 1])
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        inputs = tf.concat([new_Z, X, C], axis=-1)
        self.initial_decoded_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((self.batch_size, self.unit_size)), tf.zeros((self.batch_size, self.unit_size))) for i in range(self.n_rnn_layer)])
        
        Y, self.output_decoded_state = tf.nn.dynamic_rnn(self.decoder, inputs, dtype=tf.float32, scope = 'decode', sequence_length = self.L, initial_state=self.initial_decoded_state)
        Y = tf.reshape(Y, [self.batch_size*seq_length, -1])
        Y = tf.matmul(Y, self.weights['softmax'])+self.biases['softmax']
        Y_logits = tf.reshape(Y, [self.batch_size, seq_length, -1])
        Y = tf.nn.softmax(Y_logits)
        return Y, Y_logits

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def get_latent_vector(self, x, c, l):
        return self.sess.run(self.latent_vector, feed_dict={self.X : x, self.C : c, self.L : l})
    
    # MODIFICA: Funzione non più necessaria, quindi rimossa.
    # def cal_latent_loss(self, mean, log_sigma): ...

    def train(self, x, y, l, c):
        # MODIFICA: Esegui l'ottimizzazione e calcola solo la reconstruction loss.
        _, cost = self.sess.run([self.opt, self.reconstr_loss], feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return cost
    
    def test(self, x, y, l, c):
        # MODIFICA: Calcola la loss sul set di test senza la latent loss.
        cost = self.sess.run(self.reconstr_loss, feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return cost

    def sample(self, latent_vector, c, start_codon, seq_length):
        # NESSUNA MODIFICA NECESSARIA: Il campionamento dal decoder funziona allo stesso modo.
        l = np.ones((self.batch_size)).astype(np.int32)
        x=start_codon
        preds = []
        for i in range(seq_length):
            if i==0:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c})
            else:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c, self.initial_decoded_state:state})
            preds.append(x)
        return np.concatenate(preds,1).astype(int).squeeze()
