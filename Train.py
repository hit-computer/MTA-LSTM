#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import cPickle, os
import random
import Config

try:
    from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
except:
    pass

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

total_step = 30 #get value from output of Preprocess.py file

config = Config.Config()

word_vec = cPickle.load(open('word_vec.pkl', 'r'))
vocab = cPickle.load(open('word_voc.pkl','r'))

config.vocab_size = len(vocab)

class Model(object):
    def __init__(self, is_training, config, filename):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.size = size = config.hidden_size
        vocab_size = config.vocab_size
        
        filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'input_data': tf.FixedLenFeature([batch_size*num_steps],tf.int64),
                'target': tf.FixedLenFeature([batch_size*num_steps],tf.int64),
                'mask': tf.FixedLenFeature([batch_size*num_steps],tf.float32),
                'key_words': tf.FixedLenFeature([batch_size*config.num_keywords],tf.int64)
            })
        
        self._input_data = tf.cast(features['input_data'], tf.int32)
        self._targets = tf.cast(features['target'], tf.int32) 
        self._input_word = tf.cast(features['key_words'], tf.int32)
        self._init_output = tf.placeholder(tf.float32, [batch_size, size])
        self._mask = tf.cast(features['mask'], tf.float32)
        
        self._input_data = tf.reshape(self._input_data, [batch_size, -1])
        self._targets = tf.reshape(self._targets, [batch_size, -1])
        self._input_word = tf.reshape(self._input_word, [batch_size, -1])
        self._mask = tf.reshape(self._mask, [batch_size, -1])
        
        LSTM_cell = tf.nn.rnn_cell.LSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(
                LSTM_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([LSTM_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True, initializer=tf.constant_initializer(word_vec))
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)#返回一个tensor，shape是(batch_size, num_steps, size)
            keyword_inputs = tf.nn.embedding_lookup(embedding, self._input_word)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        gate = tf.ones([batch_size, config.num_keywords])
        
        atten_sum = tf.zeros([batch_size, config.num_keywords])
        
        with tf.variable_scope("coverage"):
            u_f = tf.get_variable("u_f", [config.num_keywords*config.word_embedding_size, config.num_keywords])
            res1 = tf.sigmoid(tf.matmul(tf.reshape(keyword_inputs, [batch_size, -1]), u_f))
            phi_res = tf.reduce_sum(self._mask, 1, keep_dims=True) * res1
            
            self.output1 = phi_res
            
        outputs = []
        output_state = self._init_output
        state = self._initial_state
        with tf.variable_scope("RNN"):
            entropy_cost = []
            for time_step in range(num_steps):
                vs = []
                for s2 in range(config.num_keywords):
                    with tf.variable_scope("RNN_attention"):
                        if time_step > 0 or s2 > 0: tf.get_variable_scope().reuse_variables()
                        u  = tf.get_variable("u", [size, 1])
                        w1 = tf.get_variable("w1", [size, size])
                        w2 = tf.get_variable("w2", [config.word_embedding_size, size])
                        b  = tf.get_variable("b1", [size])

                        vi = tf.matmul(tf.tanh(tf.add(tf.add(
                            tf.matmul(output_state, w1),
                            tf.matmul(keyword_inputs[:, s2, :], w2)), b)), u)
                        vs.append(vi*gate[:,s2:s2+1])
                
                self.attention_vs = tf.concat(vs, axis=1)
                prob_p = tf.nn.softmax(self.attention_vs)
                
                gate = gate - (prob_p / phi_res)
                
                atten_sum += prob_p * self._mask[:,time_step:time_step+1]
                
                mt = tf.add_n([prob_p[:,i:i+1]*keyword_inputs[:, i, :] for i in range(config.num_keywords)])

                with tf.variable_scope("RNN_sentence"):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(tf.concat([inputs[:, time_step, :], mt], axis=1), state) 
                    outputs.append(cell_output)
                    output_state = cell_output
            
            self._end_output = cell_output
            
        self.output2 = atten_sum    
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, size])
        
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        try:
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.reshape(self._mask, [-1])], average_across_timesteps=False)
        except:
            loss = sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.reshape(self._mask, [-1])], average_across_timesteps=False)
        
        self.cost1 = tf.reduce_sum(loss)
        self.cost2 = tf.reduce_sum((phi_res - atten_sum)**2)
        
        self._cost = cost = (self.cost1 + 0.1*self.cost2) / batch_size
        self._final_state = state
        self._prob = tf.nn.softmax(logits)
        
        if not is_training:
            prob = tf.nn.softmax(logits)
            self._sample = tf.argmax(prob, 1)
            return
        
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data
    
    @property
    def end_output(self):
        return self._end_output
    
    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr    
        
    @property
    def train_op(self):
        return self._train_op
        
    @property
    def sample(self):
        return self._sample
        
def run_epoch(session, m, eval_op):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    initial_output = np.zeros((m.batch_size, m.size))
    for step in range(total_step+1):
        state = m.initial_state.eval()
        cost, _ = session.run([m.cost, eval_op],
                                 {m.initial_state: state,
                                  m._init_output: initial_output})
        
        if np.isnan(cost):
            print 'cost is nan!!!'
            exit()
        costs += cost
        iters += m.num_steps
        
        if step and step % (total_step // 5) == 0:
            print("%d-step perplexity: %.3f cost-time: %.2f s" %
                (step, np.exp(costs / iters),
                 time.time() - start_time))
            start_time = time.time()
        
    return np.exp(costs / iters)
    
def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
                                                
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=True, config=config, filename='coverage_data')

        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()
        
        model_saver = tf.train.Saver(tf.global_variables())
        tf.train.start_queue_runners(sess=session)
        
        #model_saver = tf.train.Saver(tf.all_variables())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.4f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            
            
            if (i+1) % config.save_freq == 0:
                print 'model saving ...'
                model_saver.save(session, config.model_path+'--%d'%(i+1))
                print 'Done!'
            
if __name__ == "__main__":
    tf.app.run()
