#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import cPickle, os
import random
import Config

test_word = [u'FDA', u'menu']

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

word_vec = cPickle.load(open('word_vec.pkl', 'r'))
vocab = cPickle.load(open('word_voc.pkl','r'))

word_to_idx = { ch:i for i,ch in enumerate(vocab) }
idx_to_word = { i:ch for i,ch in enumerate(vocab) }

gen_config = Config.Config()

gen_config.vocab_size = len(vocab)

class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.size = size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) 
        self._input_word = tf.placeholder(tf.int32, [batch_size, config.num_keywords])
        self._init_output = tf.placeholder(tf.float32, [batch_size, size])
        self._mask = tf.placeholder(tf.float32, [batch_size, num_steps])
        self.seq_length = tf.placeholder(tf.float32, [batch_size, 1])

        
        LSTM_cell = tf.nn.rnn_cell.LSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(
                LSTM_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([LSTM_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True, initializer=tf.constant_initializer(word_vec))
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
            keyword_inputs = tf.nn.embedding_lookup(embedding, self._input_word)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        
        self.initial_gate = tf.ones([batch_size, config.num_keywords])
        gate = self.initial_gate
        
        atten_sum = tf.zeros([batch_size, config.num_keywords])
        
        with tf.variable_scope("coverage"):
            u_f = tf.get_variable("u_f", [config.num_keywords*config.word_embedding_size, config.num_keywords])
            res1 = tf.sigmoid(tf.matmul(tf.reshape(keyword_inputs, [batch_size, -1]), u_f))
            phi_res = self.seq_length * res1
            
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
                
                self.attention_weight = prob_p
                
                gate = gate - (prob_p / phi_res)
                self.output_gate = gate
                
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
        
        self._final_state = state
        self._prob = tf.nn.softmax(logits)

        return
        
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

 

def run_epoch(session, m, data, eval_op, state=None, is_test=False, input_words=None, verbose=False, flag = 1, last_output=None, last_gate=None, lens=None):
    """Runs the model on the given data."""
    x = data.reshape((1,1))
    initial_output = np.zeros((m.batch_size, m.size))
    if flag == 0:
        prob, _state, _last_output, _last_gate, weight, _phi, _ = session.run([m._prob, m.final_state, m.end_output, m.output_gate, m.attention_weight, m.output1, eval_op],
                             {m.input_data: x,
                              m._input_word: input_words,
                              m.initial_state: state,
                              m._init_output: initial_output,
                              m.seq_length: [[lens]]})
                              
        return prob, _state, _last_output, _last_gate, weight, _phi                      
    else:
        prob, _state, _last_output, _last_gate, weight, _ = session.run([m._prob, m.final_state, m.end_output, m.output_gate, m.attention_weight, eval_op],
                             {m.input_data: x,
                              m._input_word: input_words,
                              m.initial_state: state,
                              m._init_output: last_output,
                              m.seq_length: [[lens]],
                              m.initial_gate: last_gate})
    return prob, _state, _last_output, _last_gate, weight
    
def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        
        gen_config.batch_size = 1
        gen_config.num_steps = 1 
        
        beam_size = gen_config.BeamSize	

        initializer = tf.random_uniform_initializer(-gen_config.init_scale,
                                                gen_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model(is_training=False, config=gen_config)

        tf.initialize_all_variables().run()
        
        model_saver = tf.train.Saver(tf.all_variables())
        print 'model loading ...'
        model_saver.restore(session, gen_config.model_path+'--%d'%gen_config.save_time)
        print 'Done!'
        
        test_word = [u'体会',u'母亲',u'滴水之恩',u'母爱',u'养育之恩']
        len_of_sample = gen_config.len_of_generation
        
        _state = mtest.initial_state.eval()
        tmp = []
        beams = [(0.0, [idx_to_word[1]], idx_to_word[1])]
        for wd in test_word:
            tmp.append(word_to_idx[wd])
        _input_words = np.array([tmp], dtype=np.float32)
        test_data = np.int32([1])
        prob, _state, _last_output, _last_gate, weight, _phi  = run_epoch(session, mtest, test_data, tf.no_op(), _state, True, input_words=_input_words, flag=0, lens=len_of_sample)
        y1 = np.log(1e-20 + prob.reshape(-1))
        if gen_config.is_sample:
            try:
                top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
            except:
                top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=True, p=prob.reshape(-1))
        else:
            top_indices = np.argsort(-y1)
        b = beams[0]
        beam_candidates = []
        for i in xrange(beam_size):
            wordix = top_indices[i]
            beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_word[wordix]], wordix, _state, _last_output, _last_gate))
        beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
        beams = beam_candidates[:beam_size] # truncate to get new beams
        for xy in range(len_of_sample-1):
            beam_candidates = []
            for b in beams:
                test_data = np.int32(b[2])
                prob, _state, _last_output, _last_gate, weight = run_epoch(session, mtest, test_data, tf.no_op(), b[3], True, input_words=_input_words, flag=1, last_output=b[4], last_gate=b[5], lens=len_of_sample)
                y1 = np.log(1e-20 + prob.reshape(-1))
                if gen_config.is_sample:
                    try:
                        top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                    except:
                        top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=True, p=prob.reshape(-1))
                else:
                    top_indices = np.argsort(-y1)
                for i in xrange(beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_word[wordix]], wordix, _state, _last_output, _last_gate))
            beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
            
        print ' '.join(beams[0][1][1:]).encode('utf-8')
            
if __name__ == "__main__":
    tf.app.run()
