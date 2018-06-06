#coding:utf-8
import numpy as np
import tensorflow as tf
import cPickle, os, collections
import Config

config = Config.Config()
config.vocab_size += 4

def Read_WordVec(config):
    with open(config.vec_file, 'r') as fvec:
        wordLS = []
        vec_ls =[]
        fvec.readline()
        
        wordLS.append(u'PAD')
        vec_ls.append([0]*config.word_embedding_size)
        wordLS.append(u'START')
        vec_ls.append([0]*config.word_embedding_size)
        wordLS.append(u'END')
        vec_ls.append([0]*config.word_embedding_size)
        wordLS.append(u'UNK')
        vec_ls.append([0]*config.word_embedding_size)
        for line in fvec:
            line = line.split()
            try:
                word = line[0].decode('utf-8')
                vec = [float(i) for i in line[1:]]
                assert len(vec) == config.word_embedding_size
                wordLS.append(word)
                vec_ls.append(vec)
            except:
                print line[0]
        assert len(wordLS) == config.vocab_size
        word_vec = np.array(vec_ls, dtype=np.float32)
        
        cPickle.dump(word_vec, open('word_vec.pkl','w'), protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(wordLS, open('word_voc.pkl','w'), protocol=cPickle.HIGHEST_PROTOCOL)
        
    return wordLS, word_vec

def Read_Data(config):
    trainingdata = []
    with open(os.path.join(config.data_dir, 'TrainingData.txt'),'r') as ftext:
        for line in ftext:
            line = line.decode('utf-8')
            tmp = line.split()
            idx = tmp.index('</d>')
            doc = tmp[:idx]
            keywords = tmp[idx+1:]
            assert len(keywords) == 5
            
            trainingdata.append((doc, keywords))
    return trainingdata
    
print 'loading the trainingdata...'
DATADIR = config.data_dir
vocab, _ = Read_WordVec(config)

data = Read_Data(config)

word_to_idx = { ch:i for i,ch in enumerate(vocab) }
idx_to_word = { i:ch for i,ch in enumerate(vocab) }
data_size, _vocab_size = len(data), len(vocab)

print 'data has %d document, size of word vocabular: %d.' % (data_size, _vocab_size)
    
def data_iterator(trainingdata, batch_size, num_steps):
    epoch_size = len(trainingdata) // batch_size
    for i in range(epoch_size):
        batch_data = trainingdata[i*batch_size:(i+1)*batch_size]
        raw_data = []
        key_words = []
        for it in batch_data:
            raw_data.append(it[0])
            tmp = []
            for wd in it[1]:
                tmp.append(word_to_idx[wd])
            key_words.append(tmp)
            
        data = np.zeros((len(raw_data), num_steps+1), dtype=np.int64)
        for i in range(len(raw_data)):
            doc = raw_data[i]
            tmp = [1]
            for wd in doc:
                if wd in vocab:
                    tmp.append(word_to_idx[wd])
                else:
                    tmp.append(3)
            tmp.append(2)        
            tmp = np.array(tmp, dtype=np.int64)
            _size = tmp.shape[0]
            data[i][:_size] = tmp
        
        key_words = np.array(key_words, dtype=np.int64)
        
        x = data[:, 0:num_steps]
        y = data[:, 1:]
        mask = np.float32(x != 0)
        yield (x, y, mask, key_words)
            
            
train_data = data
writer = tf.python_io.TFRecordWriter("coverage_data")
dataLS = []
for step, (x, y, mask, key_words) in enumerate(data_iterator(train_data, config.batch_size, config.num_steps)):
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'input_data': tf.train.Feature(
                int64_list=tf.train.Int64List(value=x.reshape(-1).astype("int64"))),
            'target': tf.train.Feature(
                int64_list=tf.train.Int64List(value=y.reshape(-1).astype("int64"))),
            'mask': tf.train.Feature(
                float_list=tf.train.FloatList(value=mask.reshape(-1).astype("float"))),
            'key_words': tf.train.Feature(
                int64_list=tf.train.Int64List(value=key_words.reshape(-1).astype("int64"))),    
    }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
    
print 'total step: ',step
