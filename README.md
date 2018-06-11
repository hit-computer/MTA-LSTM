# MTA-LSTM
We focus on essay generation, which is a challenging task that generates a paragraph-level text with multiple topics. Progress towards understanding different topics and expressing diversity in this task requires more powerful generators and richer training and evaluation resources. To address this, we develop a multi-topic-aware long short-term memory (MTA-LSTM) network. In this model, we maintain a novel multi-topic coverage vector, which learns the weight of of each topic and is sequentially updated during the decoding process. Afterwards this vector is fed to an attention model to guide the generator.

The code in this repository is written in Python 2.7/TensorFlow 1.4.0. And if you use other versions of Python or TensorFlow, you should modify some code. 

## Data Set

Composition Data Set: [Download](https://pan.baidu.com/s/1_JPh5-g2rry2QmbjQ3pZ6w)

Zhihu Data Set: [Download](https://pan.baidu.com/s/1eC4gb_We33kr-ZbHn3KdIA)

## Usage

### Data

In `Data/` respository, you need to prepare two files `TrainingData.txt` and `vec.txt`(word embedding trained by word2vec), which is created from text dataset mentioned above.

### Training

Before train the model, you should set some parameters of this model in `Config.py` file. Then, you need to run `Preprocess.py` file for creating `coverage_data` file(convert trainingdata into binary formats of TensorFlow, and more detail about this can be found in [the blog](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)), `word_vec.pkl` file(this is word embedding) and `word_voc.pkl` file(vocabulary of text). At the same time, you should set `total_step` parameter in `Train.py` whose value is got from output of `Preprocess.py`

Start training the model using `Train.py`:

```
$ python Train.py
```

### Generation

After you train the model, you can generate the text in the control of word set. You should modify `Generation.py` file and set `test_word` to a set of words. Then, if you want, you can also set some parameters for generation in `Config.py` file. Generate text by run:

```
$ python Generation.py
```

