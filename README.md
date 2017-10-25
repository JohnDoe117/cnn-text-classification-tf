**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

This is a modified implementation of CNN Classification in tensorflow that is modified for the purpose of Multi class classification with Word2Vec pre-trained model. This has been modified to work with the later versions of Tensorflow.
 https://github.com/dennybritz/cnn-text-classification-tf
 Works well on both Windows and Linux.

Thanks to @j314erre for this gist https://gist.github.com/j314erre/b7c97580a660ead82022625ff7a644d8 and @Psycho7 for his Modification https://github.com/dennybritz/cnn-text-classification-tf/issues/17#issuecomment-234845061 to run without hogging up memory.

## Requirements

- Python 3.6
- Tensorflow - 1.3
- Numpy - The minimum needed version for Tensorflow will be automatically installed.

## Training

Print parameters:

```bash
python3 train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
python3 train.py --embedding_dim=300 --word2vec=./GoogleNews-vectors-negative300.bin
```
Please download this file and place it in the right path.

## Training Dataset Modification.
The training set has been concatenated into a single file and follows the following format. The first Hyphen is the separator, so ensure that the Category name does not have a Hyphen in between it.
CategoryName1-TrainingSample1
CategoryName1-TrainingSample2
CategoryName2-TrainingSample1
CategoryName2-TrainingSample2

## Important Note
Modify the parameters according to your use case, the batch size, number of epochs and the filter sizes have all been reduced to suit my application. Modify those before trying out for your use case.

## Evaluating

```bash
python3 eval.py --checkpoint_dir="./runs/1459637919/checkpoints"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.
Replace the run ID with the ID that you have just ran in the runs folder that gets generated.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
