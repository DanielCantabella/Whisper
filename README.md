
Describe in your README.md program ways in which you can improve model performance are using
### (a) Model-centric approach 

#### Tune hyperparameters:
* **Learning Rate** (learning_rate):
Try different magnitudes like 0.1, 0.01, 0.001 to see what helps the model learn better.
You can also play with learning rate schedules or policies during training.


* **Batch Size** (train_batch_size and eval_batch_size):
Smaller batches might help your model generalize, but it could take longer to train.
On the flip side, larger batches could speed things up but might lead to overfitting.


* **Epochs** (num_train_epochs):
Increase the number of training epochs to allow the model to learn more.


* **Random Seed** (seed):
Setting a random seed makes your experiments reproducible. If your results are all over the place, changing the seed might help. 


* **Optimizer** (optimizer):
Different optimizers work differently for various tasks. Alongside Adam, give other optimizers like SGD with momentum or different versions of Adam (like AdamW) a shot.
Tinker with beta parameters and epsilon values.


* **Learning Rate Scheduler** (lr_scheduler_type and lr_scheduler_warmup_steps):
Learning rate schedules can make training more stable. Try out different types like step decay, exponential decay, or cosine annealing.
Adjust warm-up steps. Too short or too long warm-up can mess with your training.


* **Training Steps** (training_steps):
The number of training steps depends on your data and how complex your model is. You might need to tweak this based on how well your model is learning.


* **Mixed Precision Training** (mixed_precision_training):
Using mixed precision training can speed things up without sacrificing accuracy.
But make sure your model and optimizer are cool with it.
You can also try turning off mixed precision training to see if it affects how well your model performs.

#### Fine-tuning model architecture:
The model we have presented consists of the **small** whisper model architecture from OpenAI but there are lots of other
options based on model size:

| Model  | Layers | Width | Heads | Parameters |
|:------:|:------:|:-----:|:-----:|:----------:|
|  Tiny  |   4    |  384  |   6   |    39M     |
|  Base  |   6    |  512  |   8   |    74M     |
| Small  |   12   |  768  |  12   |    244M    |
| Medium |   24   | 1024  |  16   |    769M    |
| Large  |   32   | 1280  |  20   |   1550M    |


### (b) Data-centric approach 

#### New data sources:
We have trained the model wit data coming from the Common Voice dataset in HuggingFace, but there are some other datasets
used for ASR tasks such as the following ones that can be used:
* https://www.voxforge.org/: contains actually 16k Hz audio and its transcriptions (short sentences) in different languages
* https://www.openslr.org/7/: TED-LIUM is an English speech recognition training corpus from TED talks, created by 
Laboratoire d’Informatique de l’Université du Maine (LIUM)
* Fleurs (Conneau et al., 2022): This dataset comprises audio files and transcripts obtained through HuggingFace datasets.
* Multilingual LibriSpeech (Pratap et al., 2020b): Test splits from various languages in the Multilingual LibriSpeech corpus.
* VoxPopuli (Wang et al., 2021): ASR data in 16 languages, including English, obtained using the official repository's script.