## Instructions
To train a CURL agent on the SoftGym tasks, run
```
python curl/run_curl.py 
```

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | CU_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | CU_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | CU_LOSS: 0.0000
```

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder
```

To train a BC RNN agent on the SoftGym tasks, run
```
python equi/run_imitation_using_rnn.py
```
Parameters:
```
--exp_name: name of the experiment
--env_name: name of the environment
--log_dir: directory to save the logs
--test_episodes: number of episodes to test the agent
--every_test: test the agent every 'every_test' episodes
--num_train_steps: number of training steps
--seed: random seed
--batch_size: batch size
--train_length: Length of input sequence to the LSTM
--use_GMM: STORE TRUE if you want to use Gaussian Mixture Model
--collect_demonstration: STORE TRUE if you want to collect demonstrations. Demonstrations will be stored in the folder 'data/RNN_imitation/video/demo.csv'
--env_kwargs_num_variations: number of variations of the environment
```

Default parameters in file equi/run_imitation_using_rnn.py
Other parameters can be found in file equi/default_config.py