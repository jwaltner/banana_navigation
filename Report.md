[//]: # (Image References)

[scores]: ./scores.png "Scores"
[scores_update_every_30]: ./scores_update_every_30.png "Scores Updating Target every 30 Frames"

# Report

This report details the algorithm used to solve the Udacity Unity Bananas environment with relevant performance metrics and ideas for future work.

## Learning Algorithm

The learning algorthm used to solve this environment was a Deep Q-Network (dqn) with experience replay and fixed Q-targets and soft updates to the target network every 4 steps.  Updating the target network every 4 steps allows us to decouple the local network training from the target network training and in theory provides better stability in training.  While this is true in general, running the training updating both the target and local network each frame did not hurt training performance for solving this particular environment.

The network itself used a deep neural net defined in [model.py](./model.py) whose hidden layers sequentially doubled then halved the number of units in a layer to allow for sufficient degrees of freedom to learn complex behaviors.  The hidden layers used a ReLU activation function and the final output layer which mapped to the four available actions did not have any additional shaping functions applied.

with experience replay, 

## Training

Training this dqn model with the agent described above enabled resulted in a solution to the environment which consistently attained scores above 13 in 479 episodes which is better than the target of 1800 episodes.  The scores attained rose to the approximate maximum rolling average of 15 at approximately 800 episodes, and then did not improve any further.  Notably, while the average of the scores was 15, there was significant variation in scores between episodes as scores of a trained agent could be as low as 0 and as high as 25 as seen here.  The training was perfomed on 2.7 GHz Intel i7 CPU and did not take an excessive amount of time to solve the environment.

### Rewards
![scores][scores]

```
Episode 100 	Average Score: 0.87 	Elapsed Time: 105.39 sec
Episode 200 	Average Score: 4.22 	Elapsed Time: 218.15 sec
Episode 300 	Average Score: 6.75 	Elapsed Time: 335.13 sec
Episode 400 	Average Score: 10.49	Elapsed Time: 453.27 sec
Episode 479 	Average Score: 13.02	Elapsed Time: 544.93 sec
Environment solved in 479 episodes!	Average Score: 13.02	Elapsed Time: 544.93 sec
Episode 500 	Average Score: 13.35	Elapsed Time: 569.86 sec
Episode 600 	Average Score: 13.65	Elapsed Time: 686.16 sec
Episode 700 	Average Score: 14.13	Elapsed Time: 804.00 sec
Episode 800 	Average Score: 14.71	Elapsed Time: 925.84 sec
Episode 815 	Average Score: 15.00	Elapsed Time: 944.29 sec
Environment solved in 815 episodes!	Average Score: 15.00	Elapsed Time: 944.29 sec
Episode 900 	Average Score: 14.60	Elapsed Time: 1046.52 sec
Episode 1000	Average Score: 14.00	Elapsed Time: 1171.18 sec
Episode 1100	Average Score: 13.90	Elapsed Time: 1292.38 sec
Episode 1200	Average Score: 14.31	Elapsed Time: 1418.26 sec
Episode 1300	Average Score: 14.49	Elapsed Time: 1537.98 sec
Episode 1400	Average Score: 13.96	Elapsed Time: 1655.83 sec
Episode 1500	Average Score: 15.08	Elapsed Time: 1774.44 sec
Episode 1600	Average Score: 14.20	Elapsed Time: 1892.12 sec
Episode 1700	Average Score: 15.06	Elapsed Time: 2007.87 sec
Episode 1800	Average Score: 14.57	Elapsed Time: 2123.50 sec
Episode 1900	Average Score: 13.83	Elapsed Time: 2239.39 sec
Episode 2000	Average Score: 14.70	Elapsed Time: 2357.32 sec
```

### Update Every X Steps

Note that there was no imact on stability of training in this case when the target model was updated every timestep.  This was surprising.  Additionally note that when the target model was updated every 30 frames instead of every 4 frames, that the number of episodes that could be processed in a given time sped up; however, the amount of performance gained each episode reduced.  The net result was slower training and more time to a given performance target.

![scores_update_every_30][scores_update_every_30]

```
Episode 100 	Average Score: 0.00 	Elapsed Time: 71.47 sec
Episode 200 	Average Score: 1.88 	Elapsed Time: 153.65 sec
Episode 300 	Average Score: 3.25 	Elapsed Time: 239.03 sec
Episode 400 	Average Score: 5.90 	Elapsed Time: 324.27 sec
Episode 500 	Average Score: 8.20 	Elapsed Time: 410.68 sec
Episode 600 	Average Score: 9.66 	Elapsed Time: 497.09 sec
Episode 700 	Average Score: 8.93 	Elapsed Time: 585.84 sec
Episode 800 	Average Score: 10.20	Elapsed Time: 673.00 sec
Episode 900 	Average Score: 9.70 	Elapsed Time: 755.02 secc
Episode 1000	Average Score: 10.04	Elapsed Time: 837.07 sec
Episode 1097	Average Score: 13.05	Elapsed Time: 917.58 sec
Environment solved in 997 episodes! 	Average Score: 13.05	Elapsed Time: 917.58 sec
Episode 1100	Average Score: 12.84	Elapsed Time: 920.16 sec
Episode 1200	Average Score: 13.13	Elapsed Time: 1002.85 sec
Episode 1300	Average Score: 13.82	Elapsed Time: 1086.08 sec
Episode 1400	Average Score: 14.36	Elapsed Time: 1169.82 sec
Episode 1447	Average Score: 15.03	Elapsed Time: 1210.50 sec
Environment solved in 1347 episodes!	Average Score: 15.03	Elapsed Time: 1210.50 sec
Episode 1500	Average Score: 15.27	Elapsed Time: 1255.53 sec
Episode 1600	Average Score: 15.60	Elapsed Time: 1340.94 sec
Episode 1700	Average Score: 15.60	Elapsed Time: 1428.46 sec
Episode 1800	Average Score: 15.31	Elapsed Time: 1515.74 sec
Episode 1900	Average Score: 14.81	Elapsed Time: 1604.57 sec
Episode 2000	Average Score: 15.70	Elapsed Time: 1690.22 sec
```

## Agent Performance

Examples of how various checkpoints of the agent performs under different scenarios are found in the [videos](./videos) directory.

### Checkpoint at 479 Episodes

The checkpoint which was attained by saving the model weights at episode 479 corresponds to the model which is able to attain at least a score of 13 over a rolling window of 100 episodes.  While this model and agent solved the environment, the behavior of the agent was fairly jittery as shown in [banana_13_jittery_performance_480p.mov](./videos/banana_13_jittery_performance_480p.mov).

### Checkpoint at 815 Episodes

The checkpoint which was attained by saving the model weights at episode 815 corresponds to the model which is able to attain at least a score of 15 over a rolling window of 100 episodes.  While this model and agent solved the environment, and the bahavior is fairly smooth as seen in herethe behavior of the agent was fairly jittery as shown in [banana_15_nice_performance_480p.mov](videos/banana_15_nice_performance_480p.mov).  This video was obtained with a random seed of the environment of `1`.  

It is curious, however, that when the random seed is the default random seed, that the agent gets stuck in a limit cycle between a blue banana and the wall.  In this case, the agent turns right to avoid the blue banana, but then turns toward the wall and then turns left to avoid the wall which puts the agent back facing the blue banana again.  This type of behavior likely explains the wide variation in the scores which which were attained by the model.  This behavior can be seen in [banana_15_limit_cycle_480p.mov](./videos/banana_15_limit_cycle_480p.mov).

### Checkpoint at 2000 Episodes

The checkpoint which was attained by saving the model weights at episode 2000 was observed as well.  Note that this agent's performance is numerically nearly identical to the previous agent checkpoint at 815 episodes; however, the behavior is once again slightly more jittery than the previous agent as shown in [banana_final_480p.mov](./videos/banana_final_480p.mov). 

## Ideas for Future Work

In the future, it would be good to investigate different models with several different features.  Suggestions include:

* Provide an outer loop watch-dog on the agent to identify limit cycle behavior.  We could identify this in training and then provide an additional penalty to the agent if a limit cycle is detected.  If we do this, then we likely would not need to write any code to explicitly tell the agent how to break out of a limit cycle, it would just figure out how not to enter a limit cycle since it would get penalties if that type of a state were encountered in training. Doing this will eliminate one behavior which leads to low scores.  After fixing this issue, I would suggest evaluating the other low score methods to determine if there are any other penalties we could add to influence the agent and encourage it to avoid sub-optimal behaviors.
* Try different numbers of units for hidden layers.  What is the minimum number of hidden units / network size to attain desired performance and solve the environment?  If instead of expanding the number of dimensions in the hidden layers, could we contract them?  What type of impact would this have on training time and accuracy?  Can we attain desireable performance in this way?
* Try a soft-max activation function on the last output layer to provide a consistent relative importance of each of the actions to the output state.  This may or may not affect performance, but would provide a more consistent output for comparing each of the actions.
* Try learning from pixels.  This may present more oportunities since we could learn directly from pixels what to do.  Likely this will take much longer to train.
* Build an image recognition model to identify location of and discriminate between yellow an blue bananas and then train the agent to respond to location of all bananas int he scene instead of the bananas which intersect a particular ray eminating from the agent's location.
* Investigate if there is an optimal number of steps between retrainings of the target network.  In this exercise, we tested both 4 and 30.  30 steps between target model retraining resulted in slower convergence to the desired performance metric.  It would be good to investigate what the trade-offs in how often the target model is updated with the speed of training and stability of training.
