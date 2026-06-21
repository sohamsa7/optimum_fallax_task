# optimum_fallax_task
An educational repository with primary intention of demonstrating reward hacking for the mentorship programme of ProjectX, VJTI.
'test.py' : Initial Code for testing purposes. The above code is just for testing purposes, it only uses gymnasium to ‘make’ the environment and uses pygame for rendering the 
game. 
It then selects an action out of the two in the ‘action sample space’ (either move right or left), performs an action and calculates the score based on the reward it gets. The CartPole never really learns anything as this is just for testing purposes. 
'train_original.py' : Training the model (PPO)
Now we will train the PPO model.
Initialise the PPO model with the name ‘model’
‘MlpPolicy’ means Multi-layer Perceptron policy that takes ‘obs’ as the input and outputs the probabilities of the cart moving either to the right or left.
‘obs’ is a variable that contains information of the cartpole at every timestep, i.e. 
obs=[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
verbose=1 indicates that we also have to print information like fps, loss etc.
In the above code, the model internally learns the cartpole movement in 100000 timesteps where each timestep usually has unique 
values of co-ordinates of the cartpole agent (obs variable). We later save it as “ppo-cartpole”
'test_ppo.py' : Testing the trained cartpole:
In this we actually test the model that we have already trained in the previous code. In this, we calculate the score of each episode based on the total rewards it accumulates.
An episode is the total timesteps it takes for the agent (cartpole) to reach the terminating condition. This can be either that the cart travels further away from the center or the pole angle exceeds a certain value. 
In this case, the average score being in between 475-500 is considered a good score.
IMPORTANT: The above code that we have studied till now only trained the PPO model, used this saved & trained PPO model used to calculate cumulative rewards. But we also have to plot the cumulative reward over episodes and also demonstrate reward hacking.
For this we use rewardcallback. Call back is a function that the PPO repeatedly calls during the training period for storing the information. This is important and the only new thing as compared to the above code as we need to also record the episodes and their respective rewards accumulated.
'train_position.py' : this is the 1st part of reward hacking. We simply reward the agent more for moving further away from the center. The agent may accumulate maximum reward while failing to achieve our objective.
'train_angle.py' : in this we do the same as above, but reward more when the pole leans further away.
