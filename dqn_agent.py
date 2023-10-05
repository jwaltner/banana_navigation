import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=32):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    # save the latest step to memory
    # learn if we are at a learning interval
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # set to evaluation mode so that weights are not updated
        # and back-prop is not performed or gradients calculated.
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        
        #==
        # Get max predicted Q values (for next states) from target model
        #==
        # detatch()
        #   https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
        #   gets a tensor which is detached from the graph.  This will not be 
        #   ever used for back-prop or gradients, it is just to get the output of 
        #   the network.
        # max(1)
        #   retrieves the maximum value across dimension 1.  This gets the value 
        #   of the maximum action for each row since the action-values are in the 
        #   nested dimension
        # [0] or .values
        #   max(1) returns a max data object which has a tensor of the values of 
        #   the maximums over the requested axis as well as the indicies of these 
        #   maximum values. you can access the values of the max with .max(1)[0]
        #   or .max(1).values.  Similarly, you can access the values of the indicies
        #   of the maximum value using .max(1)[1] or .max(1).indices
        #     torch.return_types.max(
        #     values=tensor([0.1016, 0.1145]),
        #     indices=tensor([0, 0]))
        # unsqueeze(1)
        #   since we now have a list of the maximum action-values, we need to put 
        #   it back into a shape that works with the other tensors, therefore, we need 
        #   to convert it back into a 2D tensor from its current 1D tensor.  We do this
        #   with .unsqueeze(1) to insert a dimension at the leaf level so that we end 
        #   up with a 2D tensor which is in essence an array of 1-element arrays instead 
        #   of a 1D tensor of scalars.
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1).values.unsqueeze(1)
        
        #==
        # Compute Q targets for current states 
        #==
        # add the current rewards to the discounted rewards for the optimal actions
        # given the current target network.  This is why we needed to reshape it with 
        # unsqueeze above so that our Q_targets_next will be in a compatible shape with
        # requards and dones.  Note if next state is done, then, we do not give any 
        # discounted reward.
        #
        # Important!!! both Q_targets and Q_expected are wrong... and coming up, we try
        # to push the Q_expected into the direction of the Q_target since we calculated
        # Q_targets based off the best next action with the present reward...  so we are 
        # pushing something which is wrong into the direction of something wrong... 
        # However, eventhough Q_targets is wrong, it is kind of right since we injected 
        # some reality into the equation by using real rewards for our current timestep.
        # this injection of reality will over time allow the Q_targets to converge to 
        # the optimal action value function.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        #==
        # Get expected Q values from local model
        #==
        # note here we use the real actions taken rather than the actions that we 
        # are projecting which should be taken on the next step which could be obtained
        # from self.qnetwork_target(next_states).detach().max(1).indicies.unsqueeze(1).
        # we want the actual actions for this step since we want to know what our network
        # is presently telling us to do for this step.
        #
        # Here, we calculate the action values for all states, then we use the .gather()
        # method to select from dimension==1 the actions which were actually taken.
        # in this way, we wind up with an expected value for each state using our 
        # local qnetwork given the action which was taken.  This will differ from the 
        # value which is calculated from the actual rewards and the discounted best 
        # next action using our fixed target.
        #
        # During the training we will then attempt to push the local network into the 
        # direction of a better answer which is the actual rewards and the best next action
        # from the fixed Q target.  This provides both stability and injects reality into 
        # the system.
        #
        # IMPORTANT!!! Here we do not use .disconnect() since we want to retain the 
        # gradients for qnetwork_local.  if we use .disconnect() the gradients are lost
        # and then the training can not occur.
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #==
        # Compute loss
        #==
        # using mean square error to compute the loss here.
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)