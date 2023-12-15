import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, activation: str = 'ReLU'):
        super(NeuralNetwork, self).__init__()

        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        layers = []
        
        # The first layer we add is the from input, the rest is with hidden_size
        num_inputs = input_dim
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(num_inputs, hidden_size),
                self.activation
            ])
            num_inputs = hidden_size
        layers.append(nn.Linear(num_inputs, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.network(s)


class Actor:
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.network = NeuralNetwork(self.state_dim, 2 * self.action_dim, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)

    def forward(self, state):
        '''
        This function runs the neural network above and extracts the mean and clamped log_std
        '''
        result = self.network.forward(state)
        mean, log_std = result.split(1, dim=-1)
        return mean, self.clamp_log_std(log_std)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        action: torch.Tensor, action the policy returns for the state.
        log_prob: log_probability of the action.
        '''
        
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # If working with stochastic policies, make sure that its log_std are clamped
        # using the clamp_log_std function.
        action_mean, clamped_log_std = self.forward(state)
        z = torch.randn_like(action_mean)
        std = torch.exp(clamped_log_std)
        
        random_action = action_mean + std * z
        # TODO, we have the formual form. 
        log_prob = random_action - (2 * (np.log(2) - random_action - F.softplus(-2 * random_action)))
        
        action = torch.tanh(action_mean) if deterministic else torch.tanh(random_action)
        #log_prob = Normal(action_mean, std).log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)

        log_prob = torch.as_tensor([1]) if deterministic else log_prob
        assert action.shape == (self.action_dim,) and \
               log_prob.shape == (self.action_dim,) or action.shape == (state.shape[0], 1) \
               and log_prob.shape == (state.shape[0], 1),  'Incorrect shape for action or log_prob.'
        return action, log_prob

class Critic:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, critic_lr: float, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        self.network = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.critic_lr)

    def forward(self, state_action):
        return self.network.forward(state_action)

class Value:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, value_lr: float, state_dim: int = 3, device: torch.device = torch.device('cpu')):
        super(Value, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.value_lr = value_lr
        self.state_dim = state_dim
        self.device = device
        self.setup_value()

    def setup_value(self):
        self.network = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers)
        self.value_target_net = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers)
        self.value_target_net.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.value_lr)

    def set_target(self, new_value_target_net):
        self.value_target_net = new_value_target_net

    def forward(self, state):
        return self.network.forward(state)

    def forward_target(self, state):
        return self.value_target_net.forward(state)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''

    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)
        self.loss = torch.as_tensor([0])

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

    # Something that is like run_gradient_update_step for Agent
    def update_param(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training,
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        # Feel free to instantiate any other parameters you feel you might need.
        self.hidden_layers = 128 #128
        self.hidden_size = 3 #3
        self.lr = 0.001 #0.001
        self.actor = Actor(self.hidden_layers, self.hidden_size, self.lr)
        self.value = Value(self.hidden_layers, self.hidden_size, self.lr)
        self.critics = [Critic(self.hidden_layers, self.hidden_size, self.lr),
                        Critic(self.hidden_layers, self.hidden_size, self.lr)]
        self.mse_criterion = nn.MSELoss()

        
        self.tau = 0.01 # 0.01 / 0.005
        self.gamma = 0.98 # 0.98 / 0.99
        initial_temperature = 0.25 #0.25
        temperature_learning_rate = 0.0005 # 0.0005
        self.temperature_parameter = TrainableParameter(
            init_param=initial_temperature,
            lr_param=temperature_learning_rate,
            train_param=True
        )
        self.entropy = 1

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray, action to apply on the environment, shape (1,)
        """
        state = torch.as_tensor(s, dtype=torch.float32).to(self.device)
        determinstic_sampling_flag = not train
        action, _ = self.actor.get_action_and_log_prob(state, determinstic_sampling_flag)

        action = action.detach().cpu().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic, Value], loss: torch.Tensor, clip_value: float = None):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        if clip_value:
            torch.nn.utils.clip_grad_norm_(object.network.parameters(), clip_value)
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer, and then updates the policy and critic networks
        using the sampled batch.
        '''
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        state_action_batch = torch.cat([s_batch, a_batch], dim=1)
        
        temperature = self.temperature_parameter.get_param()
        
        # Get values from neural networks
        predicted_value = self.value.forward(s_batch)
        predict_critic_0 = self.critics[0].forward(state_action_batch)
        predict_critic_1 = self.critics[1].forward(state_action_batch)
        deterministic_sampling = False
        guessed_action, log_prob = self.actor.get_action_and_log_prob(s_batch, deterministic_sampling)
        
        
        # Training the critic functions
        target_v = self.value.value_target_net(s_prime_batch)
        target_q = r_batch + self.gamma * target_v
        criterion = nn.MSELoss()
        loss0 = criterion(predict_critic_0, target_q.detach())
        self.run_gradient_update_step(self.critics[0], loss0)
        
        criterion = nn.MSELoss()
        loss1 = criterion(predict_critic_1, target_q.detach())
        self.run_gradient_update_step(self.critics[1], loss1)
        
        
        # Training value function
        state_action_batch = torch.cat([s_batch, guessed_action], dim=1)
        q_predictions = [critic.forward(state_action_batch) for critic in self.critics]
        min_q = torch.min(q_predictions[0], q_predictions[1])
        target_v = min_q - temperature * log_prob
        value_loss = self.mse_criterion(predicted_value, target_v.detach())
        self.run_gradient_update_step(self.value, value_loss)
         
        
       # Training Agent
        actor_loss = torch.mean(temperature * log_prob - min_q)
        self.run_gradient_update_step(self.actor, actor_loss)

        value_net = self.value.network
        target_net = self.value.value_target_net

        soft_update = True
        self.critic_target_update(value_net, target_net, self.tau, soft_update)
        
        # Update temperature parameters TODO
        log_prob_clone = log_prob.clone().detach()
        loss = torch.mean(temperature * self.entropy - temperature * log_prob_clone)
        self.temperature_parameter.update_param(loss)
        
        

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()