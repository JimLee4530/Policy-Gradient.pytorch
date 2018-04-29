from agent_dir.agent import Agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import *
import os

LEARNING_RATE = 1e-4
NUM_EPISODES = 100000
MAX_NUM_STEPS = 10000

def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)



        ##################
        # YOUR CODE HERE #
        ##################
        self.gamma = args.gamma
        self.state_dim = [80, 80, 1]
        self.reward_history = []

        self.lr = LEARNING_RATE

        self.pg_brain = net()
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.pg_brain.load_state_dict(torch.load(os.path.join('save_dir/' 'model-best.pth')))
            self.pg_brain.eval()
        self.pg_brain.cuda()

        self.optimizer = optim.RMSprop(self.pg_brain.parameters(), lr=self.lr )


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.test_last_observation = self.env.reset()
        self.test_last_observation = prepro(self.test_last_observation)

    def train(self):
        render = False
        prev_x = None  # used in computing the difference frame
        observation = self.env.reset()
        xs,ys, dlogps, drs = [], [], [],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        self.pg_brain.train()
        while True:
            if render: self.env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(80*80)
            prev_x = cur_x

            # forward the policy network and sample an action from the returned probability

            aprob = self.predict_action(x).data.cpu().numpy()[0]
            action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

            # record various intermediates (needed later for backprop)
            xs.append(x)  # observation
            y = 1 if action == 2 else 0  # a "fake label"
            ys.append(y)

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action)
            reward_sum += reward
            drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

            if done:  # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                epr = np.vstack(drs)
                epy = np.vstack(ys)
                xs,ys,drs = [], [], [] # reset array memory


                # compute the discounted reward backwards through time
                discounted_epr = self.discount_rewards(epr, self.gamma)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                tmp = [epx, epy, discounted_epr]
                tmp = [Variable(torch.from_numpy(np.asarray(_)), requires_grad=False).cuda().float() for _ in tmp]
                states, action, rewards = tmp
                self.optimizer.zero_grad()

                probability = self.pg_brain(states)
                # loss = torch.sum((action - probability) * rewards) / epy.shape[0]
                loss = F.binary_cross_entropy(probability, action, rewards)
                loss.backward()
                self.optimizer.step()


                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                if episode_number % 100 == 0:
                    checkpoint_path = os.path.join('save_dir', 'model-best.pth')
                    torch.save(self.pg_brain.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                reward_sum = 0
                observation = self.env.reset()  # reset env
                prev_x = None

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if test:
            self.pg_brain.eval()
        observation = prepro(observation)
        observation_delta = observation - self.test_last_observation
        self.test_last_observation = observation

        up_prob = self.predict_action(observation_delta).data.cpu().numpy()[0]
        action = 2 if np.random.uniform() < up_prob else 3
        return action

    # each step sum reward for all steps which it has influenced
    def discount_rewards(self, rewards, discount_factor):
        discount_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * discount_factor + rewards[t]
            discount_rewards[t] = running_add
        return discount_rewards

    def predict_action(self, observation):
        # observation = observation.transpose((2, 0, 1))
        observation = observation[np.newaxis, :]
        observation=Variable(torch.from_numpy(observation), requires_grad=False).cuda().float()
        probability = self.pg_brain(observation)# replace with torch network
        return probability

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.fc1 = nn.Linear(6400, 200)
        self.fc2 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.m = nn.Sigmoid()
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)


    def forward(self, observations):
        fc1 = self.relu(self.fc1(observations))
        pred = self.m(self.fc2(fc1))

        return pred

