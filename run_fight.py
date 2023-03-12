# from stable_baselines import PPO2, TRPO
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


from stable_baselines3.common.env_util import make_vec_env #INTERESTING THIS ALLOWS VECTORIZED SO I DONT HAVE TO USE THAT CRAZY LAMBDA FUNCTION

from torille import envs
import gym
import random
from argparse import ArgumentParser
import tensorflow as tf
import time

parser = ArgumentParser("Run stable-baselines on torille")
parser.add_argument("env")
parser.add_argument("agent", choices=["ppo"])
parser.add_argument("experiment_name")
parser.add_argument("--timesteps", type=int, default=int(3 * 1e6))
parser.add_argument("--randomize_engagement", action="store_true")
parser.add_argument("--turnframes", type=int, default=5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--steps_per_batch", type=int, default=1024)
parser.add_argument("--num_envs", type=int, default=1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device) # Print the device, should show 'cuda' if GPU is available.

# torch.backends.cudnn.benchmark = True # Enable cuDNN auto-tuner to find the best algorithm to use for your hardware.


import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Using GPU...")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Using CPU...")




class TorilleWrapper(gym.Wrapper): #looks like this unfucks toribash
    """ Ad-hoc wrapper for many things with torille """
    def __init__(self, env, record_every_episode, record_name, randomize_settings, **kwargs):
        super().__init__(env)

        self.record_every_episode = 5
        self.record_name = record_name
        self.randomize_settings = randomize_settings
        self.num_episodes = 0

    def step(self, action):
        # Fix info being None -> info = {}
        obs, reward, done,  _ = self.env.step(action) #need truncated
        return obs, reward, done, {}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.num_episodes += 1

        # Ad-hoc settings for destroyuke
        self.env.settings.set("custom_settings", 1)
        for key,values in self.randomize_settings.items():
            self.env.settings.set(key, random.randint(*values))

        if self.num_episodes % 100 == 0: #every 100 episodes...
            self.env.settings.set("replay_file", "%s_%d" % (self.record_name, self.num_episodes))
        
        return obs

class args:
    env = 'Toribash-DestroyUke-v1'
    agent = 'ppo'
    experiment_name = 'Operation Bruce Three'
    timesteps = 180000 #int(3 * 1e6)  #roughly 5.55 of these is a second . 60000 is 20 minutes
    randomize_engagement = True
    turnframes = 10
    ent_coef = 0.01
    steps_per_batch = 1024
    num_envs = 1

def run_experiment(args):
    
    randomization_settings = {
        "engagement_distance": (100,100),
        "turnframes": (args.turnframes, args.turnframes)
    }

    if args.randomize_engagement: 
        randomization_settings["engagement_distance"] = (100, 200) #interesting
    
    vecEnv = None
    if args.num_envs == 1:
        # Create dummyvecenv
        env = gym.make(args.env)
        env.set_draw_game(False)
        env = Monitor(TorilleWrapper(env, 100, args.experiment_name, randomization_settings), args.experiment_name)
        env.set_draw_game(False)

        vecEnv = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run           #fuck does that mean? does this unfuck the crazy toribash variables?
        # vecEnv.set_draw_game(False)
    else:
        vecEnv = []
        
        def make_env():
            env = gym.make(args.env)
            unique_id = str(time.time())[-6:]
            experiment_env_name = args.experiment_name + ("_env%s" % unique_id)
            return Monitor(TorilleWrapper(env, 100, experiment_env_name, randomization_settings), 
                           experiment_env_name)
        
        for i in range(args.num_envs):
            vecEnv.append(make_env)
        
        vecEnv = SubprocVecEnv(vecEnv)


        

    steps_per_env = args.steps_per_batch // args.num_envs



    # Standard 2 x 64 network with sigmoid activations
    # policy_kwargs = dict( net_arch=[64, 64, 64]) #disgusting. a fucking sigmoid structure. what the actual fuck. change that to elu. sigmoid jesus christ.
    # policy_kwargs = dict(net_arch=[64, 64, dict(vf=[256, 128], pi=[256, 128], activation_fn=tf.nn.elu)])
    # policy_kwargs = dict(net_arch=[dict(vf=[256]*5, pi=[256]*5, activation_fn=tf.nn.elu)])
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256, 256, 256], vf=[256, 256, 256, 256, 256], activation_fn=tf.nn.elu))




    model = None
    print("here")
    if args.agent == "ppo":
        model = PPO("MlpPolicy", 
                    vecEnv, 
                    policy_kwargs=policy_kwargs, 
                    ent_coef=args.ent_coef, 
                    n_steps=steps_per_env,
                    verbose=1,
                    device="cuda:0")
    
    


    # elif args.agent == "trpo":
    #     model = TRPO(MlpPolicy, vecEnv, policy_kwargs=policy_kwargs, 
    #                  entcoeff=args.ent_coef, timesteps_per_batch=steps_per_env,
    #                  verbose=1)

    # model=PPO.load("DEATH BOT.zip", vecEnv)

    model.learn(total_timesteps=args.timesteps)

    model.save("DEATH BOT2.zip")


if __name__ == "__main__":

    run_experiment(args) 
    print("here")