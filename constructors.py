from random_env import RandomEnvironment
import reverb
import tensorflow as tf

from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac.sac_agent import SacAgent, std_clip_transform
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from tf_agents.replay_buffers.reverb_replay_buffer import DEFAULT_TABLE
from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
from tf_agents.train.actor import collect_metrics, eval_metrics

def env_ctor(seed=7):
  env = RandomEnvironment(seed=seed)
  return env

  
def actor_ctor(observation_spec, action_spec):
  def normal_projection_net(action_spec, init_means_output_factor=0.5):
    return NormalProjectionNetwork(action_spec, mean_transform=None, state_dependent_std=True,
                                    init_means_output_factor=init_means_output_factor, std_transform=std_clip_transform,
                                    scale_distribution=True)
  actor_net = ActorDistributionNetwork(observation_spec, action_spec, fc_layer_params=(256, 256),
                                       continuous_projection_net=normal_projection_net)
  return actor_net


def critic_ctor(observation_spec, action_spec):
  critic_net = CriticNetwork((observation_spec, action_spec), observation_fc_layer_params=None,
                             action_fc_layer_params=None, joint_fc_layer_params=(256, 256))
  return critic_net


def train_observers_ctor(reverb_port, collect_only=False) -> dict:
  reverb_client = reverb.Client(f'localhost:{reverb_port}')
  observers = dict(replay_buffer=ReverbAddTrajectoryObserver(reverb_client, table_name=DEFAULT_TABLE,
                                                              sequence_length=2, stride_length=1))
  if collect_only:
    return observers
  collect_obs = collect_metrics(10)
  observers = dict(observers, reward=collect_obs[2], episode_length=collect_obs[3], steps=collect_obs[1],
                    episodes=collect_obs[0])
  return observers


def eval_observers_ctor() -> dict:
  eval_obs = eval_metrics(25)
  observers = dict(reward=eval_obs[0], episode_length=eval_obs[1],)
  return observers


def agent_ctor(train_step, obs_spec, action_spec, time_step_spec):
  return SacAgent(
      time_step_spec,
      action_spec,
      actor_network=actor_ctor(obs_spec, action_spec),
      critic_network=critic_ctor(obs_spec, action_spec),
      actor_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
      critic_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
      alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      train_step_counter=train_step,
      initial_log_alpha=1.0,
  )
