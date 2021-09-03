from absl import logging as absl_logging
from collections import defaultdict
from functools import partial
import logging
import numpy as np
import os
import ray
import reverb
import tensorflow as tf
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.experimental.distributed import reverb_variable_container, ReverbVariableContainer
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import reverb_replay_buffer, ReverbReplayBuffer
from tf_agents.specs import tensor_spec
from tf_agents.train.actor import Actor
from tf_agents.train.learner import Learner
from tf_agents.train.triggers import PolicySavedModelTrigger
from tf_agents.train.utils import train_utils, spec_utils, strategy_utils
from utilities import DuplicateLogFilter, StopWatch
from constructors import env_ctor, agent_ctor, train_observers_ctor, eval_observers_ctor, actor_ctor
from workers import TrainWorker, EvalWorker

osp = os.path


def list2dict(m: list) -> dict:
  d = defaultdict(list)
  for mm in m:
    for k,v in mm.items():
      d[k].append(v)
  return d


def combine_train_metrics(m: list):
  d = list2dict(m)
  d = dict(reward=np.mean(d['reward']), episode_length=np.mean(d['episode_length']), steps=np.sum(d['steps']),
           episodes=np.sum(d['episodes']))
  return d
  

class Trainer:
  def __init__(self):
    self.logger = logging.getLogger(__name__)
    self.absl_log_filter = DuplicateLogFilter(absl_logging.get_absl_logger())
    self.tf_log_filter = DuplicateLogFilter(tf.get_logger())
    self.tb_writer = tf.summary.create_file_writer('output', flush_millis=10*1000)

  
  def log_metrics(self, metrics: dict, prefix: str):
    s = ', '.join([f'{name}={metric:.4f}' for name, metric in metrics.items()])
    self.logger.info(f'{prefix}: {s}')

  
  def summarize_metrics(self, metrics: dict, step: int, prefix: str):
    with self.tb_writer.as_default():
      for name, metric in metrics.items():
        tf.summary.scalar(name=f'{prefix}/{name}', data=metric, step=step)


  def train(self, use_tf_functions=True):
    # hyper parameters
    train_subepisode_length = 2
    train_batch_size = 256
    prefetch_batches = 2
    n_sgd_steps = 50
    buffer_size = 10000
    buffer_init_episodes = 500
    n_workers = 1
    n_iters = 15000

    # env for seeding the replay buffer
    ss = np.random.SeedSequence()
    seed = ss.spawn(1)[0].generate_state(1)[0]
    env = env_ctor(seed=seed)
    obs_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)

    # agent
    strategy = strategy_utils.get_strategy(False, False)
    with strategy.scope():
      train_step = train_utils.create_train_step()
      agent = agent_ctor(train_step, obs_spec, action_spec, time_step_spec)
      agent.initialize()
    
    # policy weight variables and signature
    variables = {
      reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
    }
    variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype), variables)
    self.logger.info(f'Signature of variables: \n{variable_container_signature}')

    # experience signature
    experience_signature = tensor_spec.from_spec(agent.collect_policy.collect_data_spec)
    self.logger.info(f'Signature of experience: \n{experience_signature}')

    # Create and start the replay buffer and variable container server.
    exp_table = reverb.Table(name=reverb_replay_buffer.DEFAULT_TABLE,
                             max_size=env.horizon*buffer_size, sampler=reverb.selectors.Uniform(),
                             remover=reverb.selectors.Fifo(), rate_limiter=reverb.rate_limiters.MinSize(1),
                             signature=experience_signature)
    policy_table = reverb.Table(name=reverb_variable_container.DEFAULT_TABLE, sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(), rate_limiter=reverb.rate_limiters.MinSize(1),
                                max_size=1, max_times_sampled=0, signature=variable_container_signature)
    reverb_server = reverb.Server([exp_table, policy_table])
    # reverb_server.wait()

    # container holding policy weights
    variable_container = ReverbVariableContainer(f'localhost:{reverb_server.port}',
                                                 table_names=[reverb_variable_container.DEFAULT_TABLE])
    variable_container.push(variables)

    # replay buffer
    replay_buffer = ReverbReplayBuffer(agent.collect_data_spec, sequence_length=train_subepisode_length,
                                       table_name=reverb_replay_buffer.DEFAULT_TABLE, local_server=reverb_server,
                                       dataset_buffer_size=prefetch_batches*train_batch_size,
                                       num_workers_per_iterator=1)

    # seed replay buffer
    self.logger.info('*** Initial random action collection...')
    observers: dict = train_observers_ctor(reverb_server.port, collect_only=True)
    policy = PyTFEagerPolicy(RandomTFPolicy(time_step_spec, action_spec), use_tf_function=use_tf_functions)
    actor = Actor(env, policy, train_step, episodes_per_run=buffer_init_episodes, observers=observers.values())
    actor.run()
    observers['replay_buffer'].close()
    self.logger.info('*** Done.')
    
    # learner
    def experience_dataset_fn():
      with strategy.scope():
        def _filter_invalid_transition(trajectories, unused_arg1):
          return tf.reduce_all(~trajectories.is_boundary()[:-1])
        dataset = replay_buffer.as_dataset(sample_batch_size=train_batch_size, num_steps=train_subepisode_length)
        dataset = dataset.unbatch().filter(_filter_invalid_transition).batch(train_batch_size)
        dataset = dataset.prefetch(prefetch_batches)
        return dataset
    interval = n_sgd_steps * 10
    save_model_trigger = PolicySavedModelTrigger(osp.join('output', 'policy'), agent, train_step,
                                                 interval=interval, save_greedy_policy=True)
    learner = Learner('output', train_step, agent, experience_dataset_fn, triggers=[save_model_trigger, ],
                      checkpoint_interval=interval, summary_interval=n_sgd_steps*10, max_checkpoints_to_keep=2,
                      strategy=strategy)
    
    # workers
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_workers+1)]
    RemoteTrainWorker = ray.remote(TrainWorker)
    RemoteEvalWorker = ray.remote(EvalWorker)
    train_workers = [
      RemoteTrainWorker.remote(i, partial(env_ctor, seed=seed), actor_ctor,
                               partial(train_observers_ctor, reverb_port=reverb_server.port), reverb_server.port,
                               use_tf_functions)
        for i, seed in enumerate(seeds[:-1])
    ]
    eval_worker = RemoteEvalWorker.remote(0, partial(env_ctor, seed=seeds[-1]), actor_ctor, eval_observers_ctor,
                                          reverb_server.port, use_tf_functions)

    # loop
    policy_pull_ids = [w.pull_policy.remote() for w in train_workers]
    policy_pull_ids.append(eval_worker.pull_policy.remote())
    collect_timer = StopWatch(avg_mode=False)
    train_timer = StopWatch()
    train_run_infos = ray.get([w.get_observer_values.remote() for w in train_workers])
    env_steps = np.array([m['steps'] for m in train_run_infos])
    best_eval_return = -1000.0
    N = n_sgd_steps * n_iters
    while train_step.numpy() < N:
      itr = train_step.numpy() // n_sgd_steps
      do_eval = itr % 10 == 0
      do_summary = itr % 10 == 0

      # sync policy
      ray.get(policy_pull_ids)
      
      # collect
      collect_timer.start(env_steps)
      run_info_ids = [w.run.remote() for w in train_workers]
      if do_eval:
        run_info_ids.append(eval_worker.run.remote())
      
      # update
      train_timer.start(0)
      losses: LossInfo = learner.run(iterations=n_sgd_steps)
      train_speed, train_time = train_timer.stop(n_sgd_steps)
      # self.logger.info(f'Training done in {train_time:.3f} s @ {train_speed:.3f} steps/s')
      
      # sync
      train_run_infos = ray.get(run_info_ids)
      if do_eval:
        train_run_infos, eval_run_info = train_run_infos[:-1], train_run_infos[-1]
      env_steps = np.array([m['steps'] for m in train_run_infos])
      collect_time = np.max([m['time'] for m in train_run_infos])
      collect_speed, collect_time = collect_timer.stop(env_steps, override_time=collect_time)
      # self.logger.info(f'Collection done in {collect_time:.3f} s @ {collect_speed:.3f} steps/s')
      train_run_infos = combine_train_metrics(train_run_infos)
      train_run_infos = dict(train_run_infos, collect_speed=collect_speed, train_speed=train_speed,
                             loss=losses[0].numpy())

      # log and summarize
      self.log_metrics(train_run_infos, f'Train {(itr+1):03d}/{n_iters:03d}')
      if do_summary:
        self.summarize_metrics(train_run_infos, itr, 'train')
      if do_eval:
        self.log_metrics(eval_run_info, f'Eval {(itr+1):03d}/{n_iters:03d}')
        self.summarize_metrics(eval_run_info, itr, 'eval')

      # eval
      if do_eval:
        if eval_run_info['reward'] > best_eval_return:
          best_eval_return = eval_run_info['reward']
          self.logger.info(f'New best eval return {best_eval_return:.4f} at iteration {itr}')
          with open(osp.join('output', 'best_itr.txt'), 'w') as f:
            f.write(f'{itr+1} / {n_iters}\n')

      # push policy
      variable_container.push(variables)
      policy_pull_ids = [w.pull_policy.remote() for w in train_workers]
      policy_pull_ids.append(eval_worker.pull_policy.remote())
    
    # flush and close
    self.tb_writer.flush()
    close_ids = [worker.close.remote() for worker in train_workers]
    close_ids.append(eval_worker.close.remote())
    ray.get(close_ids)
    env.close()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  ray.init(include_dashboard=False)
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(1)
  gpus = tf.config.list_physical_devices('GPU')
  if len(gpus) > 0:
    tf.config.set_visible_devices([], 'GPU')
  tf.compat.v1.enable_v2_behavior()
  trainer = Trainer()
  trainer.train()