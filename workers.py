from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.eval.metric_utils import compute
from tf_agents.experimental.distributed import reverb_variable_container, ReverbVariableContainer
from tf_agents.networks.network import DistributionNetwork
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.train.actor import Actor
from tf_agents.train.utils import train_utils
from tf_agents.train.utils import train_utils, spec_utils
import time


class Worker(object):
  def __init__(self, id, env_ctor, actor_ctor, observers_ctor, reverb_port):
    self.id = id
    self.env: TFPyEnvironment = env_ctor()
    self.obs_spec, self.action_spec, self.time_step_spec = spec_utils.get_tensor_specs(self.env)
    
    # TF policy that holds variables
    actor_net: DistributionNetwork = actor_ctor(self.obs_spec, self.action_spec)
    self.tf_policy = ActorPolicy(self.time_step_spec, self.action_spec, actor_net, training=False)
    
    self.observers: dict = observers_ctor()
    self.train_step = train_utils.create_train_step()
    self.variables = {
      reverb_variable_container.POLICY_KEY: self.tf_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: self.train_step
    }
    self.variable_container = ReverbVariableContainer(f'localhost:{reverb_port}',
                                                      table_names=[reverb_variable_container.DEFAULT_TABLE])

  def run(self):
    raise NotImplementedError

  def pull_policy(self):
    self.variable_container.update(self.variables)

  def get_observer_values(self, elapsed_time=0.0):
    d = {k: v.result() for k,v in self.observers.items() if k!='replay_buffer'}
    d['time'] = elapsed_time
    return d
  
  def close(self):
    return self.env.close()


class TrainWorker(Worker):
  def __init__(self, id, env_ctor, actor_ctor, observers_ctor, reverb_port, use_tf_function):
    super(TrainWorker, self).__init__(id, env_ctor, actor_ctor, observers_ctor, reverb_port)
    self.env.reset()
    py_policy = PyTFEagerPolicy(self.tf_policy, use_tf_function=use_tf_function)
    self.actor = Actor(self.env, py_policy, self.train_step, episodes_per_run=1, observers=self.observers.values())

  def run(self):
    start_time = time.time()
    self.actor.run()
    elapsed_time = time.time() - start_time
    return self.get_observer_values(elapsed_time)
  
  def close(self):
    self.observers['replay_buffer'].close()
    return super(TrainWorker, self).close()


class EvalWorker(Worker):
  def __init__(self, id, env_ctor, actor_ctor, observers_ctor, reverb_port, use_tf_function):
    super(EvalWorker, self).__init__(id, env_ctor, actor_ctor, observers_ctor, reverb_port)
    self.py_policy = PyTFEagerPolicy(GreedyPolicy(self.tf_policy), use_tf_function=use_tf_function)
    self.num_episodes = 5

  def run(self):
    start_time = time.time()
    compute(self.observers.values(), self.env, self.py_policy, num_episodes=self.num_episodes)
    elapsed_time = time.time() - start_time
    return self.get_observer_values(elapsed_time)