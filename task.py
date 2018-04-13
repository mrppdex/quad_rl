import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        self.action_dim = 4
        self.observation_dim = self.sim.pose.shape[0]
        self.action_high = 900
        self.action_low = 0
        
        self.action_repeat = 1

        self.state_size = self.action_repeat * self.observation_dim
        
        self.action_size = self.action_dim

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.total_rewards = 0.

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #print("entered get_reward")
        #print("self.sim.pose[:3]={}".format(self.sim.pose[:3]))
        #reward = -np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        reward = min(0., -(10. - self.sim.pose[2]))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rewards = 0
        pose_all = []
        dones = False
        for _ in range(self.action_repeat):
            #observation, reward, done, _ = self.sim.step(rotor_speeds) 
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            rewards += self.get_reward()
            pose_all.append(self.sim.pose)
            dones |= done
        next_state = np.concatenate(pose_all)
        self.total_rewards += rewards
        return next_state, rewards, dones

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        self.total_rewards = 0.
        init_states = []
        for _ in range(self.action_repeat):
            init_states.append([0., 0., np.random.uniform(0.1, 0.2), 0., 0., 0.])
        state = np.concatenate(init_states)
        #state = self.sim.pose
        return state