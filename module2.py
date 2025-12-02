import gymnasium as gym
import numpy as np
import mujoco

class HumanoidEnvWrapper:
    """A wrapper for the Humanoid-v5 environment with custom reset and visuals."""
    def __init__(self, render_mode=None):
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self._customize_visuals()
        
        # Mapping from our 19-joint pose to the MuJoCo qpos indices
        # This mapping is crucial and depends on the MuJoCo model's joint order.
        self.qpos_mapping = {
            # Lower Body (8 joints)
            0: 3,  # right_hip_yaw -> qpos[3]
            1: 4,  # right_hip_pitch -> qpos[4]
            2: 5,  # right_knee -> qpos[5]
            3: 6,  # right_ankle -> qpos[6]
            4: 7,  # left_hip_yaw -> qpos[7]
            5: 8,  # left_hip_pitch -> qpos[8]
            6: 9,  # left_knee -> qpos[9]
            7: 10, # left_ankle -> qpos[10]
            # Upper Body (11 joints)
            8: 0,  # spine_pitch -> qpos[0]
            9: 1,  # neck_pitch -> qpos[1]
            10: 2, # neck_yaw -> qpos[2]
            11: 11, # left_shoulder_pitch -> qpos[11]
            12: 12, # left_shoulder_roll -> qpos[12]
            13: 13, # left_elbow -> qpos[13]
            14: 14, # left_wrist -> qpos[14]
            15: 15, # right_shoulder_pitch -> qpos[15]
            16: 16, # right_shoulder_roll -> qpos[16]
            17: 17, # right_elbow -> qpos[17]
            18: 18, # right_wrist -> qpos[18]
        }

    def _customize_visuals(self):
        """Sets robot color and floor appearance."""
        self._set_robot_color((0.2, 0.6, 1.0, 1.0))  # Blue robot
        self._set_floor_white()

    def _set_robot_color(self, color):
        model = self.env.unwrapped.model
        for i in range(model.ngeom):
            if model.geom_bodyid[i] != 0: # Skip worldbody (floor)
                model.geom_rgba[i] = color

    def _set_floor_white(self):
        model = self.env.unwrapped.model
        ground_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "ground")
        if ground_mat_id != -1:
            model.mat_rgba[ground_mat_id] = [1, 1, 1, 1] # White
            model.mat_texid[ground_mat_id] = -1 # Remove texture
        else: # Fallback
            for i in range(model.ngeom):
                if model.geom_bodyid[i] == 0 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    model.geom_rgba[i] = [1, 1, 1, 1]
                    break

    def reset(self, initial_pose=None):
        """Resets the environment, optionally to a specific initial pose."""
        obs, info = self.env.reset()
        if initial_pose is not None and len(initial_pose) == 19:
            qpos = self.env.unwrapped.data.qpos.copy()
            qvel = self.env.unwrapped.data.qvel.copy()
            
            for i, angle in enumerate(initial_pose):
                qpos_idx = self.qpos_mapping[i]
                qpos[qpos_idx] = angle
            
            self.env.unwrapped.set_state(qpos, qvel)
            obs = self.env.unwrapped._get_obs()
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def unwrapped(self):
        return self.env.unwrapped