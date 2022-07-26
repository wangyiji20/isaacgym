from isaacgym.terrain_utils import *
from isaacgym.torch_utils import *
import numpy as np
import os
import random
import math
import sys
from isaacgym import gymtorch
from isaacgym.gymtorch import *
from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import *
from infrastructure.terrain import Terrain
sys.path.append('../rsl_rl')
from rsl_rl.env import VecEnv

class Xenv(VecEnv):
    def __init__(self, params):
        self.gym = gymapi.acquire_gym()
        self.params=params
        sim_params = gymapi.SimParams()
        # set common parameters
        sim_params.dt = 1 / 40
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 10
        sim_params.physx.solver_type = 1
        sim_params.physx.default_buffer_size_multiplier = 5
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.max_depenetration_velocity= 100.0
        sim_params.physx.contact_collection:0
        sim_params.physx.max_gpu_contact_pairs=2**24
        #create sim
        sim_params.use_gpu_pipeline = True
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        self.device='cuda:0'
        
        feet_names = ['FL_foot','FR_foot','RL_foot','RR_foot']
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.trunk_index=1
        
        #load asset
        asset_root = "../../assets"
        asset_file = "urdf/a1_description/urdf/a1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file)
        self.asset_name="a1"
        self.num_dofs=12
        self.num_actions=12
        self.num_rigidbody=23
        
        
        # attach a force sensor to each foot,the poses are irrelevant
        sensor_pose = gymapi.Transform()
        for index in range(len(feet_names)):
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False # for example gravity
            sensor_options.enable_constraint_solver_forces = True # for example contacts
            sensor_options.use_world_frame = True # report forces in world frame 
            self.gym.create_asset_force_sensor(self.asset, index, sensor_pose, sensor_options)

        #create terrains
        self.num_terrains=self.params['num_terrains']
        self.terrain=Terrain(self.params)
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(), 
                                   self.terrain.triangles.flatten(), self.terrain.tm_params)
        self.terrain_roughness=torch.zeros((params['num_envs'],1), dtype=torch.long, device=self.device, requires_grad=False)
        self.heightfield=torch.tensor(self.terrain.heightfield).to(self.device)
        
        # set up the env grid
        self.num_envs = params['num_envs']
        self.num_levels=params['num_levels']
        envs_per_row = self.num_envs
        env_spacing = 0.
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing,0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        
        # cache some common handles for later use
        self.envs = []
        self.actor_handles = []
        
        
        # prepare friction randomization
        rigid_shape_prop =self.gym.get_asset_rigid_shape_properties(self.asset)
        self.friction_range=self.params['friction_range']
        self._friction=torch_rand_float(self.friction_range[0], self.friction_range[1], 
                                       (self.num_envs,1), device=self.device)
        self.friction=(self._friction+1)/2
        
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        # create and populate the environments
        self.actors_per_env=self.num_envs/self.terrain.num_terrains
        
        pose = gymapi.Transform()
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0 * math.pi)   #z-up
        for i in range(self.num_envs):

            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.envs.append(env)
            
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = self._friction[i]
            self.gym.set_asset_rigid_shape_properties(self.asset, rigid_shape_prop)
            
            terrain_idx = int(i / self.actors_per_env)
            self.env_origins[i] = self.terrain_origins[0, terrain_idx]
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.8, 0.8, (2, 1), device=self.device).squeeze(1)
            pos[2]+=0.35
            pose.p = gymapi.Vec3(*pos)
            
            actor_handle = self.gym.create_actor(env, self.asset, pose, "MyActor", i, 0)
            
            #set DOF properties
            props = self.gym.get_actor_dof_properties(env, actor_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            self.gym.set_actor_dof_properties(env, actor_handle, props) 
            
            self.actor_handles.append(actor_handle)
        
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.trunk_index=self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], 'trunk')
            
        
        self.params_scaling=1
        self.motor_strength_base=33.5 #Nm
        
        #environment parameters 
        self.Kp_mean=(self.params['Kp_range'][0]+self.params['Kp_range'][1])/2
        self.Kd_mean=(self.params['Kd_range'][0]+self.params['Kd_range'][1])/2

        self.Kp_range=[self.Kp_mean,self.Kp_mean]
        self.Kd_range=[self.Kp_mean,self.Kp_mean]
        self.payload_range=[0,0]
        self.motor_strength_range=[1,1]
        
        self.Kp=torch.zeros((self.num_envs,12),dtype=torch.float,device=self.device, requires_grad=False)
        self.Kd=torch.zeros((self.num_envs,12),dtype=torch.float,device=self.device, requires_grad=False)
        self.motor_strength=torch.zeros((self.num_envs,12),dtype=torch.float,device=self.device, requires_grad=False)
        self.payload=torch.zeros((self.num_envs,1),dtype=torch.float,device=self.device, requires_grad=False)
        
        self.rew_scaling=0.03
        
        #initialize some data
        self.actions=torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions=self.actions
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques=torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload_forces=torch.zeros((self.num_envs, 23, 3), device=self.device, dtype=torch.float)    
        self.payload_pos_base=torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)    
        self.step_buf=torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  
        self.all_ones=torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.timesout=torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device, requires_grad=False)
        
        self.gym.prepare_sim(self.sim)
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        feet_contact_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigidbody_states = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.force_sensor_readings = gymtorch.wrap_tensor(feet_contact_tensor)
        self.rigidbody_states=gymtorch.wrap_tensor(rigidbody_states)
        
        self.base_quat = self.root_states[:, 3:7]
        self.dof_pos = self.dof_state.view(self.num_envs,self.num_dofs , 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces=self.force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
        self.rigidbody_pos=self.rigidbody_states.view(self.num_envs,-1,13)[:,:,0:3]
        self.feet_vel=self.rigidbody_states.view(self.num_envs,-1,13)[:,self.feet_indices,7:10]
               
        self.last_dof_pos=torch.zeros_like(self.dof_pos)
        self.last_contact_forces=torch.zeros_like(self.contact_forces)
        
        # create a local copy of initial state, which we can send back for reset 
        self.saved_root_tensor = self.root_states.clone()
        self.saved_dof_pos_tensor=self.dof_pos.clone()
        self.saved_dof_vel_tensor=self.dof_vel.clone()
        
        # ******************quantities for training*****************
        self.num_obs=42                #obs for the base policy except the extrinsics
        self.num_privileged_obs=60     #obs for the critic, including env_factor
        self.num_extrinsics=8
        self.num_env_factors=18
        
        #for reset
        self.reset_all=torch.ones(self.num_envs, device=self.device, dtype=torch.bool)  
        
                    
    def check_termination(self):
        
        #base too low
        self.feet_height=torch.sum(self.rigidbody_pos[:,self.feet_indices,2],dim=1)/4
        self.base_height=self.rigidbody_pos[:,1,2]-self.feet_height
        self.reset_buf=torch.where(self.base_height<0.23,torch.ones_like(self.reset_buf),self.reset_buf)
        
        self.reset_buf=torch.where(torch.abs(self.base_roll)>0.65,torch.ones_like(self.reset_buf),self.reset_buf)
        self.reset_buf=torch.where(torch.abs(self.base_pitch)>0.25,torch.ones_like(self.reset_buf),self.reset_buf)
        self.reset_buf=torch.where(self.step_buf>=1000,torch.ones_like(self.reset_buf),self.reset_buf)
        self.timesout=torch.where(self.step_buf>=1000,torch.ones_like(self.timesout),self.timesout)
    
    def compute_observation(self):
        self.state=torch.cat((self.dof_pos,self.dof_vel,self.base_roll.view(-1,1),self.base_pitch.view(-1,1),
                              self.contact_indicator),dim=-1)
        self.terrain_height=torch.zeros((self.num_envs,1),dtype=torch.float,device=self.device, requires_grad=False)
        # TODO: vectorized
        #     FL_pos=self.rigidbody_pos[:,self.feet_indices[0],0:2]
        #     FR_pos=self.rigidbody_pos[:,self.feet_indices[1],0:2]
        #     front_height=(self.get_height(FL_pos)+self.get_height(FR_pos))/2
        #     RL_pos=self.rigidbody_pos[:,self.feet_indices[2],0:2]
        #     RR_pos=self.rigidbody_pos[:,self.feet_indices[3],0:2]
        #     rear_height=(self.get_height(:,RL_pos)+self.get_height(:,RR_pos))/2
        #     self.terrain_height=front_height-rear_height
            
        # compute roughness
        level_idx=torch.floor(self.root_states[:,0]/self.terrain.terrain_width).long()
        terrain_idx=torch.floor(self.root_states[:,1]/self.terrain.terrain_width).long()
        self.terrain_roughness[:,0]=self.terrain.roughness[level_idx,terrain_idx]
        # TODO: vectorized operation
        # self.terrain_roughness[:,0]=torch.where(torch.abs(self.root_states[:,0]-self.terrain_origins[level_idx,terrain_idx]                         [0])>1.,torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
        # self.terrain_roughness[:,0]=torch.where(torch.abs(self.root_states[:,1]-self.terrain_origins[level_idx,terrain_idx]                         [1])>1.,torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
        self.terrain_roughness[:,0]=torch.where(self.root_states[:,0]>self.num_levels*self.terrain.terrain_width, 
                                                torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
        self.terrain_roughness[:,0]=torch.where(self.root_states[:,0]<0,
                                                torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
        self.terrain_roughness[:,0]=torch.where(self.root_states[:,1]>self.num_terrains*self.terrain.terrain_width , 
                                                torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
        self.terrain_roughness[:,0]=torch.where(self.root_states[:,1]<0, 
                                                torch.zeros_like(self.terrain_roughness[:,0]),self.terrain_roughness[:,0])
                                                  
        self.privileged_factors=torch.cat((self.payload,self.payload_pos_base[:,:2],self.motor_strength,
                                          self.friction,self.terrain_height,self.terrain_roughness),dim=-1)

    def compute_reward(self):
        # forward reward
        rew_forward=torch.minimum(self.base_lin_vel[:,0],torch.ones_like(self.base_lin_vel[:,0])*0.35)*20
        
        # lateral movement and rotation penalty
        rew_lateral_rot=(-torch.sum(torch.square(self.base_lin_vel[:,1]),dim=0)-
                         torch.sum(torch.square(self.base_ang_vel[:,2]),dim=0))*21
        
        # power penalty
        rew_power=torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # for i in range(self.num_envs):
        #     rew_power[i]=-torch.dot(self.torques[i],(self.dof_pos[i]-self.last_dof_pos[i]))*0.002
        
        # ground impact penalty
        rew_ground=-torch.sum(torch.square(self.contact_forces-self.last_contact_forces),dim=(1,2))*0.02
        
        # smoothness penalty
        rew_smooth=-torch.sum(torch.square(self.torques-self.last_torques),dim=1)*0.001
        
        # action magnitude penalty
        rew_ac=-torch.sum(torch.square(self.actions),dim=1)*0.07
        
        # joint speed penalty
        rew_dof_vel=-torch.sum(torch.square(self.dof_vel),dim=1)*0.002
        
        # orientation penalty
        rew_orientation=-torch.sum(torch.square(self.base_ang_vel[:,0:2]),dim=1)*1.
        
        # vertical acceleration penalty
        rew_Z=-torch.sum(torch.square(self.base_lin_vel[:,2]),dim=0)*2.0
        
        # foot slip penalty
        rew_slip=torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        feet_vel_square=torch.sum(torch.square(self.feet_vel),dim=2)
        # TODO:vectorized operation
        #rew_slip=-torch.sum(self.contact_indicator*feet_vel_square,dim=1)*0.8
        # for i in range(self.num_envs):
        #     #rew_slip[i]=-torch.dot(self.contact_indicator[i],feet_vel_square[i])*0.8
        #      rew_slip[i]=-torch.sum(self.contact_indicator[i]*feet_vel_square[i])*0.8
        
        self.reward=rew_forward+rew_lateral_rot+(rew_power+rew_ground+rew_smooth+rew_ac+
                                                 rew_dof_vel+rew_orientation+rew_Z+rew_slip)*self.rew_scaling
        
        
    def pre_physics_step(self, actions):
        # decide whether to resample
        dice=random.uniform(0,1)
        if dice<self.params['resample_prob']:
            self.sample_params()

        # refresh last**
        self.last_actions=self.actions                
        self.last_torques=self.torques
        self.last_dof_pos=self.dof_pos
        self.last_contact_forces=self.contact_forces
        
        self.actions = actions.clone()
        # print('kp: ', self.Kp)
        print('action: ', self.actions)
        # print('dof_pos: ', self.dof_pos)
        # print('Kd: ', self.Kd)
        # print('dof_vel', self.dof_vel)
        # apply forces
        torques = self.Kp*(self.actions - self.dof_pos) - self.Kd*self.dof_vel
                             
        # for i in range(self.num_envs):
        #     for j in range(self.num_dofs):
        #         torques[i,j]=torch.clamp(torques[i,j], -self.motor_strength[i,j], self.motor_strength[i,j])
        
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        self.torques = torques.view(self.torques.shape)
        
        #apply payload to the trunk
        payload_offset=quat_rotate(self.base_quat,self.payload_pos_base)    #offset in world frame
        force_positions=self.rigidbody_pos.clone()
        force_positions[:,self.trunk_index,:]+=payload_offset
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.payload_forces), 
                                                       gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        
        # step the simulator
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.step_buf+=self.all_ones
        
    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_roll,self.base_pitch,_=get_euler_xyz(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.contact_indicator=(self.contact_forces[:,:,2]>1.)
        
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_observation()

                       
                         
    def reset(self,idx=None):     #idx should be tensor!
        if idx==None:
            idx=self.reset_all
        #idx=idx.squeeze()
        self.root_states[idx,:]=self.saved_root_tensor[idx,:]
        self.root_states[idx,:3]=self.env_origins[idx]
        self.root_states[idx, :2] += torch_rand_float(-0.5, 0.5, (len(idx), 2), device=self.device)
        self.root_states[idx,2]+=0.35
        #self.root_states[idx,1]=torch_rand_float(2.5, 5.5, (len(idx),1), self.device).squeeze()
        self.dof_pos[idx,:]=self.saved_dof_pos_tensor[idx,:]
        self.dof_vel[idx,:]=torch_rand_float(-0.1, 0.1, (len(idx), self.num_dofs), device=self.device)

        
        idx_int32 = idx.to(dtype=torch.int32)
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(idx_int32), len(idx_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(idx_int32), len(idx_int32))
                         
        self.last_actions[idx,:]=0.           
        self.last_torques[idx,:]=0.
        self.last_dof_pos[idx,:]=0.
        self.last_contact_forces[idx,:]=0.
        self.reset_buf[idx]=0
        self.timesout[idx]=0
        self.step_buf[idx]=0
        
            
      
    # called before every iter (or within an episode when necessary)
    def sample_params(self):
        for j in range(self.num_envs):
            terrain_level=random.randint(1,self.level)
            terrain_idx = int(j / self.actors_per_env)
            self.env_origins[j]=self.terrain_origins[terrain_level-1, terrain_idx]
            #self.saved_root_tensor[j,0]=self.terrain.terrain_width*(terrain_level-1)+1.5
            #self.terrain_roughness[j]=self.terrain.roughness[terrain_level-1,terrain_idx]

        self.Kp=torch_rand_float(self.Kp_range[0],self.Kp_range[1],(self.num_envs,12),device=self.device)
        self.Kd=torch_rand_float(self.Kd_range[0],self.Kd_range[1],(self.num_envs,12),device=self.device)
        self.motor_strength=torch_rand_float(self.motor_strength_range[0],
                                             self.motor_strength_range[1],(self.num_envs,12),device=self.device)             
        self.payload=torch_rand_float(0,self.payload_range[1],(self.num_envs,1), device=self.device)
        self.payload_forces[:,self.trunk_index,2]=-self.payload.squeeze()*9.8         #put on the trunk

        self.payload_pos_base[:,0]=(0.1 + 0.1) * torch.rand(self.num_envs, device=self.device) - 0.1
        self.payload_pos_base[:,1]=(0.07 + 0.07) * torch.rand(self.num_envs, device=self.device) - 0.07
        self.payload_pos_base[:,2]=to_torch(0.057, device=self.device).repeat(self.num_envs)
        
    def get_height(self,state):     #state=[env_ids,pos_x_y]
        # TODO: vectorized operation
        x=self.terrain.border_size+state[:,0]
        y=self.terrain.border_size+state[:,1]
        x=torch.round(x/self.params['horizontal_scale']).long
        y=torch.round(y/self.params['horizontal_scale']).long
        return self.heightfield[x,y]*self.params['vertical_scale']

    # ******************interfaces for training ****************** #
                         
    def step(self, actions: torch.Tensor):#-> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()                 
        if len(env_ids) > 0:
            self.reset(env_ids)  
        self.pre_physics_step(actions)
        self.post_physics_step()
        obs=torch.cat((self.state,self.actions),dim=-1)
        critic_obs=torch.cat((self.privileged_factors,obs),dim=-1)
        dones=self.reset_buf
        rewards=self.reward
        info={}
        info['time_outs']=self.timesout
        return obs,critic_obs,rewards,dones,info
                         
    def get_observations(self) -> torch.Tensor:
        # prepare quantities
        # self.gym.simulate(self.sim)
        
        self.base_quat = self.root_states[:, 3:7]
        self.base_roll,self.base_pitch,_=get_euler_xyz(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.contact_indicator=(self.contact_forces[:,:,2]>1.)
        
        self.state=torch.cat((self.dof_pos,self.dof_vel,self.base_roll.view(-1,1),self.base_pitch.view(-1,1),
                              self.contact_indicator),dim=-1)
        self.obs=torch.cat((self.state,self.actions),dim=-1)
        return self.obs
    
    def get_privileged_observations(self):# -> Union[torch.Tensor, None]:
        self.compute_observation()
        self.critic_obs=torch.cat((self.privileged_factors,self.obs),dim=-1)
        return self.critic_obs
    
     #refresh environment parameter in ith iteration   
    def levelup(self,i):                 
        self.params_scaling=i/self.params['n_iter']
        self.rew_scaling=math.pow(self.rew_scaling,0.997)

        # fri_len=(self.params['friction_range'][1]-self.params['friction_range'][0])/2
        # self.friction_range=[self.friction_mean-fri_len*self.params_scaling,self.friction_mean+fri_len*self.params_scaling]

        Kp_len=(self.params['Kp_range'][1]-self.params['Kp_range'][0])/2           
        self.Kp_range=[self.Kp_mean-Kp_len*self.params_scaling,self.Kp_mean+Kp_len*self.params_scaling]

        Kd_len=(self.params['Kd_range'][1]-self.params['Kd_range'][0])/2   
        self.Kd_range=[self.Kd_mean-Kd_len*self.params_scaling,self.Kd_mean+Kd_len*self.params_scaling]

        ms_len=(self.params['motor_strength_range'][1]-self.params['motor_strength_range'][0])/2   
        self.motor_strength_range=[1-ms_len*self.params_scaling,1+ms_len*self.params_scaling]

        self.payload_range[1]=self.params['payload_range'][1]*self.params_scaling

        self.level=int(i/self.params['n_iter']*self.num_levels)+1
        