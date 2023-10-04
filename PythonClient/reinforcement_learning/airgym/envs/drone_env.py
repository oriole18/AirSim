
import airsim
import numpy as np
import math
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import cv2

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "velocity":np.zeros(2),
            "distance": 0,
            
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
    
        
        self._setup_flight()

        #self.image_request = airsim.ImageRequest( 3, airsim.ImageType.DepthPerspective, True, False)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self,yaw=10):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        
        self.start_point=[-2.1,77,-2]
        
        self.target_point=[20.63,75,-2]

        self.start_x=self.start_point[0]
        self.start_y=self.start_point[1]
        self.start_z=self.start_point[2]
        self.start_yaw=yaw
        position = airsim.Vector3r(self.start_x,self.start_y)
        #orientation = airsim.to_quaternion(0, 0, self.start_yaw)
        #pose = airsim.Pose(position,orientation)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision=True)

        self.drone.moveToPositionAsync(self.start_x, 
                                       self.start_y, 
                                       self.start_z, 
                                       0
                                       )
    
    def transform_obs(self, responses):
        MIN_DEPT=0
        MAX_DEPTH=7
         # Reshape to a 2d array with correct width and height
        depth_img = airsim.list_to_2d_float_array(responses.image_data_float, responses.width, responses.height)
        depth_img = depth_img.reshape(responses.height, responses.width, 1)
        depth_img= np.interp(depth_img, (MIN_DEPT, MAX_DEPTH), (0,255))
        depth_img=depth_img/255
       
        return depth_img 

    def get_distance(self,position):

     #  print(self.target_point[:-1])
        distance= np.linalg.norm(position-self.target_point[:-1])
      #  print(distance)
        return distance
    
    def check_out(self,pos_x,pos_y):
        
        if pos_x>21 or pos_x<-2.2 or pos_y> 80 or pos_y<72 :
            return True
        
        return False

    def _get_obs(self):
        response = self.drone.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True, False)])[0]
        image = self.transform_obs(response)
        self.drone_state = self.drone.getMultirotorState()

        
        
        self.state["position"] = np.array([self.drone_state.kinematics_estimated.position.x_val,
                                           self.drone_state.kinematics_estimated.position.y_val
                                           ])
        
        self.state["velocity"] = np.array([self.drone_state.kinematics_estimated.linear_velocity.x_val,
                                           self.drone_state.kinematics_estimated.linear_velocity.y_val
                                           ])
        
        self.state["distance"] = self.get_distance(self.state["position"])
        
        collision = self.drone.simGetCollisionInfo().has_collided
        
        pos_x=self.state["position"][0]
        pos_y=self.state["position"][1]
        print('ssss')
        out=self.check_out(pos_x,pos_y)
        if out :
            collision=True
        
        self.state["collision"] = collision
        
        cv2.imshow("depth_img",image)
        cv2.waitKey(1)

        return image

    def _do_action(self, action):
       action=self.interpret_action(action)
       v_x=action[0]
       yaw=action[1]
       self.drone.moveByVelocityZBodyFrameAsync(
            vx = v_x,
            vy = 0,
            z = 0,#(self.random_alt),
            duration = 0.5,
            drivetrain = airsim.DrivetrainType.ForwardOnly,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw))
        )
    
    def _compute_reward(self):
        dist_threshold=10
        goal_threshold=1.5
        reward = 0
        done = 0
        success=False
        dist_current=self.state["distance"]
        

        if dist_current >dist_threshold:
            reward=max(-0.995,round(-(dist_current/25),3))
        else:
            reward=min(0.995,round(1/dist_current,3))
        
        # 충돌시 패널티
        if self.state['collision']:
            reward=-1
            done=1
            return reward, done,success            
        #목적지 통과
        if dist_current <goal_threshold:
            print('-----------------------------goal!!!!!!!!-----------------------')
            reward=1
            done=1
            success=True

        return reward,done,success
       

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done,success = self._compute_reward()

        return obs, reward, done,success

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = [20,0]
        elif action == 1:
            quad_offset = [0,20]
        elif action == 2:
            quad_offset = [0,-20]
        else:
            quad_offset = [0,0]

        return quad_offset


