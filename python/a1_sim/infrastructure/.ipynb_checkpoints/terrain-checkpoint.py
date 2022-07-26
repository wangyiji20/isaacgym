from isaacgym.terrain_utils import *
import torch

class Terrain:
    def __init__(self, params):
        # create all available terrain types
        self.proportions=[np.sum(params["terrain_proportion"][:i+1]) for i in range(len(params["terrain_proportion"]))]
        self.num_terrains = params['num_terrains']
        self.terrain_width = params['terrain_width']    
        self.terrain_length = self.terrain_width
        self.border_size = 20
        horizontal_scale = params['horizontal_scale']  
        vertical_scale =params['vertical_scale']  
        num_levels=params['num_levels']
        num_rows = int(self.terrain_width/horizontal_scale)
        num_cols = int(self.terrain_length/horizontal_scale)
        self.border_rows = int(self.border_size/horizontal_scale)
        self.env_origins = np.zeros((num_levels, self.num_terrains, 3))
        self.roughness=torch.zeros((params['num_levels'],params['num_terrains']),dtype=torch.float, requires_grad=False)
        
        self.heightfield = np.zeros((num_levels*num_rows+2*self.border_rows, self.num_terrains*num_cols+2*self.border_rows), dtype=np.int16)
        for j in range(self.num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,             horizontal_scale=horizontal_scale)
                
                difficulty = i / num_levels
                choice = j / self.num_terrains

                slope = 0.4*difficulty
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope,platform_size=2.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope,platform_size=2.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height,platform_size=2.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40,platform_size=2.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,platform_size=2.)
                    
                # Heightfield coordinate system
                start_x = self.border_rows + i * num_rows
                end_x = self.border_rows + (i + 1) * num_rows
                start_y = self.border_rows + j * num_cols
                end_y = self.border_rows + (j + 1) * num_cols
                self.heightfield[start_x: end_x, start_y:end_y] = terrain.height_field_raw   
                self.roughness[i,j]=terrain.std
                
                env_origin_x = (i + 0.5) * self.terrain_length
                env_origin_y = (j + 0.5) * self.terrain_width
                x1 = int((self.terrain_length/2. - 0.5) / horizontal_scale)
                x2 = int((self.terrain_length/2. + 0.5) / horizontal_scale)
                y1 = int((self.terrain_width/2. - 0.5) / horizontal_scale)
                y2 = int((self.terrain_width/2. + 0.5) / horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        # add the terrain as a triangle mesh
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.heightfield, horizontal_scale=horizontal_scale,           vertical_scale=vertical_scale, slope_threshold=0.5)
        self.tm_params = gymapi.TriangleMeshParams()
        self.tm_params.nb_vertices = self.vertices.shape[0]
        self.tm_params.nb_triangles = self.triangles.shape[0]
        self.tm_params.transform.p.x = -self.border_size            #add a border to the terrain
        self.tm_params.transform.p.y = -self.border_size
        self.tm_params.dynamic_friction=1.
        self.tm_params.restitution =0.
        self.tm_params.static_friction=1.
