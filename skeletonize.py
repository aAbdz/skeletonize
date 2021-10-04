"""skeletonize is a Python module that implements a distance transform based
skeletonization method.

https://github.com/aAbdz/skeletonize/"""
    

import numpy as np
import skfmm


class skeletonize():
    def __init__(self, speed_power=1.2, Euler_step_size=0.5, depth_th=2, length_th=None, simple_path=False, verbose=False):
        super().__init__()        
        self.speed_power = speed_power
        self.Euler_step_size = Euler_step_size
        self.simple_path = simple_path 
        self.length_th = length_th 
        self.depth_th = depth_th    
        self.verbose = verbose
    
    
    def _get_line_length(self, L):        
        length = np.sum( np.sum( (L[1:]-L[:-1])**2, axis=1 )**0.5 )
        return length
    
    
    def _point_min(self, dist, im_2d):        
        sz = dist.shape
        max_dist_ = np.max(dist)
        pd_dist = max_dist_ * np.ones(np.array(sz)+2)
        
        if im_2d:
            pd_dist[1:-1, 1:-1] = dist
            
            Fx = np.zeros(sz, dtype=np.float64)
            Fy = np.zeros(sz, dtype=np.float64)
            
            x = [1,-1, 0, 0, 1, 1,-1,-1] 
            y = [0, 0, 1,-1, 1,-1, 1,-1] 
            
            for i in range(len(x)):       
                in_ = pd_dist[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i]]
                check = in_<dist
                dist[check] = in_[check]            

                den = (x[i]**2 + y[i]**2)**0.5            
                Fx[check] = x[i]/den
                Fy[check] = y[i]/den 
            return [Fx, Fy]
        
        else:
            pd_dist[1:-1, 1:-1, 1:-1] = dist
            
            Fx = np.zeros(sz, dtype=np.float64)
            Fy = np.zeros(sz, dtype=np.float64)
            Fz = np.zeros(sz, dtype=np.float64)
            
            x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
            z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]        
            
            for i in range(len(x)):       
                in_ = pd_dist[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i], 1+z[i]:1+sz[2]+z[i]]
                check = in_<dist
                dist[check] = in_[check]            

                den = (x[i]**2 + y[i]**2 + z[i]**2)**0.5            
                Fx[check] = x[i]/den
                Fy[check] = y[i]/den 
                Fz[check] = z[i]/den             
            return [Fx, Fy, Fz]
        
    
    
    def _Euler_path_3d(self, Fx, Fy, Fz, start_point, step_size):    
        f_start_point = np.floor(start_point).astype(int)
        sz = Fx.shape
        
        x = [0, 0, 0, 0, 1, 1, 1, 1] 
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        z = [0, 1, 0, 1, 0, 1, 0, 1]
        
        neighbor_inx = np.array((x,y,z)).T

        base = f_start_point + neighbor_inx
        base[base<0] = 0
        xbase = base[:,0]; xbase[xbase>=sz[0]] = sz[0]-1
        ybase = base[:,1]; ybase[ybase>=sz[1]] = sz[1]-1
        zbase = base[:,2]; zbase[zbase>=sz[2]] = sz[2]-1
        base  = np.array((xbase,ybase,zbase)).T
        
        dist2f = np.squeeze(start_point-f_start_point)
        dist2c = 1-dist2f
        
        perc = np.array((   dist2c[0]*dist2c[1]*dist2c[2],
                            dist2c[0]*dist2c[1]*dist2f[2],
                            dist2c[0]*dist2f[1]*dist2c[2],
                            dist2c[0]*dist2f[1]*dist2f[2],
                            dist2f[0]*dist2c[1]*dist2c[2],
                            dist2f[0]*dist2c[1]*dist2f[2],
                            dist2f[0]*dist2f[1]*dist2c[2],
                            dist2f[0]*dist2f[1]*dist2f[2]  ))
        
        gradient_valueX = [Fx[tuple(i)] for i in base]*perc
        gradient_valueY = [Fy[tuple(i)] for i in base]*perc 
        gradient_valueZ = [Fz[tuple(i)] for i in base]*perc 

        gradient_value = np.array((gradient_valueX, gradient_valueY, gradient_valueZ))
        sum_g = np.sum(gradient_value, axis=1)        
        gradient = sum_g / ((np.sum(sum_g**2)+0.000001)**0.5)
        end_point = start_point - step_size*gradient
        
        if (np.any(end_point<0) or end_point[0,0]>sz[0] or end_point[0,1]>sz[1] or end_point[0,2]>sz[2]):
            end_point = np.zeros((1,3))
        return end_point
    
    
    def _Euler_path_2d(self, Fx, Fy, start_point, step_size):    
        f_start_point = np.floor(start_point).astype(int)
        sz = Fx.shape
        
        x = [0, 0, 1, 1] 
        y = [0, 1, 0, 1] 
        
        neighbor_inx = np.array((x,y)).T

        base = f_start_point + neighbor_inx
        base[base<0] = 0
        xbase = base[:,0]; xbase[xbase>=sz[0]] = sz[0]-1
        ybase = base[:,1]; ybase[ybase>=sz[1]] = sz[1]-1        
        base  = np.array((xbase,ybase)).T
        
        dist2f = np.squeeze(start_point-f_start_point)
        dist2c = 1-dist2f
        
        perc = np.array((   dist2c[0]*dist2c[1],                            
                            dist2c[0]*dist2f[1],                            
                            dist2f[0]*dist2c[1],
                            dist2f[0]*dist2f[1]  ))
        
        gradient_valueX = [Fx[tuple(i)] for i in base]*perc
        gradient_valueY = [Fy[tuple(i)] for i in base]*perc 

        gradient_value = np.array((gradient_valueX, gradient_valueY))
        sum_g = np.sum(gradient_value, axis=1)        
        gradient = sum_g / ((np.sum(sum_g**2)+0.000001)**0.5)
        end_point = start_point - step_size*gradient
        
        if np.any(end_point<0) or np.any(end_point>sz):
            end_point = np.zeros_like(end_point)
            
        return end_point
    
    
    def _Euler_shortest_path(self, dist, start_point, source_point, step_size, im_2d):       
        F = self._point_min(dist, im_2d)
        
        if im_2d:
            Fx, Fy = -F[0], -F[1]
        else: 
            Fx, Fy, Fz = -F[0], -F[1], -F[2] 
            
        itr = 0
        path = start_point
        while True:                        
            if im_2d:
                end_point = self._Euler_path_2d(Fx, Fy, start_point, step_size)                
            else: 
                end_point = self._Euler_path_3d(Fx, Fy, Fz, start_point, step_size)
               
            endpoint_dist_to_all = np.sum((source_point-end_point)**2, axis=1)**0.5
            distance_to_endpoint = np.min(endpoint_dist_to_all)
            
            if itr>=10: 
                movement = np.sum((end_point-path[itr-10])**2)**0.5
            else: 
                movement = step_size+1
            
            if np.all(end_point==0) or movement<step_size: break
        
            itr += 1            
            path = np.append(path, end_point, axis=0)            
            if distance_to_endpoint<4*step_size:
                source_inx = source_point[np.argmin(endpoint_dist_to_all)]
                path = np.append(path, np.expand_dims(source_inx, axis=0), axis=0)
                break
            
            start_point = end_point
        return path
    
    
    def _discrete_shortest_path(self, dist, start_point, im_2d):           
        sz = dist.shape   
        if im_2d:                  
            x = [0, 1,-1, 0, 0, 1, 1,-1,-1] 
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1]              
            neighbor_inx = np.array((x,y)).T
        else:
            x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
            z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]           
            neighbor_inx = np.array((x,y,z)).T
             
        path = start_point.copy()
        min_v = np.inf
        while min_v!=0:             
            ngb = start_point + neighbor_inx
            valid_ngb = np.all((np.all(ngb>=0, axis=1), np.all(ngb<sz, axis=1)), axis=0)
            ngb = ngb[valid_ngb]            
            ngb_value = dist[tuple(ngb.T)]
            min_ind = np.argmin(ngb_value)
            min_v = ngb_value[min_ind]        
            start_point = ngb[min_ind]
            path = np.append(path, np.expand_dims(start_point, axis=0), axis=0)     
        return path


    def _organize_skeleton(self, skel_seg, length_th, im_2d):        
        final_skeleton = []        
        n = len(skel_seg)
        if im_2d:
            end_points = np.zeros((n*2, 2))        
        else:
            end_points = np.zeros((n*2, 3))        
            
        l = 0
        for i in range(n):
            ss = skel_seg[i]
            l = max(l, len(ss))
            end_points[i*2] = ss[0]
            end_points[i*2+1] = ss[-1]

        connecting_distance = 2
        for i in range(n):
            ss = np.asarray(skel_seg[i])
            
            ex = np.reshape(end_points[:,0], (-1,1)); ex = np.repeat(ex, len(ss), axis=1)       
            sx = np.reshape(ss[:,0], (1,-1)); sx = np.repeat(sx, len(end_points), axis=0)
                       
            ey = np.reshape(end_points[:,1], (-1,1)); ey = np.repeat(ey, len(ss), axis=1)       
            sy = np.reshape(ss[:,1], (1,-1)); sy = np.repeat(sy, len(end_points), axis=0)
            
            if im_2d:
                dist_ = (ex-sx)**2 + (ey-sy)**2
            else:
                ez = np.reshape(end_points[:,2], (-1,1)); ez = np.repeat(ez,len(ss), axis=1)
                sz = np.reshape(ss[:,2], (1,-1)); sz = np.repeat(sz,len(end_points), axis=0)      
                dist_ = (ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2

            check = np.amin(dist_, axis=1)<connecting_distance
            check[i*2] = False
            check[i*2+1] = False            
            cut_skel = [0, len(ss)]
            if(any(check)):
                for ii in range(len(check)):
                    if(check[ii]):
                        line = dist_[ii]
                        min_ind = np.ma.argmin(line)
                        if (min_ind>2) and (min_ind<(len(line)-2)):
                            cut_skel.append(min_ind)
                            
            cut_skel = sorted(cut_skel)
            for j in range(len(cut_skel)-1):      
                skel_breaked_seg = ss[cut_skel[j]:cut_skel[j+1]]
                length_skel_seg = self._get_line_length(skel_breaked_seg)
                if length_skel_seg>=length_th:
                   final_skeleton.append(skel_breaked_seg)                   
        return final_skeleton
        
    
    def skeleton(self, obj):    
        obj = np.array(obj, dtype=np.bool)
        im_2d = True if obj.ndim==2 else False
        
        boundary_dist = skfmm.distance(obj)        
        source_point = np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)        
        max_dist_ = boundary_dist[source_point]
        speed_im = (boundary_dist / max_dist_) ** self.speed_power
        del boundary_dist
        
        flag = True
        length_threshold = 0.0
        obj = np.ones(obj.shape, dtype=np.float64)
        obj[source_point] = 0.0
        skeleton_segments = []
        source_point = np.expand_dims(source_point, axis=0)        
        while True:        
            dist = skfmm.travel_time(obj, speed_im)
            end_point = np.unravel_index(np.ma.argmax(dist), dist.shape)
            max_dist = dist[end_point]
            dist = np.ma.filled(dist, max_dist)            
            end_point = np.expand_dims(end_point, axis=0)
            
            if self.simple_path:
                shortest_path = self._discrete_shortest_path(dist, end_point, im_2d)                
            else:
                shortest_path = self._Euler_shortest_path(dist, end_point, source_point, self.Euler_step_size, im_2d)                
                
            path_length = self._get_line_length(shortest_path)
            if self.verbose:
                print(path_length)
    
            if flag:
                depth_threshold  = self.depth_th * max_dist_                
                
                longest_line_threshold = np.inf
                if self.length_th:
                    longest_line_threshold = self.length_th * path_length
                
                length_threshold = min(depth_threshold, longest_line_threshold)
                flag = False
            
            if path_length<=length_threshold: break
            
            source_point = np.append(source_point, shortest_path, axis=0)            
            skeleton_segments.append(shortest_path)
            
            shortest_path = np.floor(shortest_path).astype(int)
            obj[tuple(shortest_path.T)] = 0        
        
        final_skeleton = None
        if len(skeleton_segments) != 0:
            final_skeleton = self._organize_skeleton(skeleton_segments, length_threshold, im_2d)
 
        return final_skeleton










    
