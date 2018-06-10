from __future__ import print_function
import os
import numpy as np
import math



class Point_match:
    
    def __init__(self):
        self.theta = 0  #rotation angle
        self.a = 0  #scaling parameter
        self.p = 0  #projecting parameter
        self.t = np.array([[0.0] , [300.0]]) #translation
        self.degree_to_radian = 3.1415926 / 180
        self.radian_to_degree = 180 / 3.1415926
        
        self.gamma_0 = 30  #initial regularization hyper-parameter
        self.beta_0 = 0.00009 # initial annealing hyper-parameter
        self.beta_f = 0.1  # final annealing hyper-parameter
        self.beta_r = 1.05 # annealing changing rate
        self.ep0 = 0.01 
        self.ep1 = 0.01  # error threshold for point_match step D
        self.ep2 = 0.005 # error threshold for point_match step B
        self.Get_A_matrix = {'Affine': self.Affine,'DA_Da': self.DA_Da,'DA_Da_2': self.DA_Da_2}
        
    def Affine (self,a,theta):
        radians = theta * self.degree_to_radian
        sa = np.array([[math.exp(a) , 0] , [ 0 , math.exp(a)]])
        s_rotate = np.array([[math.cos(radians) , -math.sin(radians)] , [ math.sin(radians) , math.cos(radians)]])
        return sa.dot(s_rotate)

    def DA_Da (self,a,theta):
        radians = theta * self.degree_to_radian
        d_sa = np.array([[math.exp(a) , 0] , [ 0 , math.exp(a)]])
        s_rotate = np.array([[math.cos(radians) , -math.sin(radians)] , [ math.sin(radians) , math.cos(radians)]])
        return d_sa.dot(s_rotate)
    
    def DA_Da_2 (self,a,theta):
        radians = theta * self.degree_to_radian
        d_sa_2 = np.array([[math.exp(a) , 0] , [ 0 , math.exp(a)]])
        s_rotate = np.array([[math.cos(radians) , -math.sin(radians)] , [ math.sin(radians) , math.cos(radians)]])
        return d_sa_2.dot(s_rotate)
    
    def project (self,p,Y):
        scale_low = 1e-4
        if len(Y.shape) == 1:
            assert np.amax(Y[0])*p*scale_low + 1. != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i] = 1./(1.+p*Y[0]*scale_low)
        elif len(Y.shape) == 2:
            assert np.amax(Y[0,:])*p*scale_low + 1. != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i,:] = 1./(1.+p*Y[0,:]*scale_low)
        else:
            print ('input Y not 1D or 2D array,return 1 !!')
            return 1
        return fp


    def D_project_Dp (self,p,Y):
        scale_low = 1e-4
        if len(Y.shape) == 1:
            assert np.amax(Y[0])*p*scale_low + 1. != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i] = -Y[0]*scale_low/( (1+p*Y[0]*scale_low)**2 )
        elif len(Y.shape) == 2:
            assert np.amax(Y[0,:])*p*scale_low + 1 != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i,:] = -Y[0,:]*scale_low/( (1+p*Y[0,:]*scale_low)**2 )
        else:
            print ('input Y not 1D or 2D array,return 1 !!')
            return 1
        return fp

    def D_project_Dp_2 (self,p,Y):
        scale_low = 1e-4
        if len(Y.shape) == 1:
            assert np.amax(Y[0])*p*scale_low + 1 != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i] = 2*Y[0]*Y[0]*scale_low*scale_low/( (1+p*Y[0]*scale_low)**3 )
        elif len(Y.shape) == 2:
            assert np.amax(Y[0,:])*p*scale_low + 1 != 0, '1+p*Y 不能等於0'
            fp = np.ndarray(Y.shape,np.float64)
            for i in range (fp.shape[0]):
                fp[i,:] = 2*Y[0,:]*Y[0,:]*scale_low*scale_low/( (1+p*Y[0,:]*scale_low)**3 )
        else:
            print ('input Y not 1D or 2D array,return 1 !!')
            return 1
        return fp
    
    def Cost_function(self,X,Y,M,t,a,theta,p,gamma,arpha):
        A = self.Get_A_matrix['Affine'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t

        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        Y_k[:,:,:]=Y_change.T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        diff_ik = np.linalg.norm( (X_i-Y_k),axis = 2 )**2
        
        return np.sum(M*diff_ik) + gamma*(a**2 + p**2) - arpha * np.sum(M)
    
    def Cost_function_da1(self,X,Y,M,t,a,theta,p,gamma,arpha):
        A = self.Get_A_matrix['Affine'](a,theta)
        dA = self.Get_A_matrix['DA_Da'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t
    
        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        dA_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        Y_k[:,:,:]=Y_change.T
        dA_Y_k[:,:,:] = (self.project (p,Y)*dA.dot(Y)).T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        diff_ik = np.sum( ( (X_i-Y_k)* dA_Y_k ),axis = 2 )
       
        return -2*np.sum(M*diff_ik) + gamma*2*a

    def Cost_function_da2(self,X,Y,M,t,a,theta,p,gamma,arpha):
        A = self.Get_A_matrix['Affine'](a,theta)
        dA = self.Get_A_matrix['DA_Da'](a,theta)
        dA2 = self.Get_A_matrix['DA_Da_2'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t

        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        dA_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        dA2_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        Y_k[:,:,:]=Y_change.T
        dA_Y_k[:,:,:] = (self.project (p,Y)*dA.dot(Y)).T
        dA2_Y_k[:,:,:] = (self.project (p,Y)*dA2.dot(Y)).T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        diff_ik = np.sum( ( (X_i-Y_k)* dA2_Y_k - dA_Y_k**2 ),axis = 2 )

        return -2*np.sum(M*diff_ik) + gamma*2 
    
    def Cost_function_dp1(self,X,Y,M,t,a,theta,p,gamma,arpha):
        A = self.Get_A_matrix['Affine'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t
        diff_ik = np.zeros(shape = [X.shape[1], Y.shape[1]])

        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        dA_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        Y_k[:,:,:]=Y_change.T
        dA_Y_k[:,:,:] = (self.D_project_Dp (p,Y)*A.dot(Y)).T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        diff_ik = np.sum( ( (X_i-Y_k)* dA_Y_k ),axis = 2 )
        
        return -2*np.sum(M*diff_ik) + gamma*2*p

    def Cost_function_dp2(self,X,Y,M,t,a,theta,p,gamma,arpha):
        A = self.Get_A_matrix['Affine'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t

        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        dA_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        dA2_Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        Y_k[:,:,:]=Y_change.T
        dA_Y_k[:,:,:] = (self.D_project_Dp (p,Y)*A.dot(Y)).T
        dA2_Y_k[:,:,:] = (self.D_project_Dp_2 (p,Y)*A.dot(Y)).T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        diff_ik = np.sum( ( (X_i-Y_k)* dA2_Y_k - dA_Y_k**2 ),axis = 2 )
        
        return -2*np.sum(M*diff_ik) + gamma*2
    
    def gradient_check_D1(self,X,Y,M,t,a,theta,p,gamma,arpha, epsilon = 1e-7):
        num_parameters = 2 
        parameters_values = np.array([a , p], np.float64)
        grad = np.zeros((num_parameters), np.float64)
        J_plus = np.zeros((num_parameters), np.float64)
        J_minus = np.zeros((num_parameters), np.float64)
        gradapprox = np.zeros((num_parameters), np.float64)


        grad[0] = self.Cost_function_da1(X,Y,M,t,a,theta,p,gamma,arpha)
        grad[1] = self.Cost_function_dp1(X,Y,M,t,a,theta,p,gamma,arpha)

        for i in range(num_parameters):
            param_plus = np.copy(parameters_values)
            print (param_plus)
            param_plus[i] = param_plus[i] +  epsilon
            J_plus[i] = self.Cost_function(X,Y,M,t,param_plus[0] ,theta,param_plus[1],gamma,arpha)

            param_minus = np.copy(parameters_values)
            param_minus[i] = param_minus[i] -  epsilon
            J_minus[i] = self.Cost_function(X,Y,M,t,param_minus[0] ,theta,param_minus[1],gamma,arpha)

            gradapprox[i] = (J_plus[i] - J_minus[i])  / (2*epsilon)


        numerator = np.linalg.norm(grad - gradapprox)                       # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)     # Step 2'
        difference = numerator / denominator                                # Step 3'

        print (grad)
        print (gradapprox)

        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in first partial derivative! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "First partial derivative works perfectly fine! difference = " + str(difference) + "\033[0m")

        return difference
    
    def gradient_check_D2(self,X,Y,M,t,a,theta,p,gamma,arpha, epsilon = 1e-7):
        num_parameters = 2 
        parameters_values = np.array([a , p], np.float64)
        grad = np.zeros((num_parameters), np.float64)
        J_plus = np.zeros((num_parameters), np.float64)
        J_minus = np.zeros((num_parameters), np.float64)
        gradapprox = np.zeros((num_parameters), np.float64)


        grad[0] = self.Cost_function_da2(X,Y,M,t,a,theta,p,gamma,arpha)
        grad[1] = self.Cost_function_dp2(X,Y,M,t,a,theta,p,gamma,arpha)

        J_plus[0] = self.Cost_function_da1(X,Y,M,t,a + epsilon,theta,p,gamma,arpha)
        J_minus[0] = self.Cost_function_da1(X,Y,M,t,a - epsilon,theta,p,gamma,arpha)

        J_plus[1] = self.Cost_function_dp1(X,Y,M,t,a,theta,p+ epsilon,gamma,arpha)
        J_minus[1] = self.Cost_function_dp1(X,Y,M,t,a,theta,p- epsilon,gamma,arpha)

        for i in range (num_parameters):
            gradapprox[i] = (J_plus[i] - J_minus[i])  / (2*epsilon)


        numerator = np.linalg.norm(grad - gradapprox)                       # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)     # Step 2'
        difference = numerator / denominator                                # Step 3'

        print (grad)
        print (gradapprox)

        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in second partial derivative! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Second partial derivative works perfectly fine! difference = " + str(difference) + "\033[0m")

        return difference
    
    def get_new_theta(self,X,Y,M,t,a,p):
        A = self.Get_A_matrix['Affine'](a,0)
        W = self.project (p,Y)*A.dot(Y)

        W_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        W_k[:,:,:]=W.T
        X_i[:,:,:]=(X-t).T
        X_i = X_i.transpose(1,0,2)

        sin_ = np.sum( M*( X_i[:,:,1] * W_k[:,:,0] - X_i[:,:,0] * W_k[:,:,1] ) )
        cos_ = np.sum( M*np.sum(X_i*W_k,axis=2))

        theta = math.atan(sin_/(cos_ + 1e-7)) * self.radian_to_degree
           
        return theta
    
    def get_new_t(self,X,Y,M,a,theta,p):
        A = self.Get_A_matrix['Affine'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y)


        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        M_ik = np.ndarray((X.shape[0],M.shape[0],M.shape[1]))
        Y_k[:,:,:]=Y_change.T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        M_ik[:,:,:] = M
        M_ik = M_ik.transpose(1,2,0)

        Diff = np.sum( ( M_ik * (X_i-Y_k) ),axis = 1 )
        Diff = np.sum( ( Diff ),axis = 0 ).reshape((-1,1))
        
        Diff = Diff / ( np.sum(M) + 1e-7 )
        
        return Diff
    
    def get_M_matrix(self,X,Y,t,a,theta,p,arpha,beta):
        A = self.Get_A_matrix['Affine'](a,theta)
        Y_change = self.project (p,Y)*A.dot(Y) + t

        Y_k = np.ndarray((X.shape[1],Y.shape[1],Y.shape[0]))
        X_i = np.ndarray((Y.shape[1],X.shape[1],X.shape[0]))
        Y_k[:,:,:]=Y_change.T
        X_i[:,:,:]=X.T
        X_i = X_i.transpose(1,0,2)
        M_matrix = np.exp( beta * ( arpha - np.linalg.norm( (X_i-Y_k),axis = 2 )**2 ) )

        return M_matrix

    def soft_assign (self,M_matrix,epsilon = 2e-2):
        sl_M_matrix = np.insert(M_matrix,M_matrix.shape[1],1+epsilon,axis=1)
        sl_M_matrix = np.insert(sl_M_matrix,M_matrix.shape[0],1+epsilon,axis=0)
        sl_M_matrix_pre = np.copy(sl_M_matrix)
        for i in range(30):
            #print (sl_M_matrix)
            sum_axis_0 = np.sum(sl_M_matrix,axis=0)
            sum_axis_0[-1] = 1
            sl_M_matrix = sl_M_matrix / ( sum_axis_0 + 1e-7 )
            #print ( sl_M_matrix )
            sum_axis_1 = np.reshape(np.sum(sl_M_matrix,axis=1), (-1, 1)) 
            sum_axis_1[-1][0] = 1
            sl_M_matrix = sl_M_matrix / ( sum_axis_1 + 1e-7 )

            diff = np.sum( np.abs(sl_M_matrix_pre[0:-1,0:-1] - sl_M_matrix[0:-1,0:-1]))
            if diff < epsilon :
                break
            sl_M_matrix_pre = sl_M_matrix
        return sl_M_matrix[0:-1,0:-1]
    
    def update_pose_Bynewton(self,X,Y,M,t,a,theta,p,gamma,arpha,mode = True):
    
        u_theta = self.get_new_theta(X,Y,M,t,a,p)
        u_t = self.get_new_t(X,Y,M,a,u_theta,p)
        
        u_a = a
        u_p = p

        ax0 = 5
        px0 = 0
        count_a = 0
        count_p = 0

        if mode:
            for i in range(50):
                dv1 = self.Cost_function_da1(X,Y,M,u_t,ax0,u_theta,u_p,gamma,arpha)
                dv2 = self.Cost_function_da2(X,Y,M,u_t,ax0,u_theta,u_p,gamma,arpha)
                u_a = ax0 - dv1/(dv2+1e-5)
               
                if ( abs(u_a) > 200 ):
                    u_a = 0
                    break

                if ( abs((ax0 - u_a)/(u_a+1e-5) ) < 1e-4 ):
                    ax0 = u_a
                    break
                ax0 = u_a

            for i in range(50):
                dv1 = self.Cost_function_dp1(X,Y,M,u_t,u_a,u_theta,px0,gamma,arpha)
                dv2 = self.Cost_function_dp2(X,Y,M,u_t,u_a,u_theta,px0,gamma,arpha)
                u_p = px0 - dv1/(dv2+1e-5)
                if ( abs((px0 - u_p)/(u_p+1e-5) ) < 1e-4 ):
                    px0 = u_p
                    break
                px0 = u_p


            if (abs(u_p) > 0.2):
                u_p = 0
                
        return u_theta, u_t, u_a,u_p
    
    def point_matching(self,X,Y,arpha, mode = True):
        # X,Y : point set. Y will be matched to X
        # arpha : outlier threshold
        # mode: false when rigid transform only (no scaling and project), true when all transform
        
        u_t0 = self.t
        u_a0,u_theta0,u_p0 = self.a,self.theta,self.p
        beta = self.beta_0
        gamma = self.gamma_0
        M = np.ndarray((X.shape[1],Y.shape[1]))
        #count_while = 0
        while (beta < self.beta_f): #stepA
            #count_stepB = 0
            count_inerror = 0
            for i in range(100): # stepB
                
                #stepC and stepD
                M = self.soft_assign(self.get_M_matrix(X,Y,u_t0,u_a0,u_theta0,u_p0,arpha,beta), self.ep1)
               
                #stepE
                u_theta, u_t, u_a,u_p = self.update_pose_Bynewton(X,Y,M,u_t0,u_a0,u_theta0,u_p0,gamma,arpha,mode)

                para_error_t = np.sum(abs(u_t0-u_t))
                para_error =abs(u_a0 - u_a)  + abs(u_theta0 - u_theta) + abs(u_p0 - u_p)
                u_t0,u_a0,u_theta0,u_p0 = u_t,u_a,u_theta, u_p
                #count_stepB = count_stepB + 1
                if (para_error < self.ep2 and para_error_t < 1):
                    count_inerror = count_inerror + 1
                    if (count_inerror > 0):
                        break

            beta = beta*self.beta_r
            gamma = gamma*self.beta_r
            #count_while = count_while+1
          
        A  = self.Get_A_matrix['Affine'](u_a0,u_theta0)
        Y_ = (self.project (u_p0,Y)*A.dot(Y)) + u_t0
        
        return Y_,M,u_t0,u_theta0,u_a0,u_p0 # Y_: Y point set after matching, M:matching matrix
