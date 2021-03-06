import tensorflow as tf
import numpy as np
import copy
import utils.process as process
class lindley(object):
    
    def __init__(self,M,cat_dims,list_discrete, dic_var_type,records_d,):
        """ lindley:__init__.
        Doc::
                
                    Args:
                        M:     
                        cat_dims:     
                        list_discrete:     
                        dic_var_type:     
                        records_d:     
                        :     
                    Returns:
                       
        """
        self._M = M
        self._cat_dims = cat_dims
        self._dic_var_type = dic_var_type
        self._list_discrete = list_discrete
        self._records_d = records_d
        
    ### function for computing reward function approximation
    def R_lindley_chain(self, i, x, mask, vae, im, loc):
        '''.
        Doc::
                
                    function for computing reward function approximation
                    i: indicates the index of x_i
                    x: data matrix
                    mask: mask of missingness
                    M: number of MC samples
                    cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
                    dic_var_type: a list that indicates the whether a variable is continuous.
                    vae: a pre-trained vae
                    im: sampled missing data, a M by N by D matrix, where M is the number of samples.
                    :return:
        '''
        im_i = im[:, :, i]
        approx_KL = 0
        im_target = im[:, :, -1]
        temp_x = copy.deepcopy(x)
        for m in range(self._M):
            temp_x[loc, i] = im_i[m, loc]
            KL_I = vae.chaini_I(temp_x[loc, :], mask[loc, :], i,self._cat_dims, self._dic_var_type,)
            temp_x[loc, -1] = im_target[m, loc]
            KL_II = vae.chaini_II(temp_x[loc, :], mask[loc, :], i,self._cat_dims, self._dic_var_type,)
            approx_KL += KL_I
            approx_KL -= KL_II
        R = approx_KL / self._M
        return R

    def completion(self, x, mask, vae,):
        '''.
        Doc::
                
                    function to generate new samples conditioned on observations
                    x: underlying partially observed data
                    mask: mask of missingness
                    M: number of MC samples
                    cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
                    dic_var_type: a list that indicates the whether a variable is continuous.
                    vae: a pre-trained vae.
                    list_discrete: list of discrete variables
                    :return: sampled missing data, a M by N by D matrix, where M is the number of samples.
        '''
        ## decompress mask
        mask_flt = mask[:, np.ndarray.flatten(np.argwhere(self._dic_var_type == 0))]
        mask_cat_oh = np.array([]).reshape(x.shape[0], 0)
        for d in range(len(self._cat_dims)):
            temp = np.ones((x.shape[0], self._cat_dims[d]))
            temp[mask[:, d] == 0, :] = 0
            mask_cat_oh = np.concatenate([mask_cat_oh, temp], 1)
        mask = np.concatenate([mask_cat_oh, mask_flt ], 1)
        im = np.zeros((self._M, x.shape[0], x.shape[1]))
        for m in range(self._M):
            #tf.reset_default_graph()
            np.random.seed(42 + m)  ### added for bar plots only
            noisy_samples = vae.im(x, mask)
            noisy_samples_mix = x*mask + noisy_samples*(1-mask)
            inverted_samples = process.invert_noise(noisy_samples_mix,self._list_discrete,self._records_d)  
            im[m, :, :] = inverted_samples
    #         im[m,:,:] = noisy_samples_mix
        return im