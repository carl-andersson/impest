'''
@author: Carl Andersson
'''



import sys


import numpy as np
import tensorflow as tf


import processdata
import time
import scipy.io



# Custom version of matrix multiplication for tensors. Matrix multiplies the right most dimension of the x with the left most dimension of y 
def cMatMul(x:tf.Tensor,y:tf.Tensor, scope = "cMatMul"):
    with tf.name_scope(scope):
        
        xshape = tf.shape(x);
        yshape = tf.shape(y);
        
        mul_x = tf.reshape(x, [-1,xshape[-1]])
        mul_y = tf.reshape(y, [yshape[0],-1] )
        
        res = tf.matmul(mul_x,mul_y)
        
        shape = [] 
        i=0;
        for dim in x.get_shape().as_list()[:-1]:
            
            if dim is not None:
                shape.append(dim)
            else:
                shape.append(xshape[i])
            i+=1;
            
        i=1;
        for dim in y.get_shape().as_list()[1:]:
            
            if dim is not None:
                shape.append(dim)
            else:
                shape.append(yshape[i])
            i+=1;
        
        res = tf.reshape(res,shape);
        
    return res


# Preformfs a custom matrix multiplication for all matrices in  a batch
def batchMatMul(x:tf.Tensor,y:tf.Tensor, scope = "batchMatMul"):
    return tf.map_fn(lambda inx: cMatMul(inx[0],inx[1]),[x,y],dtype=tf.float32)

# Preformfs a custom matrix inversion for all matrices in  a batch
def batchMatInv(x:tf.Tensor, scope = "batchMatMul"):
    return tf.map_fn(lambda inx: tf.matrix_inverse(inx),x,dtype=tf.float32)


# Single layer of fully conected neuron 
def fulllayer(X:tf.Tensor,nOut:int,activation = tf.nn.sigmoid,bias:bool = True, name = "Fulllayer"):
    with tf.variable_scope(name):
        nIn = X.get_shape().as_list()[-1];
        nAvg = (nOut + nIn)/2;
        
        W = tf.get_variable('W', [nIn ,nOut] ,tf.float32, initializer=tf.random_normal_initializer(stddev=1.0/nAvg));  ## Xavier intialization
        res = cMatMul(X, W);
        if (bias):
            biasOffset = 0.0;
            if activation == tf.nn.relu:
                biasOffset = 1.0/nOut;
            b = tf.get_variable('b', [nOut],tf.float32,initializer=tf.constant_initializer(biasOffset,tf.float32) );
            res  += b;
        if activation is not None:
            res = activation(res);
    return res


if __name__ == '__main__':
    
    batchSize = 2500;
    
    N = seq_len = 125;

    n = resp_len = 50;

    U = tf.placeholder(tf.float32, [None,N], 'U') 
    Y = tf.placeholder(tf.float32,[None,N], 'Y') 
    
    TH = tf.placeholder(tf.float32,[None,n], 'TH')
    
    layers = [600 ,300, 200]
    
    nMats = 500;

    
    # Least squares caluclations
    with tf.variable_scope('LSModel', reuse = None) as scope:
        # Phi is represented in matrix notation instead of a sum as in the proposal
        PH = [];
        for t in range(0,N-n):
            ph = U[:,t:t+n];   # 0 indexed , reversed at a later stage
            PH.append(ph); 
            
        # reverse PH since it is defined that way
        PH = tf.reverse(tf.stack(PH,-1),[1]);   # (? x n x N-n  ) 
        R = batchMatMul(PH, tf.transpose(PH, [0,2,1])); #  (? x n x n) 
        
        Fn =  tf.expand_dims(batchMatMul(PH, Y[:,n:N]),-1);
        
        eTHLS =     tf.squeeze(tf.matrix_solve(R, Fn ),  -1);
                
        with tf.name_scope('Optimal_Regularization'):
            Pn = batchMatMul( tf.expand_dims(TH,2), tf.expand_dims(TH, 1) );
            _,varEst = tf.nn.moments(Y, [1] , keep_dims=True);
            SNR = tf.placeholder(tf.float32, [None,1] );
            
            varEst = 1/(1+SNR) * varEst;
            
            
            RHS = batchMatMul(Pn, Fn);
            LHS = batchMatMul(Pn, R) + tf.expand_dims(varEst,-1) * tf.expand_dims(tf.constant(np.eye(n,n),dtype=tf.float32 ),0)
            
            
            eTHOpt = tf.squeeze(tf.matrix_solve( LHS , RHS  ));
    
    
    with tf.variable_scope('KernelModel',reuse = None) as scope:
        
        currentBatchSize = tf.shape(U)[0];

        # Placholders for mean and std of the impulse response and Y for the training data.        
        THmean = tf.placeholder(tf.float32,[1,n],"THmean")
        THstd = tf.placeholder(tf.float32,[1,n],"THstd")
        Ymstd = tf.placeholder(tf.float32,[1,N],"Ystd")
        
        # Input to the network
        out = tf.concat([U,Y,(eTHLS-THmean)/THstd],-1) 
        # Layers   
        for i in range(1,len(layers)):
            out = tf.layers.dense(out, layers[i],tf.nn.relu, True)
        
        # Intitialization of S-vectors
        S = tf.get_variable("S", [nMats,n,1], tf.float32, tf.random_normal_initializer(0,1))*tf.expand_dims(THstd,-1) + tf.expand_dims(THmean,-1) ;
        
        #Drop out probability 
        keep_prob = tf.placeholder(tf.float32);
        sigma = tf.nn.softmax(tf.nn.dropout( fulllayer(out, nMats,None, True, "FinalLayerSigma"), keep_prob) );
 
        SQuads = batchMatMul(S,tf.transpose(S,[0,2,1]))
        

        SNRg = 5.5
        P = tf.tensordot(sigma, SQuads, ([1],[0]) )*(SNRg+1)  ;        
   
    
    
    
    with tf.variable_scope('PriorModel',reuse = None) as scope:
        eTHPrior =  tf.squeeze(tf.matrix_solve(batchMatMul(P,R) + tf.expand_dims(tf.constant(np.eye(n),dtype=tf.float32),0), batchMatMul(P,Fn)  ), -1);
    

    with tf.name_scope('cost'):
        g0mean = tf.reduce_mean(TH,-1,keep_dims=True);
        TH_MS = tf.reduce_mean(tf.square(TH-g0mean),-1);
        eTH_MSE = tf.reduce_mean(tf.square(TH-eTHPrior),-1)
        eTHopt_MSE = tf.reduce_mean(tf.square(TH-eTHOpt),-1)
        eTHls_MSE = tf.reduce_mean(tf.square(TH-eTHLS),-1)
        
        # improvefactor is equivilent to the measure S
        improvefactor = tf.reduce_mean(eTH_MSE/eTHls_MSE);
        improvefactoropt = tf.reduce_mean(eTHopt_MSE/eTHls_MSE);
             
        
        cost = tf.reduce_mean(eTH_MSE)
        lscost = tf.reduce_mean(eTHls_MSE)
        optcost = tf.reduce_mean(eTHopt_MSE)
        
        WRMSE = eTH_MSE / TH_MS ;
        
        # W is equivilent to the 'fit' measure used in matlab 
        W = tf.reduce_mean(100 * ( 1 - tf.sqrt(WRMSE) ))
        
        WRMSEopt =  eTHopt_MSE / TH_MS ;
        
        WRMSEls =  eTHls_MSE / TH_MS;
        
        
        Wopt = tf.reduce_mean(100 * ( 1 - tf.sqrt(WRMSEopt) ))
        
        Wls =   tf.reduce_mean(100 * ( 1 - tf.sqrt(WRMSEls) ))

        
    
            
    with tf.name_scope('optimizer'):
        l_rate = tf.placeholder(tf.float32,name = "l");
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)  

        opt = optimizer.minimize( improvefactor  );
        
        

    with tf.name_scope('summary'):
        tf.summary.scalar('cost_LS',    lscost);    
        tf.summary.scalar('cost_Prior' , cost );
        tf.summary.scalar('ImproveFactor' , improvefactor );
        

        tf.summary.scalar('W',W);
        
        
        
    with tf.name_scope('Validaiton'):
        
        valSummaries = tf.get_variable('valSummaries', [3],tf.float32,initializer=tf.constant_initializer(0,tf.float32) );
        valSummariesOnetime = tf.get_variable('valSummariesOntime', [4],tf.float32,initializer=tf.constant_initializer(0,tf.float32) );
        
        valSummaries_incOp = tf.assign_add(valSummaries, tf.stack(tf.cast(currentBatchSize,tf.float32)*[W,improvefactor,1]));
        valSummaries_clearOp = tf.assign(valSummaries,[0,0,0]);
        
        
        valSummariesOnetime_incOp = tf.assign_add(valSummariesOnetime, tf.stack(tf.cast(currentBatchSize,tf.float32)*[Wopt,Wls,improvefactoropt,1]));        
        
        
        tf.summary.scalar("W",valSummaries[0]/valSummaries[2],["Validation"])
        tf.summary.scalar("ImproveFactor", valSummaries[1]/valSummaries[2],["Validation"])
        
        tf.summary.scalar('W_Opt',valSummariesOnetime[0]/valSummariesOnetime[3],["Validation"]);
        tf.summary.scalar('W_LS' , valSummariesOnetime[1]/valSummariesOnetime[3] , ["Validation"] );
        tf.summary.scalar('ImproveFactor_Opt' , valSummariesOnetime[2]/valSummariesOnetime[3],["Validation"] );
        
        
    summary = tf.summary.merge_all();
    valid_summary = tf.summary.merge_all("Validation")


    # Data processing 
    
    data = processdata.getData("data_n30SNR.mat");
    
    Us = np.array(data["u"]);
    Ys = np.array(data["y"]);
    Gs = np.array(data["g"]);
    SNRs = np.array(data["SNR"])
    M = Us.shape[0];
    
     
    data_val = processdata.getData("data_val.mat")
    Us_val = np.array(data_val["u"]);
    Ys_val = np.array(data_val["y"]);
    Gs_val = np.array(data_val["g"]);
    SNRs_val = np.array(data["SNR"])
    M_val = Us_val.shape[0];
 
    
    Gmean = np.mean(Gs,0,keepdims=True);
    Gstd = np.std(Gs,0,keepdims=True);
    Ystd = np.std(Ys,0,keepdims=True);
   

    Dmat = processdata.getDmat();

    perm = [];
    saver = tf.train.Saver();
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer();
        sess.run(init)
        
        
        
        it_i = 0 
        logpath = 'log_SNR/train '+time.strftime("%c")
        train_writer = tf.summary.FileWriter(logpath, sess.graph)
        
        
        while it_i <= 5000:
            
            if  not len(perm):
                perm = np.random.permutation(M);
                
            idx = perm[:batchSize];
            perm = perm[batchSize:];
            
            l = 0.005 

            [sum_t,cp,c,_] = sess.run([summary,cost,lscost,opt],feed_dict={TH:Gs[idx], U:Us[idx], Y: Ys[idx],l_rate :  l , THmean:Gmean,THstd:Gstd,Ymstd:Ystd, keep_prob:0.7})
            
            
            
            print(it_i,cp,c, cp/c)
            
            if it_i % 10 == 0:
                train_writer.add_summary(sum_t, it_i);
                val_perm = np.random.permutation(M_val);
                
                sess.run([valSummaries_clearOp]);
                
                while len(val_perm):
                    idx = val_perm[:batchSize];
                    val_perm = val_perm[batchSize:];
                    
                    fdict = {TH:Gs_val[idx], U:Us_val[idx], Y: Ys_val[idx] ,THmean:Gmean,THstd:Gstd,Ymstd:Ystd,keep_prob:1.0,SNR:SNRs_val[idx]};
                    
                    if (it_i == 0):
                        sess.run([valSummariesOnetime_incOp],feed_dict=fdict)
                    sess.run([valSummaries_incOp],feed_dict=fdict)
                    
                    print(sess.run(valSummaries))
                
                [sum_val] = sess.run([valid_summary])
                
                train_writer.add_summary(sum_val, it_i);
                sys.stdout.flush()

            it_i += 1
    

    
        
        
        for i in range(int((M_val+batchSize-1)/batchSize)):
            sigma_part,eTh_part = sess.run([sigma,eTHPrior],feed_dict={TH:Gs_val[i*batchSize:(i+1)*batchSize], U:Us_val[i*batchSize:(i+1)*batchSize], Y: Ys_val[i*batchSize:(i+1)*batchSize], THmean:Gmean,THstd:Gstd,Ymstd:Ystd,keep_prob:1.0});
            if i == 0:
                sigmaRes = sigma_part
                eTh = eTh_part;
            else:
                sigmaRes =  np.concatenate((sigmaRes,sigma_part),0)
                eTh = np.concatenate((eTh,eTh_part),0)
        
        

        #remember to rescale the estimate with the variance and mean of input sequence and output sequence
        scipy.io.savemat(logpath+"/sigma.mat",dict( sigma = sigmaRes ));
        scipy.io.savemat(logpath+"/D.mat",dict(D=sess.run(S,feed_dict={THmean:Gmean,THstd:Gstd,Ymstd:Ystd}) ));
        scipy.io.savemat(logpath+"/eTH.mat",dict(dlTh=eTh));
        
        save_path = saver.save(sess, logpath+"/model.ckpt")
        
    
    
    
    
    
    
    
    
    
    
    