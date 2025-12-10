import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import scipy
#from scipy.io import savemat
import numpy as np
import scipy.io as scio

def diff(adjs, features, args, device, max_iterations=10000, threshold=1e-6):
    """
    Pre-process multiple views of adjacency matrices and feature matrices.

    Parameters:
        max_iterations (int): Maximum number of iterations to perform.
        threshold (float): Convergence threshold.

    Returns:
        torch.Tensor: A tensor containing concatenated outputs of reverse AX, reverse LX, and forward X for each view.
    """
    num_views = len(adjs)  # Number of views
    con_X_views = []
    AX=[]
    con=[]

    # Identity matrix (same for all views)
    I = torch.eye(adjs[0].shape[0], device=device)

    def iterate_matrix(mat, name, alpha, prev_X):
        with tqdm(total=max_iterations, desc=f'{name} iteration') as pbar:
            for ep in range(max_iterations):
                start_time = time.time()
                new_X = alpha * (prev_X - mat @ prev_X)#反向扩散过程X-AC
                diff = torch.norm(new_X - prev_X, p='fro').item()#新店于旧点的距离

                if diff < threshold:
                    print(f"Converged after {ep + 1} iterations ({name}), difference: {diff:.8f}")
                    return new_X

                prev_X = new_X
                pbar.set_postfix(epoch=ep + 1, diff=diff, time=time.time() - start_time)
                pbar.update(1)

        return prev_X  # Last computed value if not converged


    for view_idx in range(num_views):
        print(f" View {view_idx} Diff:")


        adj = adjs[view_idx]
        feature = features[view_idx]

        # # Compute reverse LX
        # L = I - adj  # Laplacian matrix for this view
        # reverse_LX = iterate_matrix(L, 'Reverse LX', prev_X=features)

        # Compute reverse AX using the provided formula
        reverse_AX = iterate_matrix(adj, 'Reverse AX', alpha=0.5, prev_X=feature)#X'
        #A = {'array': reverse_AX.cpu().numpy()}

        AX.append(reverse_AX)
        # Compute forward propagation
        forward_X = feature.clone()
        for _ in range(args.forward_diff_layer):
            forward_X = adj @ forward_X
        forward_X += feature

        # Concatenate reverse and forward results for this view
        con_X = torch.cat([forward_X, reverse_AX], dim=1)#组合新的特征空间
        con.append(con_X)


        # Store the result for this view
        con_X_views.append(con_X)

    # # Stack the results for all views
    # con_X_views = torch.stack(con_X_views, dim=1)
    AX=[tensor.cpu().detach().numpy() for tensor in AX]
    #AX = np.array(AX.cpu())
    con= [tensor.cpu().detach().numpy() for tensor in con]
    AX=np.hstack(AX)
    con=np.hstack(con)

    #savemat('con_X.mat', {'con_cell': np.array(con, dtype=object)})
    # savemat('reverse_AX.mat', {'con_cell': np.array(AX, dtype=object)})
    # scipy.io.savemat('reverse_AX.mat', {'data': AX})
    # scipy.io.savemat('con_X.mat', {'data': con})
    # savemat('con_X.mat', con)
    # savemat('reverse_AX.mat', AX)
    return con_X_views
