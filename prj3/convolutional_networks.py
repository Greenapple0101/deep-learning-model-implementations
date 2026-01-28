import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU  # Ensure these are correctly implemented or adjust imports accordingly

def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')

class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        """
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']

        # Compute the dimensions of the output
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride

        # Pad the input
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # Initialize the output tensor
        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)

        # Perform the convolution
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW
                        window = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = torch.sum(window * w[f]) + b[f]

        cache = (x, w, b, conv_param, x_padded)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
        """
        x, w, b, conv_param, x_padded = cache
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        N, F, H_out, W_out = dout.shape

        dx_padded = torch.zeros_like(x_padded)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        # Compute db
        db = torch.sum(dout, dim=(0, 2, 3))

        # Compute dw and dx_padded
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW

                        window = x_padded[n, :, h_start:h_end, w_start:w_end]
                        dw[f] += window * dout[n, f, i, j]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

        # Remove padding from dx_padded
        if pad == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, pad:-pad, pad:-pad]

        return dx, dw, db


class FastConv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        Fast forward pass using PyTorch's built-in Conv2d.
        """
        out = torch.nn.functional.conv2d(x, w, b, stride=conv_param['stride'], padding=conv_param['pad'])
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Fast backward pass using PyTorch's autograd.grad.
        """
        x, w, b, conv_param = cache

        # Enable gradient tracking for x, w, b
        x = x.clone().detach().requires_grad_(True)
        w = w.clone().detach().requires_grad_(True)
        b = b.clone().detach().requires_grad_(True)

        # Forward pass
        out = torch.nn.functional.conv2d(x, w, b, stride=conv_param['stride'], padding=conv_param['pad'])

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=out,
            inputs=(x, w, b),
            grad_outputs=dout,
            retain_graph=False,
            create_graph=False
        )

        dx, dw, db = grads

        return dx, dw, db


class MaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out: Output data, of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride

        # Initialize output tensor
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + pool_height
                        w_start = j * stride
                        w_end = w_start + pool_width
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = torch.max(window)

        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H', W')
        - cache: A tuple of (x, pool_param) as in the forward pass

        Returns:
        - dx: Gradient with respect to x, of shape (N, C, H, W)
        """
        x, pool_param = cache
        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out, W_out = dout.shape[2], dout.shape[3]

        dx = torch.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + pool_height
                        w_start = j * stride
                        w_end = w_start + pool_width

                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = torch.max(window)
                        mask = (window == max_val)
                        dx[n, c, h_start:h_end, w_start:w_end] += mask.float() * dout[n, c, i, j]

        return dx


class BatchNorm(object):
    @staticmethod
    def backward_alt(dout, cache):
        """
        An alternative implementation of the backward pass for BatchNorm.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Values from the forward pass (x_norm, gamma, x_centered, inv_std, std, eps)

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        # Unpack the cache
        x_norm, gamma, x_centered, inv_std, std, eps = cache
        N, D = dout.shape

        # Compute dbeta and dgamma
        dbeta = torch.sum(dout, dim=0)  # Gradient of beta
        dgamma = torch.sum(dout * x_norm, dim=0)  # Gradient of gamma

        # Backprop through the normalization
        dx_norm = dout * gamma  # Scale by gamma
        dvar = torch.sum(dx_norm * x_centered * -0.5 * inv_std**3, dim=0)  # Gradient of variance
        dmean = torch.sum(dx_norm * -inv_std, dim=0) + dvar * torch.mean(-2.0 * x_centered, dim=0)  # Gradient of mean
        dx = (dx_norm * inv_std) + (dvar * 2.0 * x_centered / N) + (dmean / N)  # Combine gradients

        return dx, dgamma, dbeta

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        Inputs:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of
            features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            # Step 1: calculate mean
            sample_mean = torch.mean(x, dim=0)

            # Step 2: subtract mean vector of every training example
            x_centered = x - sample_mean

            # Step 3: calculate variance
            sample_var = torch.var(x, dim=0, unbiased=False)

            # Step 4: add eps for numerical stability, then sqrt
            std = torch.sqrt(sample_var + eps)

            # Step 5: invert std
            inv_std = 1. / std

            # Step 6: execute normalization
            x_norm = x_centered * inv_std

            # Step 7: scale and shift
            out = gamma * x_norm + beta

            cache = (x_norm, gamma, x_centered, inv_std, std, eps)

            # Update running mean and variance
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
        elif mode == 'test':
            # Use running mean and variance to normalize
            x_norm = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_norm + beta
            cache = (x_norm, gamma, x - running_mean, 1. / torch.sqrt(running_var + eps), torch.sqrt(running_var + eps), eps)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        x_norm, gamma, x_centered, inv_std, std, eps = cache

        # Step 7
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(dout * x_norm, dim=0)
        dx_norm = dout * gamma

        # Step 6
        dx_centered1 = dx_norm * inv_std
        dinv_std = torch.sum(dx_norm * x_centered, dim=0)

        # Step 5
        dstd = -dinv_std / (std ** 2)

        # Step 4
        dsample_var = 0.5 * dstd / std

        # Step 3
        dsample_mean = torch.sum(-dx_centered1, dim=0) + dsample_var * torch.mean(-2. * x_centered, dim=0)

        # Step 2
        dx_centered2 = dsample_var * 2. * x_centered / dout.shape[0]

        # Step 1
        dx1 = dx_centered1 + dx_centered2 + dsample_mean / dout.shape[0]

        dx = dx1

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        N, C, H, W = x.shape
        # Transpose to (N, H, W, C) to apply batchnorm on last dimension
        x_transposed = x.permute(0, 2, 3, 1).reshape(-1, C)
        out_transposed, cache = BatchNorm.forward(x_transposed, gamma, beta, bn_param)
        out = out_transposed.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        N, C, H, W = dout.shape
        # Transpose to (N, H, W, C) to apply batchnorm on last dimension
        dout_transposed = dout.permute(0, 2, 3, 1).reshape(-1, C)
        dx_transposed, dgamma, dbeta = BatchNorm.backward(dout_transposed, cache)
        dx = dx_transposed.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return dx, dgamma, dbeta


##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################

class FastMaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        """
        Fast forward pass using PyTorch's built-in MaxPool2d.
        """
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

        # Create a MaxPool2d layer
        pool_layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)

        # Perform pooling
        out = pool_layer(x)

        cache = (x, pool_param, pool_layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Fast backward pass using PyTorch's autograd.grad.
        """
        x, pool_param, pool_layer = cache

        # Enable gradient tracking for x
        x = x.clone().detach().requires_grad_(True)

        # Forward pass
        out = pool_layer(x)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=dout,
            retain_graph=False,
            create_graph=False
        )

        dx = grads[0]

        return dx


class Conv_ReLU(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution followed by a ReLU.
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):
    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution, a ReLU, and a pool.
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer.
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_BatchNorm_ReLU(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        """
        Convenience layer that performs a convolution, batch normalization, and ReLU.
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-batchnorm-relu convenience layer.
        """
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        """
        Convenience layer that performs a convolution, batch normalization, ReLU, and pooling.
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-batchnorm-relu-pool convenience layer.
        """
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Linear_ReLU_BatchNorm(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs a linear transform, batch normalization, and ReLU.
        """
        a, fc_cache = Linear.forward(x, w, b)
        an, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Linear_BatchNorm_ReLU(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs a linear transform, batch normalization, and ReLU.
        """
        a, fc_cache = Linear.forward(x, w, b)
        an, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float32,
                 device='cpu'):
        """
        Initialize a new network.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dims  # Channels, Height, Width

        # Ensure H and W are integers
        if isinstance(H, torch.Tensor):
            H = H.item()
        if isinstance(W, torch.Tensor):
            W = W.item()

        # Initialize weights and biases for the convolutional layer
        F = num_filters  # Number of filters
        HH = WW = filter_size  # Filter height and width
        stride = 1
        pad = (filter_size - 1) // 2  # To preserve spatial dimensions

        # Compute the size after convolution (before pooling)
        H_conv = 1 + (H + 2 * pad - HH) // stride
        W_conv = 1 + (W + 2 * pad - WW) // stride

        # Compute the size after pooling
        pool_height, pool_width, pool_stride = 2, 2, 2
        H_pool = (H_conv - pool_height) // pool_stride + 1
        W_pool = (W_conv - pool_width) // pool_stride + 1

        # Flattened dimension after pooling
        flattened_dim = F * H_pool * W_pool

        # Initialize parameters
        self.params['W1'] = torch.randn(F, C, HH, WW, dtype=dtype, device=device) * weight_scale
        self.params['b1'] = torch.zeros(F, dtype=dtype, device=device)

        self.params['W2'] = torch.randn(flattened_dim, hidden_dim, dtype=dtype, device=device) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

        self.params['W3'] = torch.randn(hidden_dim, num_classes, dtype=dtype, device=device) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

        # Store H_pool and W_pool for backward pass
        self.H_pool = H_pool
        self.W_pool = W_pool

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("Loaded checkpoint from {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # You should use functions defined in your implementation above      #
        # (e.g., Conv_ReLU_Pool) to simplify your implementation.            #
        ######################################################################
        # Forward pass: conv - relu - pool - linear - relu - linear
        out, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out_flat = out.reshape(out.shape[0], -1)
        out, cache2 = Linear_ReLU.forward(out_flat, W2, b2)
        scores, cache3 = Linear.forward(out, W3, b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        ####################################################################
        # Compute softmax loss and gradients
        loss, dscores = softmax_loss(scores, y)
        # Add regularization to loss
        loss += 0.5 * self.reg * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2) + torch.sum(W3 ** 2))

        # Backward pass
        dout = Linear.backward(dscores, cache3)
        dout, grads['W3'], grads['b3'] = dout
        dout, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout, cache2)

        # Reshape dout back to (N, F, H_pool, W_pool)
        dout = dout.reshape(out.shape[0], W1.shape[0], self.H_pool, self.W_pool)
        dx, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout, cache1)

        # Add regularization to gradients
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        ####################################################################
        #                       END OF YOUR CODE                           #
        ######################################################################

        return loss, grads


import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU  # Ensure these are correctly implemented or adjust imports accordingly

import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU  # Ensure these are correctly implemented or adjust imports accordingly

import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU  # Ensure these are correctly implemented or adjust imports accordingly

import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU  # Ensure these are correctly implemented or adjust imports accordingly

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    if K is None:
        # Linear layer
        std = torch.sqrt(torch.tensor(gain / Din, dtype=dtype, device=device))
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * std
    else:
        # Convolutional layer
        fan_in = Din * K * K
        std = torch.sqrt(torch.tensor(gain / fan_in, dtype=dtype, device=device))
        weight = torch.randn(Dout, Din, K, K, dtype=dtype, device=device) * std
    return weight

class DeepConvNet(object):

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 64],
                 max_pools=[0, 1],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer='kaiming',
                 dtype=torch.float,
                 device='cpu'):

        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype
        self.num_filters = num_filters  # Store num_filters for backward pass
        self.device = device  # Store device

        if device == 'cuda':
            device = 'cuda:0'

        C, H, W = input_dims  # Channels, Height, Width

        # Ensure H and W are integers
        if isinstance(H, torch.Tensor):
            H = H.item()
        if isinstance(W, torch.Tensor):
            W = W.item()

        for i in range(len(num_filters)):
            F = num_filters[i]
            HH, WW = 3, 3  # Kernel size for all layers
            stride, pad = 1, 1

            # Initialize convolutional layer weights and biases
            if weight_initializer == 'kaiming':
                W_param = kaiming_initializer(C, F, K=HH, relu=True, device=device, dtype=dtype)
            else:
                if isinstance(weight_scale, str) and weight_scale == 'kaiming':
                    raise ValueError("Invalid weight_scale: weight_scale should be a float, or use weight_initializer='kaiming'.")
                if not isinstance(weight_scale, (float, int)):
                    raise TypeError(f"weight_scale must be a float or int, got {type(weight_scale)}.")
                W_param = torch.randn(F, C, HH, WW, dtype=dtype, device=device) * float(weight_scale)

            b = torch.zeros(F, dtype=dtype, device=device)

            self.params[f'W{i+1}'] = W_param
            self.params[f'b{i+1}'] = b

            # Initialize batchnorm parameters if enabled
            if self.batchnorm:
                gamma = torch.ones(F, dtype=dtype, device=device)
                beta = torch.zeros(F, dtype=dtype, device=device)
                self.params[f'gamma{i+1}'] = gamma
                self.params[f'beta{i+1}'] = beta

            # Update spatial dimensions if max pooling is applied
            if i in max_pools:
                H = H // 2
                W = W // 2

            C = F  # Update input channels for the next layer

        # Ensure H and W are integers after all pooling
        H = int(H)
        W = int(W)

        # Store H_pool and W_pool for backward pass
        self.H_pool = H
        self.W_pool = W

        # Compute flattened dimension for the final linear layer
        flattened_dim = int(num_filters[-1]) * H * W

        # Initialize the final linear layer
        if weight_initializer == 'kaiming':
            W_final = kaiming_initializer(flattened_dim, num_classes, K=None, relu=False, device=device, dtype=dtype)
        else:
            W_final = torch.randn(flattened_dim, num_classes, dtype=dtype, device=device) * weight_scale
        b_final = torch.zeros(num_classes, dtype=dtype, device=device)

        self.params[f'W{self.num_layers}'] = W_final
        self.params[f'b{self.num_layers}'] = b_final

        # Initialize batchnorm parameters if applicable
        self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))] if batchnorm else []


    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
            'num_filters': self.num_filters,  # Save num_filters
            'H_pool': self.H_pool,
            'W_pool': self.W_pool,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']
        self.num_filters = checkpoint['num_filters']
        self.H_pool = checkpoint['H_pool']
        self.W_pool = checkpoint['W_pool']
        self.device = device  # Update device

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        if self.batchnorm:
            for i in range(len(self.bn_params)):
                for p in ["running_mean", "running_var"]:
                    if p in self.bn_params[i]:
                        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

        print("Loaded checkpoint from {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        # Move input to the correct device and dtype
        X = X.to(dtype=self.dtype, device=self.device)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        # Pass conv_param to the forward pass for the convolutional layer
        # All conv layers use kernel size 3 and padding 1
        conv_param = {'stride': 1, 'pad': 1}

        # Pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        cache = []
        out = X

        ##################################################################
        # TODO: Implement the forward pass for the DeepConvNet,          #
        # computing the class scores for X and storing them in the      #
        # scores variable. Use the fast versions of convolution and      #
        # max pooling layers, or the convolutional sandwich layers,      #
        # to simplify your implementation.                              #
        ##################################################################
        # Forward pass through each convolutional macro layer
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            if self.batchnorm:
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
            if (i - 1) in self.max_pools:
                # Use Conv_BatchNorm_ReLU_Pool if batchnorm is enabled
                if self.batchnorm:
                    out, cache_i = Conv_BatchNorm_ReLU_Pool.forward(
                        out, W, b, gamma, beta, conv_param, self.bn_params[i-1], pool_param)
                else:
                    # Use Conv_ReLU_Pool
                    out, cache_i = Conv_ReLU_Pool.forward(out, W, b, conv_param, pool_param)
            else:
                # Use Conv_BatchNorm_ReLU if batchnorm is enabled
                if self.batchnorm:
                    out, cache_i = Conv_BatchNorm_ReLU.forward(
                        out, W, b, gamma, beta, conv_param, self.bn_params[i-1])
                else:
                    # Use Conv_ReLU
                    out, cache_i = Conv_ReLU.forward(out, W, b, conv_param)
            cache.append(cache_i)

        # Forward pass through the final linear layer
        out = out.reshape(out.shape[0], -1)
        W = self.params[f'W{self.num_layers}']
        b = self.params[f'b{self.num_layers}']
        out, cache_final = Linear.forward(out, W, b)
        cache.append(cache_final)

        scores = out
        ##################################################################
        #                             END OF YOUR CODE                   #
        ##################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for DeepConvNet, storing the loss #
        # and gradients in the loss and grads variables. Compute data loss  #
        # using softmax, and make sure that grads[k] holds the gradients   #
        # for self.params[k]. Don't forget to add L2 regularization!      #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                #
        ####################################################################
        # Compute softmax loss and gradients
        loss, dscores = softmax_loss(scores, y)
        # Add regularization to loss
        loss += 0.5 * self.reg * sum(torch.sum(W ** 2) for k, W in self.params.items() if 'W' in k)

        # Backward pass through the final linear layer
        dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dscores, cache_final)
        grads[f'W{self.num_layers}'] += self.reg * self.params[f'W{self.num_layers}']

        # Reshape dx to [N, F, H_pool, W_pool] before passing to conv layers
        dout = dx.reshape(X.shape[0], self.num_filters[-1], self.H_pool, self.W_pool)

        # Backward pass through convolutional macro layers
        for i in range(self.num_layers - 1, 0, -1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            if self.batchnorm:
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
            cache_i = cache[i-1]
            if (i - 1) in self.max_pools:
                if self.batchnorm:
                    # Conv_BatchNorm_ReLU_Pool.backward returns (dx, dw, db, dgamma, dbeta)
                    dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, cache_i)
                    grads[f'gamma{i}'] = dgamma
                    grads[f'beta{i}'] = dbeta
                else:
                    # Conv_ReLU_Pool.backward returns (dx, dw, db)
                    dx, dw, db = Conv_ReLU_Pool.backward(dout, cache_i)
            else:
                if self.batchnorm:
                    # Conv_BatchNorm_ReLU.backward returns (dx, dw, db, dgamma, dbeta)
                    dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, cache_i)
                    grads[f'gamma{i}'] = dgamma
                    grads[f'beta{i}'] = dbeta
                else:
                    # Conv_ReLU.backward returns (dx, dw, db)
                    dx, dw, db = Conv_ReLU.backward(dout, cache_i)
            grads[f'W{i}'] = dw + self.reg * W
            grads[f'b{i}'] = db
            dout = dx  # Pass the gradient to the next layer

        ####################################################################
        #                       END OF YOUR CODE                           #
        ######################################################################

        return loss, grads




def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Typically, a higher weight scale and a larger learning rate
    # can help in overfitting on a small dataset.
    weight_scale = 1e-2
    learning_rate = 1e-2
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    from eecs598.solver import Solver  # Solver 클래스 임포트

    # 모델 생성
    model = DeepConvNet(
        input_dims=(3, 32, 32),
        num_filters=[8, 64],  # 각 층의 필터 개수
        max_pools=[0, 1],  # Pooling이 적용될 위치
        batchnorm=False,  # BatchNorm 사용 여부
        dtype=dtype,
        device=device
    )

    # Solver 생성
    solver = Solver(
        model=model,
        data=data_dict,
        num_epochs=30,
        batch_size=100,
        update_rule=adam,  # Adam 함수 직접 전달
        optim_config={
            'learning_rate': 1e-2,  # 학습률 설정
        },
        verbose=True,
        print_every=100,
        device=device  # 장치 설정 (예: 'cuda')
    )

    return solver



def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    if K is None:
        # Linear layer
        std = torch.sqrt(torch.tensor(gain / Din, dtype=dtype, device=device))
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * std
    else:
        # Convolutional layer
        fan_in = Din * K * K
        std = torch.sqrt(torch.tensor(gain / fan_in, dtype=dtype, device=device))
        weight = torch.randn(Dout, Din, K, K, dtype=dtype, device=device) * std
    return weight


##################################################################
#           Optimization Algorithms                            #
##################################################################


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w = w - config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A torch tensor of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    momentum = config['momentum']
    v = momentum * v - config['learning_rate'] * dw
    next_w = w + v
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    decay_rate = config['decay_rate']
    epsilon = config['epsilon']
    cache = config['cache']
    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
    next_w = w - config['learning_rate'] * dw / (torch.sqrt(cache) + epsilon)
    config['cache'] = cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    config['t'] += 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    learning_rate = config['learning_rate']
    epsilon = config['epsilon']

    m = config['m']
    v = config['v']
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_hat = m / (1 - beta1 ** config['t'])
    v_hat = v / (1 - beta2 ** config['t'])
    next_w = w - learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
    config['m'] = m
    config['v'] = v
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config
