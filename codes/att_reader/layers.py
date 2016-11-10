import theano
import theano.tensor as tensor
import numpy

from att_reader.utils import prfx, norm_weight, ortho_weight
from core.utils import dot, sharedX
from core.commons import Sigmoid, Tanh, Rect, global_trng, Linear, ELU

"""
    We have functions to create the layers and initialize them.
"""

# batch normalization
def bn(x, gamma=1., beta=0., prefix=""):
    assert x.ndim == 2
    mean, var = x.mean(axis=0), x.var(axis=0)
    mean.tag.bn_statistic = True
    mean.tag.bn_label = prefix + "_mean"
    var.tag.bn_statistic = True
    var.tag.bn_label = prefix + "_var"
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x,
        gamma=gamma, beta=beta,
        mean=tensor.shape_padleft(mean),
        std=tensor.shape_padleft(tensor.sqrt(var + 1e-5)))
    assert y.ndim == 2
    return y

# sequence-wise batch normalization as in Laurent et al 2015
def bn_sequence(x, gamma=1., beta=0., mask=None, prefix=""):
    assert x.ndim == 3
    n = mask.sum()
    n = theano.tensor.opt.Assert("mask all zero")(n, n > 0)
    mean = (x * mask / n).sum(axis=[0, 1])
    var = (((x - mean[None, None, :])**2) * mask / n).sum(axis=[0, 1])
    mean.tag.bn_statistic = True
    mean.tag.bn_label = prefix + "_mean"
    var.tag.bn_statistic = True
    var.tag.bn_label = prefix + "_var"
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x.reshape((x.shape[0] * x.shape[1],) + tuple(x.shape[i] for i in range(2, x.ndim))),
        gamma=gamma, beta=beta,
        mean=tensor.shape_padleft(mean),
        std=tensor.shape_padleft(tensor.sqrt(var + 1e-6))).reshape(x.shape)
    assert y.ndim == 3
    return y


profile = False
layers = {
          'ff': ('param_init_fflayer', 'fflayer'),
          'bnff': ('param_init_bnfflayer', 'bnfflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'bnlstm': ('param_init_bnlstm', 'bnlstm_layer'),
          'normff': ('param_init_normfflayer', 'normfflayer'),
          'normlstm': ('param_init_normlstm', 'normlstm_layer'),
          'lstm_tied': ('param_init_lstm_tied', 'lstm_tied_layer'),
          }


# layer
def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       params,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True,
                       use_bias=True):
    if nin is None:
        nin = options['dim_proj']

    if nout is None:
        nout = options['dim_proj']

    params[prfx(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)

    if use_bias:
        params[prfx(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            use_bias=True,
            activ='lambda x: tensor.tanh(x)',
            **kwargs):

    if use_bias:
        return eval(activ)(dot(state_below, tparams[prfx(prefix, 'W')]) + tparams[prfx(prefix, 'b')])
    else:
        return eval(activ)(dot(state_below, tparams[prfx(prefix, 'W')]))

# batch-normalized feedforward layer: linear transformation + batch normalization + point-wise nonlinearity
def param_init_bnfflayer(options,
                         params,
                         prefix='bnff',
                         nin=None,
                         nout=None,
                         ortho=True,
                         use_bias=True):
    if nin  is None: nin  = options['dim_proj']
    if nout is None: nout = options['dim_proj']
    params[prfx(prefix, 'W')] = orthogonal((nin, nout))
    params[prfx(prefix, 'gamma')] = 0.1 * numpy.ones((nout,)).astype('float32') #TODO CHECK GAMMA INIT
    if use_bias:
        params[prfx(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def bnfflayer(tparams,
              state_below,
              options,
              prefix='rconv',
              use_bias=True,
              activ='lambda x: tensor.tanh(x)',
              **kwargs):
    W     = tparams[prfx(prefix, 'W'    )]
    gamma = tparams[prfx(prefix, 'gamma')]
    b     = tparams[prfx(prefix, 'b'    )] if use_bias else 0
    return eval(activ)(bn(dot(state_below, W), gamma, b, prefix=prefix))


# GRU layer
def param_init_gru(options,
                   params,
                   prefix='gru',
                   nin=None,
                   dim=None,
                   hiero=False):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    if not hiero:
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[prfx(prefix, 'W')] = W
        params[prfx(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prfx(prefix, 'U')] = U
    Wx = norm_weight(nin, dim)
    params[prfx(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prfx(prefix, 'Ux')] = Ux
    params[prfx(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')
    return params


def gru_layer(tparams,
              state_below,
              options,
              prefix='gru',
              mask=None,
              nsteps=None,
              truncate=None,
              init_state=None,
              **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('Ux').shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    if mask.ndim == 3 and mask.ndim == state_below.ndim:
        mask = mask.reshape((mask.shape[0], \
                mask.shape[1] * mask.shape[2])).dimshuffle(0, 1, 'x')
    elif mask.ndim == 2:
        mask = mask.dimshuffle(0, 1, 'x')

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = dot(state_below, param('W')) + param('b')
    state_belowx = dot(state_below, param('Wx')) + param('bx')

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.concatenate([[init_state0] \
                                                for i in xrange(options['batch_size'])],
                                            axis=0)
            tparams[prfx(prefix, 'h0')] = init_state0

    U = tparams[prfx(prefix, 'U')]
    Ux = tparams[prfx(prefix, 'Ux')]

    def _step_slice(mask, sbelow, sbelowx, sbefore, U, Ux):
        preact = dot(sbefore, U)
        preact += sbelow

        r = Sigmoid(_slice(preact, 0, dim))
        u = Sigmoid(_slice(preact, 1, dim))

        preactx = dot(r * sbefore, Ux)

        # preactx = preactx
        preactx = preactx + sbelowx

        h = Tanh(preactx)

        h = u * sbefore + (1. - u) * h
        h = mask[:, None] * h + (1. - mask)[:, None] * sbefore

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=[U, Ux],
                                name=prfx(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# batch-normalized LSTM layer
def param_init_bnlstm(options,
                    params,
                    prefix='bnlstm',
                    nin=None,
                    dim=None):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)],
                           axis=1)
    params[prfx(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)],
                           axis=1)
    params[prfx(prefix,'U')] = U
    params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    initial_gamma, initial_beta = 0.1, 0.0
    params[prfx(prefix,'recurrent_gammas')] = initial_gamma * numpy.ones((4 * dim,)).astype('float32')
    params[prfx(prefix,'input_gammas')]     = initial_gamma * numpy.ones((4 * dim,)).astype('float32')
    params[prfx(prefix,'output_gammas')]    = initial_gamma * numpy.ones((1 * dim,)).astype('float32')
    params[prfx(prefix,'output_betas' )]    = initial_beta  * numpy.ones((1 * dim,)).astype('float32')
    return params


def bnlstm_layer(tparams, state_below,
               options,
               prefix='bnlstm',
               mask=None, one_step=False,
               init_state=None,
               init_memory=None,
               nsteps=None,
               **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[prfx(prefix, 'h0')] = init_state0

    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]
    non_seqs.extend(list(map(param, "recurrent_gammas input_gammas output_gammas output_betas".split())))

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before, *args):
        recurrent_term = bn(dot(sbefore, param('U')), gamma=param('recurrent_gammas'), prefix=prefix + "_recurrent")
        input_term = sbelow
        if not options["bn_input_not"] and not options["bn_input_sequencewise"]:
            input_term = bn(input_term, gamma=param('input_gammas'), prefix=prefix + "_input")

        preact = recurrent_term + input_term + param('b')

        i = Sigmoid(_slice(preact, 0, dim))
        f = Sigmoid(_slice(preact, 1, dim))
        o = Sigmoid(_slice(preact, 2, dim))
        c = Tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before

        c_ = bn(c, gamma=param('output_gammas'), beta=param('output_betas'), prefix=prefix + "_output")
        h = o * tensor.tanh(c_)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    lstm_state_below = dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))

    if one_step:
        # if this is the case then the sequence of lstm_state_below states
        # should be batch-normalized by the caller
        assert False
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], \
                                 mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        if options["bn_input_not"]:
            # no input normalization (but keep parameters in graph so theano doesn't die)
            lstm_state_below += 0* bn_sequence(lstm_state_below, gamma=param('input_gammas'), mask=mask, prefix=prefix + "_input")
        elif options["bn_input_sequencewise"]:
            # batch-normalize input sequence-wise
            lstm_state_below = bn_sequence(lstm_state_below, gamma=param('input_gammas'), mask=mask, prefix=prefix + "_input")

        rval, updates = theano.scan(_step,
                                    sequences=[mask, lstm_state_below],
                                    outputs_info = [init_state,
                                                    init_memory],
                                    name=prfx(prefix, '_layers'),
                                    non_sequences=non_seqs,
                                    strict=True,
                                    n_steps=nsteps)
    return rval


# ----------------------------------------------------------------------------
# NORMALIZATION PROPAGATION
# ----------------------------------------------------------------------------


def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    assert len(shape) == 2
    flat_shape = (shape[0], numpy.prod(shape[1:]))
    a = numpy.random.normal(0.0, 1.0, flat_shape)
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shp
    q = q.reshape(shape)
    value = q[:shape[0], :shape[1]].astype('float32')
    value /= numpy.sqrt((value**2).sum(axis=0))
    print('Norm of the lines of the matrix:')
    print(numpy.sqrt((value**2).sum(axis=0)))
    return value


initial_c_gamma = 1.0
initial_x_gamma = 1.0
initial_h_gamma = 1.0
initial_beta = 0.0


def param_init_normlstm(options,
                        params,
                        prefix='bnlstm',
                        nin=None,
                        dim=None):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([orthogonal((nin, dim)),
                           orthogonal((nin, dim)),
                           orthogonal((nin, dim)),
                           orthogonal((nin, dim))],
                           axis=1)
    params[prfx(prefix,'W')] = W
    U = numpy.concatenate([orthogonal((dim, dim)),
                           orthogonal((dim, dim)),
                           orthogonal((dim, dim)),
                           orthogonal((dim, dim))],
                           axis=1)
    params[prfx(prefix,'U')] = U
    params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    params[prfx(prefix,'bn_gammas')] = numpy.ones((4 * dim,)).astype('float32')
    params[prfx(prefix,'recurrent_gammas')] = initial_c_gamma * numpy.ones((4 * dim,)).astype('float32')
    params[prfx(prefix,'input_gammas')]     = initial_x_gamma * numpy.ones((4 * dim,)).astype('float32')
    params[prfx(prefix,'output_gammas')]    = initial_h_gamma * numpy.ones((1 * dim,)).astype('float32')
    params[prfx(prefix,'output_betas' )]    = initial_beta  * numpy.ones((1 * dim,)).astype('float32')
    return params


def _get_stds(): #lol
    def sigmoid(x):
        return 1./(1 + numpy.exp(-x))

    def sample(n=1000000):
        return numpy.random.randn(n).astype(numpy.float32)

    i = sigmoid(initial_x_gamma * sample() + initial_h_gamma * sample()) 
    o = sigmoid(initial_x_gamma * sample() + initial_h_gamma * sample())
    f = sigmoid(initial_x_gamma * sample() + initial_h_gamma * sample() + 0) # TODO ADD EVENTUAL FORGET BIAS
    g = numpy.tanh(initial_x_gamma * sample() + initial_h_gamma * sample())
    std_c = numpy.sqrt(((i.var() + i.mean()**2)*g.var() + i.var()*g.mean()**2)/(1 - f.var() - f.mean()**2))
    std_h = numpy.sqrt((numpy.tanh(initial_c_gamma * sample())).var() * (o.var() + o.mean()**2))
    print 'std_c', std_c
    print 'std_h', std_h
    return numpy.array(std_c, dtype='float32'), numpy.array(std_h, dtype='float32')


# NORM LSTM
def normlstm_layer(tparams, state_below,
                   options,
                   prefix='bnlstm',
                   mask=None, one_step=False,
                   init_state=None,
                   init_memory=None,
                   nsteps=None,
                   **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]

    # SEQUENCE WISE BATCH NORM BEFORE LSTM
    if options["bn_input_not"]:
        # keep parameters in graph so theano doesn't die)
        state_below += 0* bn_sequence(state_below, gamma=param('bn_gammas'),
                                      mask=mask, prefix=prefix + "_input")
    elif options["bn_input_sequencewise"]:
        state_below = bn_sequence(state_below, gamma=param('bn_gammas'), 
                                  mask=mask, prefix=prefix + "_input")


    Wx = param('W')
    Wh = param('U')
    nx = tensor.sqrt((Wx**2).sum(axis=0, keepdims=True))
    norm_Wx = Wx * param('input_gammas').dimshuffle('x', 0) / nx
    nh = tensor.sqrt((Wh**2).sum(axis=0, keepdims=True))
    norm_Wh = Wh * param('recurrent_gammas').dimshuffle('x', 0) / nh
    Wx.tag.normalize = True
    Wh.tag.normalize = True
    
    dim = norm_Wh.shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[prfx(prefix, 'h0')] = init_state0

    b = param('b')
    non_seqs = [norm_Wh]
    non_seqs.extend(list(map(param, "output_gammas output_betas".split())))

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    lstm_state_below = dot(state_below, norm_Wx) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before,
              norm_Wh, output_gammas, output_betas):
        recurrent_term = dot(sbefore, norm_Wh)
        input_term = sbelow
        preact = recurrent_term + input_term

        i = Sigmoid(_slice(preact, 0, dim))
        f = Sigmoid(_slice(preact, 1, dim))
        o = Sigmoid(_slice(preact, 2, dim))
        g = Tanh(_slice(preact, 3, dim))

        std_c, std_h = _get_stds()
        c = f * cell_before + i * g
        c_norm = (c * output_gammas) / std_c  + output_betas
        h = o * Tanh(c_norm)  / std_h

        c = mask * c + (1. - mask) * cell_before
        h = mask * h + (1. - mask) * sbefore
        return h, c


    if one_step:
        # if this is the case then the sequence of lstm_state_below states
        # should be batch-normalized by the caller
        assert False
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], \
                                 mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(_step,
                                    sequences=[mask, lstm_state_below],
                                    outputs_info = [init_state,
                                                    init_memory],
                                    name=prfx(prefix, '_layers'),
                                    non_sequences=non_seqs,
                                    strict=True,
                                    n_steps=nsteps)
    return rval


# NORM LINEAR
def param_init_normfflayer(options,
                           params,
                           prefix='normff',
                           nin=None,
                           nout=None,
                           ortho=True,
                           use_bias=True):
    if nin  is None: nin  = options['dim_proj']
    if nout is None: nout = options['dim_proj']
    params[prfx(prefix, 'W')] = orthogonal((nin, nout))
    params[prfx(prefix, 'gamma')] = 1.0 * numpy.ones((nout,)).astype('float32') #TODO INITIAL GAMMA FF LAYER
    if use_bias:
        params[prfx(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def normfflayer(tparams,
                state_below,
                options,
                prefix='rconv',
                use_bias=True,
                activ='lambda x: tensor.tanh(x)',
                **kwargs):
    W     = tparams[prfx(prefix, 'W'    )]
    gamma = tparams[prfx(prefix, 'gamma')]
    b     = tparams[prfx(prefix, 'b'    )] if use_bias else 0
    W.tag.normalize = True
    n = tensor.sqrt((W**2).sum(axis=0, keepdims=True))
    norm_W = W * gamma.dimshuffle('x', 0) / n
    # TODO: DEAL WITH THE ACTIVATION 
    raise NotImplementedError
    return eval(activ)(dot(state_below, norm_W) + b)


# ----------------------------------------------------------------------------
# BASIC LSTM
# ----------------------------------------------------------------------------

# LSTM layer
def param_init_lstm(options,
                    params,
                    prefix='lstm',
                    nin=None,
                    dim=None):
    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)],
                           axis=1)

    params[prfx(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)],
                           axis=1)

    params[prfx(prefix,'U')] = U
    params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params


def lstm_layer(tparams, state_below,
               options,
               prefix='lstm',
               mask=None, one_step=False,
               init_state=None,
               init_memory=None,
               nsteps=None,
               **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[prfx(prefix, 'h0')] = init_state0

    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before, *args):
        preact = dot(sbefore, param('U'))
        preact += sbelow
        preact += param('b')

        i = Sigmoid(_slice(preact, 0, dim))
        f = Sigmoid(_slice(preact, 1, dim))
        o = Sigmoid(_slice(preact, 2, dim))
        c = Tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    lstm_state_below = dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))
    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], \
                                 mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(_step,
                                    sequences=[mask, lstm_state_below],
                                    outputs_info = [init_state,
                                                    init_memory],
                                    name=prfx(prefix, '_layers'),
                                    non_sequences=non_seqs,
                                    strict=True,
                                    n_steps=nsteps)
    return rval


# LSTM layer
def param_init_lstm_tied(options,
                         params,
                         prefix='lstm_tied',
                         nin=None,
                         dim=None):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)

    params[prfx(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)

    params[prfx(prefix, 'U')] = U
    params[prfx(prefix, 'b')] = numpy.zeros((3 * dim,)).astype('float32')

    return params


def lstm_tied_layer(tparams,
                    state_below,
                    options,
                    prefix='lstm_tied',
                    mask=None,
                    one_step=False,
                    init_state=None,
                    init_memory=None,
                    nsteps=None,
                    **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.concatenate([[init_state0] \
                                                for i in xrange(options['batch_size'])],
                                            axis=0)
            tparams[prfx(prefix, 'h0')] = init_state0

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before):
        preact = dot(sbefore, param('U'))
        preact += sbelow
        preact += tparams[prfx(prefix, 'b')]

        f = Sigmoid(_slice(preact, 0, dim))
        o = Sigmoid(_slice(preact, 1, dim))
        c = Tanh(_slice(preact, 2, dim))

        c = f * cell_before + (1 - f) * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    state_below = dot(state_below, param('W')) + param('b')

    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[init_state,
                                                  init_memory],
                                    name=prfx(prefix, '_layers'),
                                    n_steps=nsteps)
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options,
                        params,
                        prefix='gru_cond',
                        nin=None,
                        dim=None,
                        dimctx=None):

    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options,
                            params,
                            prefix,
                            nin=nin,
                            dim=dim)
    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[prfx(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[prfx(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx)
    params[prfx(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prfx(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx)
    params[prfx(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prfx(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[prfx(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prfx(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams,
                   state_below,
                   options,
                   prefix='gru',
                   mask=None,
                   context=None,
                   one_step=False,
                   init_memory=None,
                   init_state=None,
                   context_mask=None,
                   nsteps=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prfx(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = dot(context, tparams[prfx(prefix, 'Wc_att')]) + tparams[prfx(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = dot(state_below, tparams[prfx(prefix, 'Wx')]) + \
            tparams[prfx(prefix, 'bx')]

    state_below_ = dot(state_below, tparams[prfx(prefix, 'W')]) + \
            tparams[prfx(prefix, 'b')]

    state_belowc = dot(state_below, tparams[prfx(prefix, 'Wi_att')])

    def _step_slice(mask,
                    sbelow,
                    sbelowx,
                    xc_, sbefore,
                    ctx_, alpha_,
                    pctx_, cc_,
                    U, Wc,
                    Wd_att, U_att,
                    c_tt, Ux, Wcx):
        # attention
        pstate_ = dot(sbefore, Wd_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ += xc_
        pctx__ = Tanh(pctx__)
        alpha = dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask

        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)
        # current context

        preact = dot(sbefore, U)
        preact += sbelow
        preact += dot(ctx_, Wc)
        preact = Sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = dot(sbefore, Ux)
        preactx *= r
        preactx += sbelowx
        preactx += dot(ctx_, Wcx)

        h = Tanh(preactx)

        h = u * sbefore + (1. - u) * h
        h = mask[:, None] * h + (1. - mask)[:, None] * sbefore

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[prfx(prefix, 'U')],
                   tparams[prfx(prefix, 'Wc')],
                   tparams[prfx(prefix, 'Wd_att')],
                   tparams[prfx(prefix, 'U_att')],
                   tparams[prfx(prefix, 'c_tt')],
                   tparams[prfx(prefix, 'Ux')],
                   tparams[prfx(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples, context.shape[2]),
                                                  tensor.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_,
                                                   context]+shared_vars,
                                    name=prfx(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def dropout_layer(state_before,
                  use_noise,
                  p=0.5):
    proj = tensor.switch(use_noise,
            state_before * global_trng.binomial(state_before.shape,
                                                p=p, n=1,
                                                dtype=state_before.dtype)/p,
                                                state_before)
    return proj
