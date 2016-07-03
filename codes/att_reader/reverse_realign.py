# code to generate a left-justified reversed batch for bidirectional LSTM,
# padded with repetitions of the sequence ("wrap" in numpy.pad parlance)
import theano, numpy
from theano import tensor

def bring_to_front(seq, items):
    lst = list(seq)
    for item in reversed(items):
        lst.insert(0, lst.pop(lst.index(item)))
    return lst

#def reverse_realign(batch, mask, batch_axis=0, time_axis=1):
#    return batch

def reverse_realign(batch, mask, batch_axis=0, time_axis=1):
    batch_dims = bring_to_front(list(range(batch.ndim)), [batch_axis, time_axis])
    mask_dims  = bring_to_front(list(range( mask.ndim)), [batch_axis, time_axis])

    print batch_dims, mask_dims, batch_axis, time_axis, batch.ndim, mask.ndim

    batch = batch.dimshuffle(*batch_dims)
    mask = mask.dimshuffle(*mask_dims)

    batch = _reverse_realign(batch, mask)

    print [batch_dims[batch_dims[i]] for i in batch_dims]

    # invert dimshuffle
    batch = batch.dimshuffle(*[batch_dims[batch_dims[i]] for i in batch_dims])

    return batch

def _reverse_realign(batch, mask):
    # assuming shape is (batch, time, ...)
    n_samples = batch.shape[0]
    n_timesteps = batch.shape[1]
    lengths = tensor.iround(mask.sum(axis=1, keepdims=True))
    if lengths.ndim > 2:
        lengths = lengths.flatten(ndim=2)
    indices = tensor.repeat(tensor.arange(n_timesteps)[None, :], n_samples, axis=0)
    reverse_indices = lengths - 1 - indices % lengths
    flat_indices = (tensor.arange(n_samples)[:, None] * n_timesteps + reverse_indices).flatten()
    if False:
        # flat_indices doesn't take axes > 2 into account; expand it to include the indices for higher axes
        rest_size = tensor.prod(batch.shape[2:])
        flat_indices = ((flat_indices * rest_size)[:, None] + tensor.arange(rest_size)[None, :]).flatten()
        flat_batch = batch.flatten()
    else:
        flat_batch = batch.reshape((batch.shape[0] * batch.shape[1], -1))
    flat_indices.name = "flat_indices"
    flat_batch.name = "flat_batch"
    flat_reverse_batch = flat_batch[flat_indices]

    #from theano.tests.breakpoint import PdbBreakpoint
    #breakpointOp = PdbBreakpoint("Raise1")
    #flat_reverse_batch = breakpointOp(1., flat_reverse_batch, batch, mask)[0]
    #import pdb; pdb.set_trace()

    reverse_batch = flat_reverse_batch.reshape(batch.shape)
    print reverse_batch.ndim, mask.ndim
    #from theano.tests.breakpoint import PdbBreakpoint
    #breakpointOp = PdbBreakpoint("Raise1")
    #reverse_batch = breakpointOp(1., reverse_batch, batch, mask)[0]
    reverse_batch = reverse_batch * mask.astype(reverse_batch.dtype)
    reverse_batch.name = "reverse_batch"

    return reverse_batch

def test_reverse_realign(n_samples, n_timesteps):
    def pad_batch(examples):
        batch = numpy.stack([numpy.pad(example,
                                 pad_width=([(0, n_timesteps - len(example))] +
                                            [(0, 0)] * (example.ndim - 1)),
                                 mode="wrap")
                          for example in examples])
        mask = numpy.zeros((n_samples, n_timesteps))
        for i in range(n_samples):
            mask[i, numpy.arange(len(examples[i]))] = 1.
        return batch, mask

    def construct_batch():
        lengths = numpy.random.randint(3, n_timesteps, size=(n_samples,))
        examples = [numpy.arange(length) for length in lengths]
        batch, mask = pad_batch(examples)
        return batch.astype(numpy.float32), mask.astype(numpy.float32)

    x = tensor.matrix("x")
    mask = tensor.matrix("mask")

    npx, npmask = construct_batch()
    theano.config.compute_test_value = "warn"
    x.tag.test_value = npx
    mask.tag.test_value = npmask

    inputs = [x, mask]
    input_values = [npx, npmask]

    x = x.T
    npx = npx.T
    mask = mask.T
    npmask = npmask.T

    # optionally embed to test 3d behavior
    embed = False
    if embed:
        W = theano.shared(numpy.random.rand(1, 10), name="W")
        e = tensor.dot(x[:, :, None], W)
        refe = numpy.dot(npx[:, :, None], W.get_value())
    else:
        e = x
        refe = npx

    re = reverse_realign(e, mask, time_axis=0, batch_axis=1)

    refre, refmask = pad_batch([refe[npmask[:, i].astype(numpy.bool), i][::-1] for i in range(n_samples)])
    refre = numpy.rollaxis(refre, 0, 2)
    refmask = refmask.T

    f = theano.function(inputs, [e, re])
    npe, npre = f(*input_values)

    assert numpy.allclose(npmask, refmask)

    if not numpy.allclose(npre, refre):
        import matplotlib.pyplot as plt
        for var in "npe npre refre npmask".split():
            plt.matshow(locals()[var])
            plt.title(var)
        plt.show()

    assert numpy.allclose(npre, refre)

if __name__ == "__main__":
    test_reverse_realign(10, 20)
