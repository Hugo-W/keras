# Keras backends

## What is a "backend"?

Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Keras.

At this time, Keras has two backend implementations available: the **TensorFlow** backend and the **Theano** backend.

- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google, Inc.
- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA/MILA Lab at Université de Montréal.

In the future, we are likely to add more backend options. If you are interested in developing a new backend, get in touch!

----

## Switching from one backend to another

If you have run Keras at least once, you will find the Keras configuration file at:

`~/.keras/keras.json`

If it isn't there, you can create it.

The default configuration file looks like this:

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

Simply change the field `backend` to either `"theano"` or `"tensorflow"`, and Keras will use the new configuration next time you run any Keras code.

You can also define the environment variable ``KERAS_BACKEND`` and this will
override what is defined in your config file :

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

----

## Using the abstract Keras backend to write new code

If you want the Keras modules you write to be compatible with both Theano and TensorFlow, you have to write them via the abstract Keras backend API. Here's an intro.

You can import the backend module via:
```python
from keras import backend as K
```

The code below instantiates an input placeholder. It's equivalent to `tf.placeholder()` or `T.matrix()`, `T.tensor3()`, etc.

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```

The code below instantiates a shared variable. It's equivalent to `tf.variable()` or `theano.shared()`.

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

Most tensor operations you will need can be done as you would in TensorFlow or Theano:

```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

----

## Backend functions


### count_params


```python
count_params(x)
```


Return number of scalars in a tensor.

- __Return__: numpy integer.

----

### batch_dot


```python
batch_dot(x, y, axes=None)
```


Batchwise dot product.

batch_dot results in a tensor with less dimensions than the input.
If the number of dimensions is reduced to 1, we use `expand_dims` to
make sure that ndim is at least 2.

__Arguments__

x, y: tensors with ndim >= 2
- __axes__: list (or single) int with target dimensions

__Returns__

A tensor with shape equal to the concatenation of x's shape
(less the dimension that was summed over) and y's shape
(less the batch dimension and the dimension that was summed over).
If the final rank is 1, we reshape it to (batch_size, 1).

__Examples__

Assume x = [[1, 2], [3, 4]]   and y = [[5, 6], [7, 8]]
batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
of x.dot(y.T), although we never have to calculate the off-diagonal
elements.

Shape inference:
Let x's shape be (100, 20) and y's shape be (100, 30, 20).
If dot_axes is (1, 2), to find the output shape of resultant tensor,
loop through each dimension in x's shape and y's shape:
x.shape[0] : 100 : append to output shape
x.shape[1] : 20 : do not append to output shape,
dimension 1 of x has been summed over. (dot_axes[0] = 1)
y.shape[0] : 100 : do not append to output shape,
always ignore first dimension of y
y.shape[1] : 30 : append to output shape
y.shape[2] : 20 : do not append to output shape,
dimension 2 of y has been summed over. (dot_axes[1] = 2)

output_shape = (100, 30)

----

### gather


```python
gather(reference, indices)
```


reference: a tensor.
- __indices__: an int tensor of indices.

- __Return__: a tensor of same type as reference.

----

### sum


```python
sum(x, axis=None, keepdims=False)
```


Sum of the values in a tensor, alongside the specified axis.

----

### prod


```python
prod(x, axis=None, keepdims=False)
```


Multiply the values in a tensor, alongside the specified axis.

----

### any


```python
any(x, axis=None, keepdims=False)
```


Bitwise reduction (logical OR).

----

### all


```python
all(x, axis=None, keepdims=False)
```


Bitwise reduction (logical AND).

----

### normalize_batch_in_training


```python
normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.0001)
```


Compute mean and std for batch then apply batch_normalization on batch.

----

### batch_normalization


```python
batch_normalization(x, mean, var, beta, gamma, epsilon=0.0001)
```


Apply batch normalization on x given mean, var, beta and gamma.

----

### permute_dimensions


```python
permute_dimensions(x, pattern)
```


Transpose dimensions.

pattern should be a tuple or list of
dimension indices, e.g. [0, 2, 1].

----

### repeat_elements


```python
repeat_elements(x, rep, axis)
```


Repeat the elements of a tensor along an axis, like np.repeat.

If x has shape (s1, s2, s3) and axis=1, the output
will have shape (s1, s2 * rep, s3).

----

### resize_images


```python
resize_images(X, height_factor, width_factor, dim_ordering)
```


Resize the images contained in a 4D tensor of shape
- [batch, channels, height, width] (for 'th' dim_ordering)
- [batch, height, width, channels] (for 'tf' dim_ordering)
by a factor of (height_factor, width_factor). Both factors should be
positive integers.

----

### resize_volumes


```python
resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering)
```


Resize the volume contained in a 5D tensor of shape
- [batch, channels, depth, height, width] (for 'th' dim_ordering)
- [batch, depth, height, width, channels] (for 'tf' dim_ordering)
by a factor of (depth_factor, height_factor, width_factor).
Both factors should be positive integers.

----

### repeat


```python
repeat(x, n)
```


Repeat a 2D tensor.

If x has shape (samples, dim) and n=2,
the output will have shape (samples, 2, dim).

----

### batch_flatten


```python
batch_flatten(x)
```


Turn a n-D tensor into a 2D tensor where
the first dimension is conserved.

----

### expand_dims


```python
expand_dims(x, dim=-1)
```


Add a 1-sized dimension at index "dim".

----

### squeeze


```python
squeeze(x, axis)
```


Remove a 1-dimension from the tensor at index "axis".

----

### temporal_padding


```python
temporal_padding(x, padding=1)
```


Pad the middle dimension of a 3D tensor
with "padding" zeros left and right.

Apologies for the inane API, but Theano makes this
really hard.

----

### spatial_2d_padding


```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```


Pad the 2nd and 3rd dimensions of a 4D tensor
with "padding[0]" and "padding[1]" (resp.) zeros left and right.

----

### spatial_3d_padding


```python
spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='th')
```


Pad the 2nd, 3rd and 4th dimensions of a 5D tensor
with "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right.

----

### one_hot


```python
one_hot(indices, nb_classes)
```


Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
- __Output__: (n + 1)D one hot representation of the input
with shape (batch_size, dim1, dim2, ... dim(n-1), nb_classes)

----

### reverse


```python
reverse(x, axes)
```


Reverse a tensor along the the specified axes

----

### batch_get_value


```python
batch_get_value(xs)
```


Returns the value of more than one tensor variable,
as a list of Numpy arrays.

----

### print_tensor


```python
print_tensor(x, message='')
```


Print the message and the tensor when evaluated and return the same
tensor.

----

### stop_gradient


```python
stop_gradient(variables)
```


Returns `variables` but with zero gradient with respect to every other
variables.

----

### rnn


```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__Arguments__

- __inputs__: tensor of temporal data of shape (samples, time, ...)
(at least 3D).
- __step_function__:
- __Parameters__:
	- __input__: tensor with shape (samples, ...) (no time dimension),
	representing input for the batch of samples at a certain
	time step.
	- __states__: list of tensors.
- __Returns__:
	- __output__: tensor with shape (samples, ...) (no time dimension),
	- __new_states__: list of tensors, same length and shapes
	as 'states'.
- __initial_states__: tensor with shape (samples, ...) (no time dimension),
containing the initial values for the states used in
the step function.
- __go_backwards__: boolean. If True, do the iteration over
the time dimension in reverse order.
- __mask__: binary tensor with shape (samples, time),
with a zero for every element that is masked.
- __constants__: a list of constant values passed at each step.
- __unroll__: whether to unroll the RNN or to use a symbolic loop (`scan`).
- __input_length__: must be specified if using `unroll`.

__Returns__

A tuple (last_output, outputs, new_states).
- __last_output__: the latest output of the rnn, of shape (samples, ...)
- __outputs__: tensor with shape (samples, time, ...) where each
	entry outputs[s, t] is the output of the step function
	at time t for sample s.
- __new_states__: list of tensors, latest states returned by
	the step function, of shape (samples, ...).

----

### switch


```python
switch(condition, then_expression, else_expression)
```


condition: scalar tensor.

----

### elu


```python
elu(x, alpha=1.0)
```


 Exponential linear unit

__Arguments__

- __x__: Tensor to compute the activation function for.
- __alpha__: scalar

----

### dropout


```python
dropout(x, level, noise_shape=None, seed=None)
```


Sets entries in `x` to zero at random,
while scaling the entire tensor.

__Arguments__

- __x__: tensor
- __level__: fraction of the entries in the tensor
that will be set to 0.
- __noise_shape__: shape for randomly generated keep/drop flags,
must be broadcastable to the shape of `x`
- __seed__: random seed to ensure determinism.

----

### conv2d


```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None, filter_dilation=(1, 1))
```


2D convolution.

__Arguments__

- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __border_mode__: string, "same" or "valid".
- __dim_ordering__: "tf" or "th".
Whether to use Theano or TensorFlow dimension ordering
in inputs/kernels/ouputs.

----

### deconv2d


```python
deconv2d(x, kernel, output_shape, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```


2D deconvolution (transposed convolution).

__Arguments__

- __kernel__: kernel tensor.
- __output_shape__: desired dimensions of output.
- __strides__: strides tuple.
- __border_mode__: string, "same" or "valid".
- __dim_ordering__: "tf" or "th".
Whether to use Theano or TensorFlow dimension ordering
in inputs/kernels/ouputs.

----

### conv3d


```python
conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', volume_shape=None, filter_shape=None)
```



Run on cuDNN if available.
- __border_mode__: string, "same" or "valid".

----

### ctc_batch_cost


```python
ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


Runs CTC loss algorithm on each batch element.

__Arguments__

- __y_true__: tensor (samples, max_string_length) containing the truth labels
- __y_pred__: tensor (samples, time_steps, num_categories) containing the prediction,
	or output of the softmax
- __input_length__: tensor (samples,1) containing the sequence length for
	each batch item in y_pred
- __label_length__: tensor (samples,1) containing the sequence length for
	each batch item in y_true

__Returns__

Tensor with shape (samples,1) containing the
CTC loss of each element

----

### variable


```python
variable(value, dtype='float32', name=None)
```


Instantiate a tensor variable.

----

### placeholder


```python
placeholder(shape=None, ndim=None, dtype='float32', sparse=False, name=None)
```


Instantiate an input data placeholder variable.

----

### shape


```python
shape(x)
```


Return the shape of a tensor.

- __Warning__: type returned will be different for
Theano backend (Theano tensor type) and TF backend (TF TensorShape).

----

### eval


```python
eval(x)
```


Run a graph.

----

### zeros


```python
zeros(shape, dtype='float32', name=None)
```


Instantiate an all-zeros variable.

----

### ones


```python
ones(shape, dtype='float32', name=None)
```


Instantiate an all-ones variable.

----

### eye


```python
eye(size, dtype='float32', name=None)
```


Instantiate an identity matrix.

----

### epsilon


```python
epsilon()
```


Returns the value of the fuzz
factor used in numeric expressions.

----

### set_epsilon


```python
set_epsilon(e)
```


Sets the value of the fuzz
factor used in numeric expressions.

----

### floatx


```python
floatx()
```


Returns the default float type, as a string
(e.g. 'float16', 'float32', 'float64').

----

### cast_to_floatx


```python
cast_to_floatx(x)
```


Cast a Numpy array to floatx.

----

### image_dim_ordering


```python
image_dim_ordering()
```


Returns the image dimension ordering
convention ('th' or 'tf').

----

### set_image_dim_ordering


```python
set_image_dim_ordering(dim_ordering)
```


Sets the value of the image dimension
ordering convention ('th' or 'tf').

----

### backend


```python
backend()
```


Publicly accessible method
for determining the current backend.






