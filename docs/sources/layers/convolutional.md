<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L14)</span>
### Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

Convolution operator for filtering neighborhoods of one-dimensional inputs.
When using this layer as the first layer in a model,
either provide the keyword argument `input_dim`
(int, e.g. 128 for sequences of 128-dimensional vectors),
or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
of 10 vectors of 128-dimensional vectors).

__Example__


```python
# apply a convolution 1d of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
model.add(Convolution1D(32, 3, border_mode='same'))
# now model.output_shape == (None, 10, 32)
```

__Arguments__

- __nb_filter__: Number of convolution kernels to use
	(dimensionality of the output).
- __filter_length__: The extension (spatial or temporal) of each filter.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample_length__: factor by which to subsample output.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias
	(i.e. make the layer affine rather than linear).
- __input_dim__: Number of channels/dimensions in the input.
	Either this argument or the keyword argument `input_shape`must be
	provided when using this layer as the first layer in a model.
- __input_length__: Length of input sequences, when it is constant.
	This argument is required if you are going to connect
	`Flatten` then `Dense` layers upstream
	(without it, the shape of the dense outputs cannot be computed).

__Input shape__

3D tensor with shape: `(samples, steps, input_dim)`.

__Output shape__

3D tensor with shape: `(samples, new_steps, nb_filter)`.
`steps` value might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L184)</span>
### AtrousConvolution1D

```python
keras.layers.convolutional.AtrousConvolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Atrous Convolution operator for filtering neighborhoods of one-dimensional inputs.
A.k.a dilated convolution or convolution with holes.
When using this layer as the first layer in a model,
either provide the keyword argument `input_dim`
(int, e.g. 128 for sequences of 128-dimensional vectors),
or `input_shape` (tuples of integers, e.g. (10, 128) for sequences
of 10 vectors of 128-dimensional vectors).

__Example__


```python
# apply an atrous convolution 1d with atrous rate 2 of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(AtrousConvolution1D(64, 3, atrous_rate=2, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new atrous conv1d on top
model.add(AtrousConvolution1D(32, 3, atrous_rate=2, border_mode='same'))
# now model.output_shape == (None, 10, 32)
```

__Arguments__

- __nb_filter__: Number of convolution kernels to use
	(dimensionality of the output).
- __filter_length__: The extension (spatial or temporal) of each filter.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample_length__: factor by which to subsample output.
- __atrous_rate__: Factor for kernel dilation. Also called filter_dilation
	elsewhere.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias
	(i.e. make the layer affine rather than linear).
- __input_dim__: Number of channels/dimensions in the input.
	Either this argument or the keyword argument `input_shape`must be
	provided when using this layer as the first layer in a model.
- __input_length__: Length of input sequences, when it is constant.
	This argument is required if you are going to connect
	`Flatten` then `Dense` layers upstream
	(without it, the shape of the dense outputs cannot be computed).

__Input shape__

3D tensor with shape: `(samples, steps, input_dim)`.

__Output shape__

3D tensor with shape: `(samples, new_steps, nb_filter)`.
`steps` value might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L299)</span>
### Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Convolution operator for filtering windows of two-dimensional inputs.
When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

__Examples__


```python
# apply a 3x3 convolution with 64 output filters on a 256x256 image:
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
# now model.output_shape == (None, 64, 256, 256)

# add a 3x3 convolution on top, with 32 output filters:
model.add(Convolution2D(32, 3, 3, border_mode='same'))
# now model.output_shape == (None, 32, 256, 256)
```

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __nb_row__: Number of rows in the convolution kernel.
- __nb_col__: Number of columns in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 2. Factor by which to subsample output.
	Also called strides elsewhere.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".
- __bias__: whether to include a bias
	(i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
`rows` and `cols` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L655)</span>
### AtrousConvolution2D

```python
keras.layers.convolutional.AtrousConvolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), atrous_rate=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Atrous Convolution operator for filtering windows of two-dimensional inputs.
A.k.a dilated convolution or convolution with holes.
When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

__Examples__


```python
# apply a 3x3 convolution with atrous rate 2x2 and 64 output filters on a 256x256 image:
model = Sequential()
model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2), border_mode='valid', input_shape=(3, 256, 256)))
# now the actual kernel size is dilated from 3x3 to 5x5 (3+(3-1)*(2-1)=5)
# thus model.output_shape == (None, 64, 252, 252)
```

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __nb_row__: Number of rows in the convolution kernel.
- __nb_col__: Number of columns in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 2. Factor by which to subsample output.
	Also called strides elsewhere.
- __atrous_rate__: tuple of length 2. Factor for kernel dilation.
	Also called filter_dilation elsewhere.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
`rows` and `cols` values might have changed due to padding.

__References__

- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L794)</span>
### SeparableConvolution2D

```python
keras.layers.convolutional.SeparableConvolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), depth_multiplier=1, dim_ordering='default', depthwise_regularizer=None, pointwise_regularizer=None, b_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, b_constraint=None, bias=True)
```

Separable convolution operator for 2D inputs.

Separable convolutions consist in first performing
a depthwise spatial convolution
(which acts on each input channel separately)
followed by a pointwise convolution which mixes together the resulting
output channels. The `depth_multiplier` argument controls how many
output channels are generated per input channel in the depthwise step.

Intuitively, separable convolutions can be understood as
a way to factorize a convolution kernel into two smaller kernels,
or as an extreme version of an Inception block.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

__Theano warning__


This layer is only available with the
TensorFlow backend for the time being.

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __nb_row__: Number of rows in the convolution kernel.
- __nb_col__: Number of columns in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 2. Factor by which to subsample output.
	Also called strides elsewhere.
- __depth_multiplier__: how many output channel to use per input channel
	for the depthwise convolution step.
- __depthwise_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the depthwise weights matrix.
- __pointwise_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the pointwise weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __depthwise_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the depthwise weights matrix.
- __pointwise_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the pointwise weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".
- __bias__: whether to include a bias
	(i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
`rows` and `cols` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L496)</span>
### Deconvolution2D

```python
keras.layers.convolutional.Deconvolution2D(nb_filter, nb_row, nb_col, output_shape, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Transposed convolution operator for filtering windows of two-dimensional inputs.
The need for transposed convolutions generally arises from the desire
to use a transformation going in the opposite direction of a normal convolution,
i.e., from something that has the shape of the output of some convolution
to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with said convolution. [1]

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

__Examples__


```python
# apply a 3x3 transposed convolution with stride 1x1 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), border_mode='valid', input_shape=(3, 12, 12)))
# output_shape will be (None, 3, 14, 14)

# apply a 3x3 transposed convolution with stride 2x2 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25), subsample=(2, 2), border_mode='valid', input_shape=(3, 12, 12)))
model.summary()
# output_shape will be (None, 3, 25, 25)
```

__Arguments__

- __nb_filter__: Number of transposed convolution filters to use.
- __nb_row__: Number of rows in the transposed convolution kernel.
- __nb_col__: Number of columns in the transposed convolution kernel.
- __output_shape__: Output shape of the transposed convolution operation.
	tuple of integers (nb_samples, nb_filter, nb_output_rows, nb_output_cols)
	Formula for calculation of the output shape [1], [2]:
	o = s (i - 1) + a + k - 2p, \quad a \in \{0, \ldots, s - 1\}
	- __where__:
		i - input size (rows or cols),
		k - kernel size (nb_filter),
		s - stride (subsample for rows or cols respectively),
		p - padding size,
		a - user-specified quantity used to distinguish between
		the s different possible output sizes.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano/TensorFlow function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 2. Factor by which to oversample output.
	Also called strides elsewhere.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
`rows` and `cols` values might have changed due to padding.

__References__

[1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285 "arXiv:1603.07285v1 [stat.ML]")
[2] [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
[3] [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1030)</span>
### Convolution3D

```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Convolution operator for filtering windows of three-dimensional inputs.
When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __kernel_dim1__: Length of the first dimension in the convolution kernel.
- __kernel_dim2__: Length of the second dimension in the convolution kernel.
- __kernel_dim3__: Length of the third dimension in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of Numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 3. Factor by which to subsample output.
	Also called strides elsewhere.
	- __Note__: 'subsample' is implemented by slicing the output of conv3d with strides=(1,1,1).
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).

__Input shape__

5D tensor with shape:
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)` if dim_ordering='tf'.
`new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1551)</span>
### Cropping1D

```python
keras.layers.convolutional.Cropping1D(cropping=(1, 1))
```

Cropping layer for 1D input (e.g. temporal sequence).
It crops along the time dimension (axis 1).

__Arguments__

- __cropping__: tuple of int (length 2)
	How many units should be trimmed off at the beginning and end of
	the cropping dimension (axis 1).

__Input shape__

3D tensor with shape (samples, axis_to_crop, features)

__Output shape__

3D tensor with shape (samples, cropped_axis, features)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1591)</span>
### Cropping2D

```python
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), dim_ordering='default')
```

Cropping layer for 2D input (e.g. picture).
It crops along spatial dimensions, i.e. width and height.

__Arguments__

- __cropping__: tuple of tuple of int (length 2)
	How many units should be trimmed off at the beginning and end of
	the 2 cropping dimensions (width, height).
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

4D tensor with shape:
(samples, depth, first_axis_to_crop, second_axis_to_crop)

__Output shape__

4D tensor with shape:
(samples, depth, first_cropped_axis, second_cropped_axis)

__Examples__


```python
# Crop the input 2D images or feature maps
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)), input_shape=(3, 28, 28)))
# now model.output_shape == (None, 3, 24, 20)
model.add(Convolution2D(64, 3, 3, border_mode='same))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# now model.output_shape == (None, 64, 20, 16)

```


----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1676)</span>
### Cropping3D

```python
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), dim_ordering='default')
```

Cropping layer for 3D data (e.g. spatial or saptio-temporal).

__Arguments__

- __cropping__: tuple of tuple of int (length 3)
	How many units should be trimmed off at the beginning and end of
	the 3 cropping dimensions (kernel_dim1, kernel_dim2, kernerl_dim3).
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

5D tensor with shape:
(samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)

__Output shape__

5D tensor with shape:
(samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)


----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1231)</span>
### UpSampling1D

```python
keras.layers.convolutional.UpSampling1D(length=2)
```

Repeat each temporal step `length` times along the time axis.

__Arguments__

- __length__: integer. Upsampling factor.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, upsampled_steps, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1263)</span>
### UpSampling2D

```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), dim_ordering='default')
```

Repeat the rows and columns of the data
by size[0] and size[1] respectively.

__Arguments__

- __size__: tuple of 2 integers. The upsampling factors for rows and columns.
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1326)</span>
### UpSampling3D

```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), dim_ordering='default')
```

Repeat the first, second and third dimension of the data
by size[0], size[1] and size[2] respectively.

__Arguments__

- __size__: tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

5D tensor with shape:
`(samples, channels, dim1, dim2, dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, dim1, dim2, dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(samples, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1393)</span>
### ZeroPadding1D

```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```

Zero-padding layer for 1D input (e.g. temporal sequence).

__Arguments__

- __padding__: int
	How many zeros to add at the beginning and end of
	the padding dimension (axis 1).

__Input shape__

3D tensor with shape (samples, axis_to_pad, features)

__Output shape__

3D tensor with shape (samples, padded_axis, features)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1428)</span>
### ZeroPadding2D

```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='default')
```

Zero-padding layer for 2D input (e.g. picture).

__Arguments__

- __padding__: tuple of int (length 2)
	How many zeros to add at the beginning and end of
	the 2 padding dimensions (axis 3 and 4).
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

4D tensor with shape:
(samples, depth, first_axis_to_pad, second_axis_to_pad)

__Output shape__

4D tensor with shape:
(samples, depth, first_padded_axis, second_padded_axis)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1488)</span>
### ZeroPadding3D

```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), dim_ordering='default')
```

Zero-padding layer for 3D data (spatial or spatio-temporal).

__Arguments__

- __padding__: tuple of int (length 3)
	How many zeros to add at the beginning and end of
	the 3 padding dimensions (axis 3, 4 and 5).
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

5D tensor with shape:
(samples, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)

__Output shape__

5D tensor with shape:
(samples, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)
