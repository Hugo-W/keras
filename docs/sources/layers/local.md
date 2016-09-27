<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/local.py#L10)</span>
### LocallyConnected1D

```python
keras.layers.local.LocallyConnected1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

The `LocallyConnected1D` layer works similarly to
the `Convolution1D` layer, except that weights are unshared,
that is, a different set of filters is applied at each different patch
of the input.
When using this layer as the first layer in a model,
either provide the keyword argument `input_dim`
(int, e.g. 128 for sequences of 128-dimensional vectors), or `input_shape`
(tuple of integers, e.g. `input_shape=(10, 128)`
for sequences of 10 vectors of 128-dimensional vectors).
Also, note that this layer can only be used with
a fully-specified input shape (`None` dimensions not allowed).

__Example__

```python
# apply a unshared weight convolution 1d of length 3 to a sequence with
# 10 timesteps, with 64 output filters
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# now model.output_shape == (None, 8, 64)
# add a new conv1d on top
model.add(LocallyConnected1D(32, 3))
# now model.output_shape == (None, 6, 32)
```

__Arguments__

- __nb_filter__: Dimensionality of the output.
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
- __border_mode__: Only support 'valid'. Please make good use of
	ZeroPadding1D to achieve same output length.
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
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).
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

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/local.py#L188)</span>
### LocallyConnected2D

```python
keras.layers.local.LocallyConnected2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

The `LocallyConnected2D` layer works similarly
to the `Convolution2D` layer, except that weights are unshared,
that is, a different set of filters is applied at each
different patch of the input.
When using this layer as the
first layer in a model, provide the keyword argument `input_shape` (tuple
of integers, does not include the sample axis), e.g.
`input_shape=(3, 128, 128)` for 128x128 RGB pictures.
Also, note that this layer can only be used with
a fully-specified input shape (`None` dimensions not allowed).

__Examples__

```python
# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image:
model = Sequential()
model.add(LocallyConnected2D(64, 3, 3, input_shape=(3, 32, 32)))
# now model.output_shape == (None, 64, 30, 30)
# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, 3, 3))
# now model.output_shape == (None, 32, 28, 28)
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
- __border_mode__: Only support 'valid'. Please make good use of
	ZeroPadding2D to achieve same output shape.
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
