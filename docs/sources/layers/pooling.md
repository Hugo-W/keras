<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L54)</span>
### MaxPooling1D

```python
keras.layers.pooling.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```

Max pooling operation for temporal data.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, downsampled_steps, features)`.

__Arguments__

- __pool_length__: size of the region to which max pooling is applied
- __stride__: integer, or None. factor by which to downscale.
	2 will halve the input.
	If None, it will default to `pool_length`.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L174)</span>
### MaxPooling2D

```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

Max pooling operation for spatial data.

__Arguments__

- __pool_size__: tuple of 2 integers,
	factors by which to downscale (vertical, horizontal).
	(2, 2) will halve the image in each dimension.
- __strides__: tuple of 2 integers, or None. Strides values.
	If None, it will default to `pool_size`.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
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
`(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L323)</span>
### MaxPooling3D

```python
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

Max pooling operation for 3D data (spatial or spatio-temporal).

__Arguments__

- __pool_size__: tuple of 3 integers,
	factors by which to downscale (dim1, dim2, dim3).
	(2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

5D tensor with shape:
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L84)</span>
### AveragePooling1D

```python
keras.layers.pooling.AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
```

Average pooling for temporal data.

__Arguments__

- __pool_length__: factor by which to downscale. 2 will halve the input.
- __stride__: integer, or None. Stride value.
	If None, it will default to `pool_length`.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, downsampled_steps, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L216)</span>
### AveragePooling2D

```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

Average pooling operation for spatial data.

__Arguments__

- __pool_size__: tuple of 2 integers,
	factors by which to downscale (vertical, horizontal).
	(2, 2) will halve the image in each dimension.
- __strides__: tuple of 2 integers, or None. Strides values.
	If None, it will default to `pool_size`.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
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
`(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L363)</span>
### AveragePooling3D

```python
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

Average pooling operation for 3D data (spatial or spatio-temporal).

__Arguments__

- __pool_size__: tuple of 3 integers,
	factors by which to downscale (dim1, dim2, dim3).
	(2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

5D tensor with shape:
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L430)</span>
### GlobalMaxPooling1D

```python
keras.layers.pooling.GlobalMaxPooling1D()
```

Global max pooling operation for temporal data.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

2D tensor with shape: `(samples, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L416)</span>
### GlobalAveragePooling1D

```python
keras.layers.pooling.GlobalAveragePooling1D()
```

Global average pooling operation for temporal data.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

2D tensor with shape: `(samples, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L497)</span>
### GlobalMaxPooling2D

```python
keras.layers.pooling.GlobalMaxPooling2D(dim_ordering='default')
```

Global max pooling operation for spatial data.

__Arguments__

- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

2D tensor with shape:
`(nb_samples, channels)`

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L469)</span>
### GlobalAveragePooling2D

```python
keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')
```

Global average pooling operation for spatial data.

__Arguments__

- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
	It defaults to the `image_dim_ordering` value found in your
	Keras config file at `~/.keras/keras.json`.
	If you never set it, then it will be "tf".

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

2D tensor with shape:
`(nb_samples, channels)`
