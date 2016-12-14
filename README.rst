pandas_vectors
===============

These are a bunch of convenience functions to help with the use of
vectors stored in pandas dataframes.

For example, if you have a dataframe with columns of ``my_vector_x``,
``my_vector_y`` and ``my_vector_z`` then you find yourself writing code like
this:
::

  for vector in ['my_vector_x', 'my_vector_y', 'my_vector_z']:
    df[vector[:-2] + '_new' + vector[-2:]] = func(df[vector])

Now, you can write:
::

  import pandas_vectors as pv
  for vector,new in zip(pv.indexer('my_vector'), pv.indexer('my_vector_new')):
    df[new] = func(df[vector])

In fact, you can simplify it more:
::

  df = pv.transform(df, 'my_vector', '_new', func)

All the functions that take a vector as an input take a list of vectors.
::

  df = pv.magnitude(df, ['my_vector', 'my_new_vector'])

Functions that take ``df`` as the first argument return the modified ``df``.

Don't use ``_x``,``_y`` and ``_z`` for your vector names? No problem.

::

  # Set the vector suffixes to the argument given
  pv.set_vectornames(['_u', '_v', '_w'])
  # There are also some builtin shortcuts
  pv.set_vectornames('xy') # ['_x', '_y']
  pv.set_vectornames('xyz') # ['_x', '_y', '_z']
  pv.set_vectornames('pyr') # ['_p', '_y', '_r']
  pv.set_vectornames('PYR') # ['_pitch', '_yaw', '_roll']

This can also be set temporarily using ``with``:
::

  pv.set_vectornames('xyz')
  with pv.vectornames('xy'):
    df = pv.magnitude(df, 'my_vector', '_magxy') # only xy magnitude
  df = pv.magnitude(df, 'my_vector', '_mag') # xyz magnitude
