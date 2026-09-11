Документация пакета jacobi-eigen
================================

Пакет реализует **метод Якоби (вращений)** для вычисления всех собственных
чисел и собственных векторов вещественной симметричной матрицы.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   theory
   api

Быстрый старт
-------------

.. code-block:: python

   import numpy as np
   from jacobi_eigen import jacobi_eigh

   A = np.array([[4.0, 1.0, 0.0],
                 [1.0, 3.0, 1.0],
                 [0.0, 1.0, 2.0]])

   result = jacobi_eigh(A)
   print(result.eigenvalues)          # [1.267949 3. 4.732051]
   print(result.residual_norm(A))     # ~1e-15

Индексы и таблицы
-----------------

* :ref:`genindex`
* :ref:`modindex`
