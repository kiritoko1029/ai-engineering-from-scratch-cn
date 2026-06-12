# 向量、矩阵与运算

> 每个神经网络本质上都只是带有额外步骤的矩阵乘法。

**类型：** 构建
**语言：** Python、Julia
**先修要求：** 第一阶段，第01课（线性代数基础概念）
**时长：** 约60分钟

## 学习目标

- 构建一个具备逐元素运算、矩阵乘法、转置、行列式计算以及求逆功能的 Matrix 类  
- 区分逐元素乘法与矩阵乘法，并说明两种运算的适用场景  
- 仅使用自实现的 Matrix 类来实现一个密集型神经网络层（`relu(W @ x + b)`）  
- 解释广播规则以及偏置项在神经网络框架中的添加机制

## 问题所在

你想构建一个神经网络。你阅读了相关代码，看到了如下内容：

```
output = activation(weights @ input + bias)
```

那个 `@` 符号表示矩阵乘法。`weights` 是一个矩阵，而 `input` 则是一个向量。如果你不了解这些运算的含义，这一行代码就如同魔法一般；但如果你已经熟悉它们，这实际上就是用三次运算完成了层级的整个前向传播过程。

模型处理的每张图像都是像素值构成的矩阵，每个词嵌入也是一个向量。在任何一个神经网络中，每一层都代表着一次矩阵变换。正如不理解变量就无法编写代码一样，如果不精通矩阵运算，也就无法构建人工智能系统。

本课程将从零开始帮助你掌握这些矩阵运算技能。

## 概念概述

### 向量：有序的数字列表

向量是由数值构成的列表，同时具有方向和大小。在人工智能领域中，向量用于表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

二维向量 `[3, 4]` 表示平面上的坐标点 (3, 4)。其长度（模长）为 5，符合 3-4-5 直角三角形的关系。

### 矩阵：数字网格

矩阵是一种二维网格，由行和列组成。一个 m × n 的矩阵包含 m 行和 n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵用于将输入向量转换为输出向量。一个具有784个输入和128个输出的层会使用一个128×784大小的权重矩阵。

### 为何形状至关重要

矩阵乘法遵循严格的规则：`(m x n) @ (n x p) = (m x p)`。内层维度的大小必须相等。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果在 PyTorch 中遇到形状不匹配的错误，原因就在于此。

### 操作映射表

| 操作 | 功能描述 | 神经网络中的应用 |
|-----------|-------------|-------------------|
| 加法 | 元素级组合 | 为输出添加偏置项 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度值 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 交换行与列 | 反向传播 |
| 行列式 | 单一数值摘要 | 判断矩阵是否可逆 |
| 逆矩阵 | 撤销变换 | 求解线性方程组 |
| 单位矩阵 | 不产生任何影响的矩阵 | 初始化参数、残差连接 |

### 元素级乘法与矩阵乘法

这一区别常常让初学者感到困惑。

逐元素运算：将对应位置的元素相乘。两个矩阵的形状必须完全相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：即行向量与列向量的点积运算。两个矩阵的内维度必须相等。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的操作，不同的结果，不同的规则。

### 广播

当将偏置向量添加到输出矩阵时，两者形状不匹配。广播机制会自动扩展较小的数组以使其适配。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

所有现代框架都会自动执行此操作。理解这一机制有助于在图形显示异常但代码仍能运行的情况下避免混淆。

```figure
vector-projection
```

## 构建它

### 步骤 1：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步骤 2：包含核心操作的矩阵类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步骤 3：验证其运行效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步骤 4：连接神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这是一个单层全连接层：`output = relu(W @ x + b)`。所有神经网络中的每一层全连接层都是如此工作的。

## 使用它

NumPy 仅需更少的代码行即可完成上述所有操作，且执行速度快上数个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符会调用 `__matmul__` 方法。NumPy 则通过用 C 和 Fortran 编写的优化过的 BLAS 程序来实现该运算。虽然数学原理相同，但执行速度可快 100 倍。

NumPy 中的广播机制：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy会自动将一维的偏置值广播到两行中。这就是所有神经网络框架中添加偏置值的原理。

## 发布它

本课旨在通过几何直观来生成用于教授矩阵运算的提示语。详情请参见 `outputs/prompt-matrix-operations.md`。

此处实现的 Matrix 类是我们将在第 3 阶段第 10 课中构建的微型神经网络框架的基础。

## 练习题

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 的结果，并确认其为单位矩阵。请使用三个不同的 2x2 矩阵进行测试。当行列式为零时会发生什么？

2. **实现 3x3 矩阵的逆矩阵计算。** 扩展 Matrix 类，利用伴随矩阵法计算 3x3 矩阵的逆矩阵。并将结果与 NumPy 的 `np.linalg.inv` 函数进行对比测试。

3. **构建双层神经网络。** 仅使用你自定义的 Matrix 类（不得使用 NumPy），构建一个具有三层结构的神经网络：输入层（3个节点） -> 隐藏层（4个节点） -> 输出层（2个节点）。为各层权重初始化随机值，执行前向传播，并确认所有数据的形状均正确。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 向量 | “箭头” | 数字的有序列表。在人工智能中：高维空间中的一个点。 |
| 矩阵 | “数字表格” | 线性变换。它将一个空间中的向量映射到另一个空间。 |
| 矩阵乘法 | “直接把数字相乘” | 第一个矩阵的每一行与第二个矩阵的每一列进行点积运算。顺序很重要。 |
| 转置 | “翻转它” | 交换行和列。将 m x n 矩阵转换为 n x m 矩阵。在反向传播中至关重要。 |
| 行列式 | “矩阵中的某个数值” | 衡量矩阵对面积（二维）或体积（三维）的缩放程度。行列式为 0 表示该变换会挤压某一维度。 |
| 逆矩阵 | “撤销矩阵变换” | 能够逆转该变换的矩阵。仅当行列式不为 0 时才存在。 |
| 单位矩阵 | “无聊的矩阵” | 相当于乘以 1 的矩阵。用于残差连接结构（如 ResNets）中。 |
| 广播机制 | “神奇的形状匹配” | 通过沿缺失维度重复元素，将较小的数组扩展为与较大数组相同的形状。 |
| 元素级运算 | “普通乘法” | 对对应位置的元素进行相乘。两个数组必须具有相同的形状（或能够通过广播机制匹配）。 |

## 延伸阅读

- [3Blue1Brown：线性代数精要](https://www.3blue1brown.com/topics/linear-algebra) - 为此处讲解的每一项运算提供可视化直观理解
- [NumPy 关于广播机制的文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 所遵循的精确规则
- [斯坦福大学 CS229 线性代数复习资料](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简明参考手册
