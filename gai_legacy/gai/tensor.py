import numpy as np
from typing import List, Optional, Tuple, Union, Set


class Tensor:
    def __init__(self, data, requires_grad=False, depends_on=None, creation_op=None, label=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._depends_on: List[Tuple["Tensor", "Function"]] = depends_on or []
        self._creation_op = creation_op
        self.label = label

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)

        topo = []
        visited = set()
        self._build_topo(self, visited, topo)

        self.grad = grad
        for t in reversed(topo):
            if t.requires_grad and t._depends_on:
                grads = t._creation_op.backward(t._depends_on, t.grad)
                for (dep, _), g in zip(t._depends_on, grads):
                    if dep.requires_grad:
                        dep.grad = g if dep.grad is None else dep.grad + g

    def _build_topo(self, t, visited, topo):
        if id(t) in visited:
            return
        visited.add(id(t))
        for dep, _ in t._depends_on:
            self._build_topo(dep, visited, topo)
        topo.append(t)

    def zero_grad(self):
        self.grad = None

    def reshape(self, *shape):
        return self._op(Reshape, self, shape=shape)

    def transpose(self, axes=None):
        return self._op(Transpose, self, axes=axes)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self._op(Sub, self, other)

    def __rsub__(self, other):
        other = Tensor(other)
        return other._op(Sub, other, self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self._op(Mul, self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self._op(Div, self, other)

    def __rtruediv__(self, other):
        other = Tensor(other)
        return other._op(Div, other, self)

    def __neg__(self):
        return self._op(Neg, self)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self._op(MatMul, self, other)

    def __pow__(self, power):
        return self._op(Pow, self, exponent=power)

    def __getitem__(self, idx):
        return self._op(GetItem, self, idx=idx)

    def sum(self, axis=None, keepdims=False):
        return self._op(Sum, self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._op(Mean, self, axis=axis, keepdims=keepdims)

    def _op(self, op_class, *args, **kwargs):
        requires_grad = any(
            a.requires_grad for a in args if isinstance(a, Tensor)
        )
        depends_on = [(a, kwargs) for a in args if isinstance(a, Tensor)]
        result_data = op_class.forward(*[a.data if isinstance(a, Tensor) else a for a in args], **kwargs)
        result = Tensor(result_data, requires_grad=requires_grad, depends_on=depends_on, creation_op=op_class)
        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return self.transpose()

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.data == other.data
        return self.data == other


class Function:
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(depends_on, grad):
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(a, b):
        return a + b

    @staticmethod
    def backward(depends_on, grad):
        a, _ = depends_on[0]
        b, _ = depends_on[1]
        return [_restore_broadcast(grad, a.shape), _restore_broadcast(grad, b.shape)]


def _restore_broadcast(grad, original_shape):
    g = grad.copy()
    while g.ndim > len(original_shape):
        g = g.sum(axis=0)
    for i, dim in enumerate(original_shape):
        if dim == 1 and g.shape[i] > 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(original_shape)


class Sub(Function):
    @staticmethod
    def forward(a, b):
        return a - b

    @staticmethod
    def backward(depends_on, grad):
        return [grad, -grad]


class Mul(Function):
    @staticmethod
    def forward(a, b):
        return a * b

    @staticmethod
    def backward(depends_on, grad):
        a, _ = depends_on[0]
        b, _ = depends_on[1]
        return [_restore_broadcast(grad * b.data, a.shape), _restore_broadcast(grad * a.data, b.shape)]


class Div(Function):
    @staticmethod
    def forward(a, b):
        return a / b

    @staticmethod
    def backward(depends_on, grad):
        a, _ = depends_on[0]
        b, _ = depends_on[1]
        return [_restore_broadcast(grad / b.data, a.shape), _restore_broadcast(-grad * a.data / (b.data ** 2), b.shape)]


class Neg(Function):
    @staticmethod
    def forward(a):
        return -a

    @staticmethod
    def backward(depends_on, grad):
        return [-grad]


class MatMul(Function):
    @staticmethod
    def forward(a, b):
        return a @ b

    @staticmethod
    def backward(depends_on, grad):
        a, _ = depends_on[0]
        b, _ = depends_on[1]

        # gradient w.r.t. a
        if b.ndim <= 2:
            da = grad @ b.data.T
        else:
            da = grad @ np.swapaxes(b.data, -2, -1)
        da = _reduce_to_shape(da, a.shape)

        # gradient w.r.t. b
        if a.ndim <= 2:
            db = a.data.T @ grad
        else:
            if b.ndim <= 2:
                a_flat = a.data.reshape(-1, a.shape[-1])
                grad_flat = grad.reshape(-1, grad.shape[-1])
                db = a_flat.T @ grad_flat
            else:
                db = np.swapaxes(a.data, -2, -1) @ grad
                db = _reduce_to_shape(db, b.shape)
        db = db.reshape(b.shape)

        return [da, db]


def _reduce_to_shape(arr, target_shape):
    result = arr.copy()
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and result.shape[i] > 1:
            result = result.sum(axis=i, keepdims=True)
    return result.reshape(target_shape)


class Pow(Function):
    @staticmethod
    def forward(a, exponent=2.0):
        return a ** exponent

    @staticmethod
    def backward(depends_on, grad):
        a, kw = depends_on[0]
        exponent = kw.get("exponent", 2.0)
        return [exponent * (a.data ** (exponent - 1)) * grad]


class Sum(Function):
    @staticmethod
    def forward(a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(depends_on, grad):
        a, kw = depends_on[0]
        axis = kw.get("axis")
        keepdims = kw.get("keepdims", False)
        if keepdims:
            return [np.broadcast_to(grad, a.shape)]
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                grad = np.expand_dims(grad, axis=ax)
            return [np.broadcast_to(grad, a.shape)]
        return [np.full_like(a.data, grad)]


class Mean(Function):
    @staticmethod
    def forward(a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(depends_on, grad):
        a, kw = depends_on[0]
        axis = kw.get("axis")
        keepdims = kw.get("keepdims", False)
        if axis is not None:
            n = a.data.shape[axis] if isinstance(axis, int) else np.prod([a.data.shape[ax] for ax in axis])
        else:
            n = a.data.size
        if keepdims:
            return [np.broadcast_to(grad / n, a.shape)]
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                grad = np.expand_dims(grad, axis=ax)
            return [np.broadcast_to(grad, a.shape) / n]
        return [np.full_like(a.data, grad) / n]


class Reshape(Function):
    @staticmethod
    def forward(a, shape=None):
        return a.reshape(shape)

    @staticmethod
    def backward(depends_on, grad):
        a, _ = depends_on[0]
        return [grad.reshape(a.shape)]


class Transpose(Function):
    @staticmethod
    def forward(a, axes=None):
        return np.transpose(a, axes=axes)

    @staticmethod
    def backward(depends_on, grad):
        a, kw = depends_on[0]
        axes = kw.get("axes")
        if axes is None:
            return [grad.T]
        inv_axes = np.argsort(axes)
        return [np.transpose(grad, axes=inv_axes)]


class GetItem(Function):
    @staticmethod
    def forward(a, idx):
        return a[idx]

    @staticmethod
    def backward(depends_on, grad):
        a, kw = depends_on[0]
        idx = kw.get("idx")
        out = np.zeros_like(a.data)
        out[idx] = grad
        return [out]
