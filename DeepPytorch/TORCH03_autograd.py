#!/usr/bin/env python
# coding: utf-8

# # TORCH 03. Autograd: automotic differentiation
# - `autograd` package provides automatic differentiation for all operations on Tensors

# ## Tensor

# In[1]:


import torch
print(torch.__version__)


# In[2]:


# Create a tensor and set `requires_grad=True` to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)


# In[3]:


# Do a tensor operation
y = x + 2
print(y, end='\n\n')

# y was created as a result of an operation, so it has a `grad_fn`
print(y.grad_fn, end='\n\n')

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)


# In[4]:


# `.requires_grad_( ... )` changes an existing Tensor's `requires_grad` flag in-place.
# The input flag defaults to `False` if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)    # default is False
a.requires_grad_(True)    # Set requres_grad as True
print(a.requires_grad)    # It will be a True
b = (a * a).sum()
print(b.grad_fn)          # Since requires_grad is True, exists grad_fn


# ## Gradients

# In[5]:


out


# In[6]:


# Since `out` contains a single scalar,
# `out.backward()` is equivalent to `out.backward(torch.tensor(1,))`.
out.backward()  # 역전파 실시


# In[7]:


# Print gradients d(out)/dx
print(x.grad)


# $$o = {\cfrac{1}{4}}{\sum_{i}{z_{i}}}$$
# $$z_{i} = 3(x_{i}+2)^{2}$$
# $$z_{i}|_{x_{i}=1}=27$$
# $$\text{Therefore,}{\;}\cfrac{{\partial}o}{{\partial}x_{i}}=\cfrac{3}{2}(x_{i}+2)$$
# $${\cfrac{{\partial}o}{{\partial}x_{i}}}\bigg{|}_{x_{i}=1}=\cfrac{9}{2}=4.5$$

# $$\text{Mathematically, if you have a vector valued function}\;\vec{y}=f(\vec{x}),$$
# $$\text{then the gradient of}\;\vec{y}\text{ with respect to}\;\vec{x}\text{ is a jacobian matrix:}$$
# $$J=\begin{pmatrix}
# \cfrac{{\partial}y_{1}}{{\partial}x_{1}} & \cdots & \cfrac{{\partial}y_{1}}{{\partial}x_{n}}\\
# \vdots & \ddots & \vdots\\
# \cfrac{{\partial}y_{m}}{{\partial}x_{1}} & \cdots & \cfrac{{\partial}y_{m}}{{\partial}x_{n}}\\
# \end{pmatrix}$$

# $$\text{Generally speaking, `torch.autograd` is an engine for computing vector-Jacobian product.}$$
# $$\text{That is, given any vector }v=(v1{\quad}v2{\quad}{\cdots}{\quad}v_{m})^{T}\text{, compute the product }v^{T}{\cdot}J$$
# $$\text{If }v\text{ happens to be the gradient of a scalar function }l=g\big{(}\vec{y}\big{)}\text{, that is, }v=\bigg{(}\cfrac{{\partial}l}{{\partial}y_{1}}\;\cdots\;\cfrac{{\partial}l}{{\partial}y_{m}}\bigg{)}^{T}\text{,}$$
# $$\text{then by the chain rule, the vector-Jacobian product would be the gradient of }l\text{ with respect to }\vec{x}\text{:}$$
# $$J^{T}\cdot{v}=\begin{pmatrix}
# \cfrac{{\partial}y_{1}}{{\partial}x_{1}} & \cdots & \cfrac{{\partial}y_{m}}{{\partial}x_{1}}\\
# \vdots & \ddots & \vdots\\
# \cfrac{{\partial}y_{1}}{{\partial}x_{n}} & \cdots & \cfrac{{\partial}y_{m}}{{\partial}x_{n}}\\
# \end{pmatrix}\begin{pmatrix}
# \cfrac{{\partial}l}{{\partial}y_{1}}\\
# \vdots\\
# \cfrac{{\partial}l}{{\partial}y_{m}}\\
# \end{pmatrix}=\begin{pmatrix}
# \cfrac{{\partial}l}{{\partial}x_{1}}\\
# \vdots\\
# \cfrac{{\partial}l}{{\partial}x_{n}}\\
# \end{pmatrix}$$
# $$\text{(Note that }v^{T}\cdot{J}\text{ gives a row vector which can be treated as a column vector by taking }{J}^{T}\cdot{v}\text{)}$$
# $$\text{This characteristic of vector-Jacobian product makes it very convenient to feed external gradients into a model that has non-scalar output.}$$

# In[8]:


# vector-Jacobian product example
x = torch.randn(3, requires_grad=True)
print('x :', x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print('y :', y)


# In[9]:


# y is not a scalar,
# `torch.autograd` could not compute the full jacobian directly,
# but if we just want the vector-jacobian product,
# simply pass the vector to `backward` as argument.
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)


# In[10]:


# Stop autograd from tracking history on Tensors with `.requires_grad=True`
# By wrapping the code block in `with torch.no_grad():`
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


# In[11]:


# Or by using `.detach()` to get a new Tensor with the same content
# but that does not require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


# ## AUTOMATIC DIFFERENTIATION PACKAGE - TORCH.AUTOGRAD

# ## `__init__.py`

# In[12]:


# SOURCE CODE FOR TORCH.AUTOGRAD
import torch
import warnings

from torch.autograd.variable import Variable
from torch.autograd.function import Function, NestedIOFunction
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.autograd.grad_mode import no_grad, enable_grad, set_grad_enabled
from torch.autograd.anomaly_mode import detect_anomaly, set_detect_anomaly
from torch.autograd import profiler


# In[13]:


__all__ = ['Variable', 'Function', 'backward', 'grad_modea']


# In[14]:


def _make_grads(outputs, grads):
    new_grads = []
    for out, grad in zip(outputs, grads):
        # Gradient가 torch.Tensor객체 일 경우
        if isinstance(grad, torch.Tensor):
            # out과 grad의 shape 체크
            if not out.shape == grad.shape:
                raise RuntimeError("Mismatch in shape: grad_output["
                                   + str(grads.index(grad)) + "] has a shape of "
                                   + str(grad.shape) + " and output["
                                   + str(outputs.index(out)) + "] has a shape of "
                                   + str(out.shape) + ".")
            new_grads.append(grad)
        # Gradient가 None일 경우
        elif grad is None:
            # requires_grad == True :
            if out.requires_grad:
                # out이 scalar가 아닐 경우 에러 처리
                if out.numel() != 1:
                    '''
                    # Returns the total number of elements in the `input` tensor.
                    >>> a = torch.randn(1, 2, 3, 4, 5)
                    >>> torch.numel(a)
                    120
                    >>> a = torch.zeros(4, 4)
                    >>> torch.numel(a)
                    16
                    '''
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                
                new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
            # requires_grad == False : None 추가
            else:
                new_grads.append(None)
        # Gradient가 torch.Tensor 혹은 None이 아닐 경우 에러처리
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    # tuple로 return
    return tuple(new_grads)


# In[15]:


def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False,
             grad_variables=None):
    """
    Computes the sum of gradients of given tensors w.r.t. graph leaves.
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeErorr("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")
    
    tensors = (tensors, ) if isinstance(tensors, torch.Tensor) else tuple(tensors)
    
    if grad_tensors is None:
        grad_tensors = [None] * len(tensors)
    elif isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    else:
        grad_tensors = list(grad_tensors)
    
    grad_tensors = _make_grads(tensors, grad_tensors)
    if retain_graph is None:
        retain_graph = create_graph
        
    # 위에서 설정만 잡아주고 돌리는건 C++ Imperative Engine에서 돌린다.
    Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True)  # allow_unreachable flag


# In[16]:


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
         only_inputs=True, allow_unused=False):
    """
    Computes and returns the sum of gradients of outputs w.r.t. the inputs.
    """
    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now "
                      "(defualts to True). To accumulate gradient for other "
                      "parts of the graph, please use torch.autograd.backward.")
    
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    
    if grad_outpus is None:
        grad_outputs = [None] * len(outputs)
    elif isinstance(grad_outputs, torch.Tensor):
        grad_outputs = [grad_outputs]
    else:
        grad_outputs = list(grad_outputs)
    """
    아니, 지금 elif랑 else랑 차이가 뭐야?
    ```python
    >>> a
    tensor([[1.8083, 1.6985],
            [2.0055, 1.6993]], requires_grad=True)
    >>> [a]
    [tensor([[1.8083, 1.6985],
             [2.0055, 1.6993]], requires_grad=True)]
    >>> list(a)
    [tensor([1.8083, 1.6985], grad_fn=<SelectBackward>),
     tensor([2.0055, 1.6993], grad_fn=<SelectBackward>)]
    ```
    때문에 위와 같이 다르게 처리해줘야한다!
    """
    grad_outputs = _make_grads(outputs, grad_outputs)
    
    if retain_graph is None:
        retain_graph = create_graph
    
    return Variable._execution_engine.run_backward(
        outputs, grad_outputs, retain_graph, create_graph,
        inputs, allow_unused)


# In[17]:


# This function applies in case of gradient checkpointing for memory
# optimization. Currently, for gradient checkpointing, we only support imperative
# backwards call i.e. torch.autograd.backward() and the torch.autograd.grad() won't
# work. The reason being that: torch.autograd.grad() only calculates the grads
# for the inputs that are passed by user but it doesn't calculate grad for
# anything else e.g. model parameters like weights, bias etc. However, for
# torch.autograd.backward(), we would actually compute the grad for the weights as well.
#
# This function returns whether the checkpointing is valid i.e. torch.autograd.backward
# or not i.e. torch.autograd.grad. The implementation works by maintaining a thread
# local variable in torch/csrc/autograd/engine.cpp which looks at the NodeTask
# in the stack and before a NodeTask is executed in evaluate_function, it
# checks for whether reentrant backwards is imperative or not.
# See https://github.com/pytorch/pytorch/pull/4594 for more discussion/context
def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()


# In[18]:


def variable(*args, **kwargs):
    warnings.warn("torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead")
    return torch.tensor(*args, **kwargs)


# ```python
# if not torch._C._autograd_init():
#     raise RuntimeError("autograd initialization failed")
# ```

# ## Locally disabling gradient computation

# ### `torch.autograd.no_grad`
# - Context-manager that disabled gradient calculation

# In[19]:


no_grad


# In[27]:


x = torch.tensor([1], requires_grad=True)


# https://github.com/pytorch/pytorch/issues/17345
# Bug in here:
# ```
# pytorch/torch/multiprocessing/reductions.py
# 
# 83        t = torch.nn.parameter.Parameter(t) 
# 84    t.requires_grad = requires_grad 
# ```

# In[28]:


x = torch.tensor([1.0], requires_grad=True)
x


# In[31]:


x2 = x * 2
x.requires_grad


# In[32]:


with torch.no_grad():
    y = x * 2
y.requires_grad


# In[33]:


@torch.no_grad()
def doubler(x):
    return x * 2

z = doubler(x)
z.requires_grad


# ### `torch.autograd.enable_grad`
# - Context-manager that enables gradient calculation

# In[34]:


enable_grad


# In[35]:


x = torch.tensor([1.0], requires_grad=True)

with torch.no_grad():
    with torch.enable_grad():
        y = x * 2
y.requires_grad


# In[36]:


y.backward()


# In[37]:


x.grad


# In[38]:


@torch.enable_grad()
def doubler(x):
    return x * 2

with torch.no_grad():
    z = doubler(x)
    
z.requires_grad


# ### `torch.autograd.set_grad_enabled`
# - Context-manager that sets gradient calculation to on or off
# - When using `enabled_grad` context manager, `set_grad_enabled(False)` has no effect.

# In[40]:


x = torch.tensor([1.0], requires_grad=True)
is_train = False

with torch.set_grad_enabled(is_train):
    y = x * 2
    
y.requires_grad


# In[41]:


torch.set_grad_enabled(True)
y = x * 2
y.requires_grad


# In[42]:


torch.set_grad_enabled(False)
y = x * 2
y.requires_grad


# In[47]:


# 응용해보자
x = torch.tensor([1.0], requires_grad=True)

@torch.enable_grad()
def linear_transform(x, a, b):
    return a * x + b

def linear_transform2(x, a, b):
    return a * x + b

is_train = False
with torch.set_grad_enabled(is_train):
    y = linear_transform(x, 2, 3)
    z = linear_transform2(x, 2, 3)

y.requires_grad, z.requires_grad


# ## In-place operations on Tensors
# - `Variable` is deprecated

# ```python
# class Tensor(torch._C._TensorBase):
#     ...
#     def backward(self, gradient=None, retain_graph=None, create_graph=False):
#         torch.autograd.backward(self, gradient, retain_graph, create_graph)
#     ...
# ```
