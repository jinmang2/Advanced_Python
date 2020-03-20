from collections import OrderedDict, namedtuple
import functools, itertools, weakref, warnings

import torch
from torch.nn.parameter import Parameter
import torch.utils.hooks as hooks

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_key'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()
    
    __str__ = __repr__
    
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    
    """
    Base class for all neural network modules.
    
    Module, Buffer, Parameter, Hook >> 이 친구들이 누군지 명확하게 알아야
    이 클래스를 정복, 흐름을 장악할 수 있다!
    
    아직 100% 탐구를 못한 메서드들도 많다...
    C++ Engine에서 돌아가는 코드는 지금 못보지만
    그게 아닌 메서드들은 어떤 원리로 동작하는지 정확하게 파악하자!
    
    Usage::
        
        import torch.nn as nn
        import torch.nn.functional as F
        
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
                
    """
    
    dump_patches = False
    _version = 1
    
    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        print('INIT 생성자 동작합니다.')
        torch._C._log_api_usage_once("python.nn_module")  # log 찍는 것?
        
        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        
    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        print('FORWARD 동작합니다.')
        raise NotImplementedError
        
    def register_buffer(self, name, tensor):
        """
        Adds a persistent buffer to the module.
        
        Example::
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        print('REGISTER_BUFFER 동작합니다.')
        # 예외 처리
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise keyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attributes '{}' already exists".format(name))
            
        # tensor 예외 처리
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(tensor), name))
        # 할당
        else:
            self._buffers[name] = tensor
            
    def register_parameter(self, name, param):
        """
        Adds a parameter to the module.
        """
        print('REGISTER_PARAMETER 동작합니다.')
        # 예외 처리
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise keyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attributes '{}' already exists".format(name))
        
        # param 예외 처리
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif params.grad_fn: # 이건 왜 있을까?
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parametesr[name] = params
            
    def add_module(self, name, module):
        """
        Adds a child module to the current module.
        """
        print('ADD_MODULE 동작합니다.')
        # 예외 처리
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(
                torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        # 할당
        self._modules[name] = module
        
    def _apply(self, fn):
        print('_APPLY 동작합니다.')
        for module in self.children():
            module._apply(fn)
        
        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                """
                Defined in File Functions.h
                
                If the new tensor has compatible tensor type as the existing tensor,
                the current behavior is to change the tensor in-place using `.data =`,
                and the future behavior is to overwrite the existing tensor. However,
                changing the current behavior is a BC-breaking change, and we want it
                to happen in future releases. So for now we introduce the
                `torch.__future__.get_overwrite_module_params_on_conversion()`
                global flag to let the user control whether they want the future
                behavior of overwriting the existing tensor or not.
                """
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False
            
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    self._parameters[key] = Parameter(param_applied, param.requres_grad)
                    
                if paaram.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                    if should_ues_set_data:
                        param.grad = grad_applied
                    else:
                        assert param.grad.is_leaf
                        self._parameters[key].grad = grad_applied.requires_grad_(param.grad.requires_grad)
        
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
                
        return self
    
    def apply(self, fn):
        print('APPLY 동작합니다.')
        """
        Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        
        Example::
            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
        
    def cuda(self, device=None):
        """
        Moves all model parameters and buffers to the GPU.
        """
        return self._apply(lambda t: t.cuda(device))
    
    def cpu(self):
        """
        Moves all model parameters and buffers to the CPU.
        """
        return self._apply(lambda t: t.cpu())
    
    def type(self, dst_type):
        """
        Casts all parameters and buffers to :attr:`dst_type`.
        """
        return self._apply(lambda t: t.type(dst_type))
    
    def float(self):
        """
        Casts all floating point paramters and buffers to ``float`` datatype.
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)
    
    def double(self):
        """
        Casts all floating point parameters and buffers to ``double`` datatype.
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
    
    def half(self):
        """
        Casts all floating point parameters and buffers to ``half`` datatype.
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
    
    def bfloat16(self):
        """
        Casts all floating point parameters and buffers to ``bfloat16`` datatype.
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)
    
    def to(self, *args, **kwargs):
        print('TO 동작합니다.')
        """
        Moves and/or casts the parameters and buffers.
        
        This can be called as
        .. function:: to(device=None, dtype=None, non_blocking=False)
        .. function:: to(dtype, non_blocking=False)
        .. function:: to(tensor, non_blocking=False)
        .. function:: to(memory_format=torch.channels_last)
        
        Example::
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)
        """
        
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}',format(dtype))
                
        def convert(t):
            if convert_to_format is not None and t.dim() == 4:
                return t.to(device, dtype if t.is_floating_point() else None, non_blocking, 
                            memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking)
        
        return self._apply(convert)
    
    """
    Hook 추가 함수
        - self.register_backward_hook
        - self.register_forward_pre_hook
        - self.register_forward_hook
        
    What is Hook?
        Hook: 계층의 출력이나 grad_output 을 살펴보거나 수정
    """
    def register_backward_hook(self, hook):
        print('REGISTER_BACKWARD_HOOK 동작합니다.')
        """
        Registers a backward hook on the module.
        """
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle
    
    def register_forward_pre_hook(self, hook):
        print('REGISTER_FORWARD_PRE_HOOK 동작합니다.')
        """
        Registers a forward pre-hook on the module.
        """
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle
    
    def register_forward_hook(self, hook):
        print('REGISTER_FORWARD_HOOK 동작합니다.')
        """
        Registers a forward hook on the module.
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle
    
    """이건 아직도 모르겠어! 나중에 살펴보자."""
    def _slow_forward(self, *input, **kwargs):
        print('_SLOW_FORWARD 동작합니다.')
        tracing_state = torch._C._get_tracing_state()
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        recording_scopes = torch.jit._trace_module_map is not None
        if recording_scopes:
            name = torch.jit._trace_module_map[self] if self in torch.jit._trace_module_map else None
            if name:
                cur_scope_name = tracing_state.current_scope()
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = self.forward(*input, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result
    
    def __call__(self, *input, **kwargs):
        print('__CALL__ 동작합니다.')
        """
        < Callable Object >
        Instance가 호출됐을 때 실행
        
        `x()`와 `x.__call__()`이 동일!
        """
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
    
    def __setstate__(self, state):
        print('__SETSTATE__ 동작합니다.')
        """
        < Object Pickling >
        what is pickling?
            파이썬 데이터 구조의 직렬화 프로세스
            객체를 저장하고 나중에 검색(캐싱)할 때 매우 유용
            걱정과 혼란의 주 요인
            
        `__setstate__(self, state)`: 객체가 unpickle되었을 때
            객체의 상태는 객체의 `__dict__`에 직접 적용되지 않고 전달.
        """
        self.__dict__.update(state)
        # Support loading old checkpints that don't have the following attrs:
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
            
    def __getattr__(self, name):
        print('__GETATTR__동작합니다.')
        """
        < 속성 접근 제어하기 >
        파이썬은 클래스에 대한 진정한 캡슐화가 부족한가?
        getter, setter를 사용하여 개인 속성을 정의할 수 있는 방법이 없는가?
        Nope!
        "매직"을 통해 많은 양의 캡슐화를 그냥 수행
        
        `__getattr__(self, name)`: 사용자가 존재하지 않는 속성에 
            엑세스하려고 시도할 때의 행위를 정의.
            일반적인 맞춤법 오류를 포착, 리다이렉트,
            더 이상 사용되지 않는 속성 
            (원하는 경우 해당 속성을 계산하고 반환하도록 선택 가능)
            사용에 대한 경고를 제공하거나,
            `AttributeError`를 손쉽게 전달할 때 유용.
            존재하지 않는 속성에 엑세스할 때만 호출되므로 실제 캡슐화 솔루션 X
        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
            
    def __setattr__(self, name, value):
        print('__SETATTR__ 동작합니다.')
        """
        < 속성 접근 제어하기 >
        파이썬은 클래스에 대한 진정한 캡슐화가 부족한가?
        getter, setter를 사용하여 개인 속성을 정의할 수 있는 방법이 없는가?
        Nope!
        "매직"을 통해 많은 양의 캡슐화를 그냥 수행
        
        `__setattr__(self, name, value)`: 캡슐화 솔루션
            특성값의 변경 사항에 대한 사용자 지정 규칙을 정의
            해당 특성의 존재 여부에 관계없이 특성에 할당할 동작 정의
            
        Caution!!
        ```python
        def __setattr__(self, name, value):
            self.name = value
            # 속성이 할당될 때마다 __setattr__()이 호출. (재귀)
            # 이는 self.__setattr__(name, value)를 의미
            # 무한 재귀 발생, 이를 방지해줘야 함
            
        def __setattr__(self, name, value):
            self.__dict__[name] = value # 클래스의 dict의 이름에 할당
            # 커스톰 동작을 정의
        ```
        """
        def remove_from(*dict):
            for d in dicts:
                if name in d:
                    del d[name]
        params = self.__dict__.get('_paramters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            print(self.__dict__)
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise ValueError("cannot assign '{}' as child module '{}' "
                                     "(torch.nn.Module or None expected)"
                                     .format(torch.typename(value), name))
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
                    
    def __delattr__(self, name):
        print('__DELATTR__ 동작합니다.')
        """
        < 속성 접근 제어하기 >
        파이썬은 클래스에 대한 진정한 캡슐화가 부족한가?
        getter, setter를 사용하여 개인 속성을 정의할 수 있는 방법이 없는가?
        Nope!
        "매직"을 통해 많은 양의 캡슐화를 그냥 수행
        
        `__delattr__(self, name)`: `__setattr__`과 완전히 동일
            그러나 속성을 설정하는 대신 삭제하는 것.
            무한 재귀(`__delattr__` 구현시 `del self.name`을 호출하면
            무한 재귀가 발생)를 방지하기 위해 `__setattr__`과 동일한 예방 조치를
            취해야한다.
        """
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name) # self말고 object로 호출
            
    """
    state_dict_hook 관련 함수들
    
    뭘 위해서 존재하는지를 공부하자!
    """
    def _register_state_dict_hook(self, hook):
        print('_REGISTER_STATE_DICT_HOOK 동작합니다.')
        handle = hooks.RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        print('_SAVE_TO_STATE_DICT 동작합니다.')
        """
        Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                desetination[prefix + name] = buf if keep_vars else buf.detach()
                
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        print('STATE_DICT 동작합니다.')
        """
        Returns a dictionary containing a whole state of the module.
        
        Example::
            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
    
    def _register_load_state_dict_pre_hook(self, hook):
        print('_REGISTER_LOAD_STATE_DICT_PRE_HOOK 동작합니다.')
        handle = hooks.RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = hook
        return handle
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        print('_LOAD_FROM_STATE_DICT 동작합니다.')
        """
        Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants.
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict, strict=True):
        print('LOAD_STATE_DICT 동작합니다.')
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants.
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys) # 여기서 나오는구나!
    
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        print('_NAMED_MEMBERS 동작합니다.')
        """
        Helper method for yielding various names + members of modules.
        """
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
                
    def parameters(self, recurse=True):
        print('PARAMETERS 동작합니다.')
        """
        Returns an iterator over module parameters.
        
        Example::
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param
            
    def named_parameters(self, prefix='', recurse=True):
        print('NAMED_PARAMETERS 동작합니다.')
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        
        Example::
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
            
    def buffers(self, recurse=True):
        print('BUFFERS 동작합니다.')
        """
        Returns an iterator over module buffers.
        
        Example::
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
        """
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf
            
    def named_buffers(self, prefix='', recurse=True):
        print('NAMED_BUFFERS 동작합니다.')
        """
        Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.
        
        Example::
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())
        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self):
        print('CHILDREN 동작합니다.')
        """
        Returns an iterator over immediate children modules.
        
        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module
            
    def named_children(self):
        print('NAMED_CHILDREN 동작합니다.')
        """
        Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        
        Yields:
            (string, Module): Tuple containing a name and child module
            
        Example:
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
                
    def modules(self):
        print('MODULES 동작합니다.')
        """
        Returns an iterator over all members in the network.
        
        Yields:
            Module: a module in the network
            
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
            
        Example::
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(1, 1)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            
            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)
        """
        for name, module in self.named_modules():
            yield module
            
    def named_modules(self, memo=None, prefix=''):
        print('NAMED_MODULES 동작합니다.')
        """
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        
        Yield:
            (string, Module): Tuple of name and module
            
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
            
        Example::
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)
            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
                    
    def train(self, mode=True):
        print('TRAIN 동작합니다.')
        """
        Sets the module in training mode.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self, mode=True):
        print('EVAL 동작합니다.')
        """
        Sets the module in evaluation mode.
        """
        return self.train(False)
    
    def requires_grad_(self, requires_grad=True):
        print('REQUIRES_GRAD_ 동작합니다.')
        """
        Change if autograd should record operation on parameters in this
        module.
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self
    
    def zero_grad(self):
        print('ZERO_GRAD 동작합니다.')
        """
        Sets gradients of all model parameters to zero.
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                
    def share_memory(self):
        print('SHARE_MEMORY 동작합니다.')
        return self._apply(lambda t: t.share_memory_())
    
    def _get_name(self):
        print('_GET_NAME 동작합니다.')
        return self.__class__.__name__
    
    def extra_repr(self):
        print('EXTRA_REPR 동작합니다.')
        """
        Set the extra representation of the module
        """
        return ''
    
    def __repr__(self):
        print('__REPR__ 동작합니다.')
        """
        < 클래스 표현하기 >
        클래스를 문자열로 표현!
        
        `__repr__(self)`: 클래스의 인스턴스에서 `repr()`이 호출될 때의
            동작을 정의. `repr()`은 주로 기계가 읽을 수 있는 출력을 대상으로,
            `str()`은 사람이 읽을 수 있도록 만들어짐.
        """
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
    
    def __dir__(self):
        print('__DIR__ 동작합니다.')
        """
        < 클래스 표현하기 >
        클래스를 문자열로 표현!
        
        `__dir__(self)`: 클래스의 인스턴스에서 `dir()`이 호출될 때의
            동작을 정의. 이 메서드는 사용자의 attribute 목록을 반환.
            일반적으로 `__dir__`을 구현하는 것은 불필요, `__getattr__` 또는
            `__getattribute__`를 재정의하거나 그렇지 않으면 동적으로
            속성을 생성하는 경우 클래스를 대화식으로 사용하는 것이
            매우 중요할 수 있음.
        """
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)
    
    def _replicate_for_data_parallel(self):
        print('_REPLICATE_FOR_DATA_PARALLEL 동작합니다.')
        replica = self.__new__(type(self))
        """
        < 생성 및 초기화 >
        가장 기본적인 매직 메서드인 `__init__`는 모두 알고 있음.
        그러나 `x = SomeClass()`를 호출하면 `__init__`이 먼저 호출되지 않음.
        사실 `__new__` 메서드가 먼저 실행되고 실제로 인스턴스를 생성한 다음
        생성시에 인수를 초기화 프로그램에 전달.

        `__new__(cls, [...)`: 객체의 인스턴스화에서 호출되는 첫 번째 메서드.
            클래스를 취한 다음 `__init__`에 전달할 다른 인수를 취함.
            이를 정의하는 일은 드물지만 튜플이나 문자열과 같은 불변 유형을 
            서브 클래싱하는 경우에는 그 용도가 있음.
            자세한 내용은 아래 링크를 참고
            https://www.python.org/download/releases/2.2/descrintro/#__new__
        """
        replica.__dict__ = self.__dict__.copy()
        replica._parameters = replica._parameters.copy()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()

        # Warn users that gradients don't behave as expected on replica modules
        old_zero_grad = replica.__class__.zero_grad
        weak_self = weakref.ref(replica)

        def zero_grad():
            print('ZERO_GRAD 동작합니다.')
            warnings.warn(
                "Calling .zero_grad() from a module that was passed to a nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")
            replica = weak_self()
            if replica:
                old_zero_grad(replica)

        replica.zero_grad = zero_grad

        return replica