import torch
import psutil
import importlib

DEVICE_OVERRIDE = None

def has_dml():
    loader = importlib.find_loader('torch_directml')
    if loader is None:
        return False
    
    import torch_directml
    return torch_directml.is_available()

def set_device_name(name):
    global DEVICE_OVERRIDE
    DEVICE_OVERRIDE = name

def get_device_name():
    global DEVICE_OVERRIDE
    if DEVICE_OVERRIDE is not None and DEVICE_OVERRIDE != "":
        return DEVICE_OVERRIDE

    name = 'cpu'

    if torch.cuda.is_available():
        name = 'cuda'
    elif has_dml():
        name = 'dml'

    return name

def get_device(verbose=False):
    name = get_device_name()

    if verbose:
        if name == 'cpu':
            print("No hardware acceleration is available, falling back to CPU...")    
        else:
            print(f"Hardware acceleration found: {name}")

    if name == "dml":
        import torch_directml
        return torch_directml.device()

    return torch.device(name)

def get_device_batch_size():
    available = 1
    name = get_device_name()

    if name == "dml":
        # there's nothing publically accessible in the DML API that exposes this
        # there's a method to get currently used RAM statistics... as tiles
        available = 1
    elif name == "cuda":
        _, available = torch.cuda.mem_get_info()
    elif name == "cpu":
        available = psutil.virtual_memory()[4]

    availableGb = available / (1024 ** 3)
    if availableGb > 14:
        return 16
    elif availableGb > 10:
        return 8
    elif availableGb > 7:
        return 4
    return 1

def get_device_count(name=get_device_name()):
    if name == "cuda":
        return torch.cuda.device_count()
    if name == "dml":
        import torch_directml
        return torch_directml.device_count()

    return 1


if has_dml():
    _cumsum = torch.cumsum
    _repeat_interleave = torch.repeat_interleave
    _multinomial = torch.multinomial
    
    _Tensor_new = torch.Tensor.new
    _Tensor_cumsum = torch.Tensor.cumsum
    _Tensor_repeat_interleave = torch.Tensor.repeat_interleave
    _Tensor_multinomial = torch.Tensor.multinomial

    torch.cumsum = lambda input, *args, **kwargs: ( _cumsum(input.to("cpu"), *args, **kwargs).to(input.device) )
    torch.repeat_interleave = lambda input, *args, **kwargs: ( _repeat_interleave(input.to("cpu"), *args, **kwargs).to(input.device) )
    torch.multinomial = lambda input, *args, **kwargs: ( _multinomial(input.to("cpu"), *args, **kwargs).to(input.device) )
    
    torch.Tensor.new = lambda self, *args, **kwargs: ( _Tensor_new(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.cumsum = lambda self, *args, **kwargs: ( _Tensor_cumsum(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.repeat_interleave = lambda self, *args, **kwargs: ( _Tensor_repeat_interleave(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.multinomial = lambda self, *args, **kwargs: ( _Tensor_multinomial(self.to("cpu"), *args, **kwargs).to(self.device) )