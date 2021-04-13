"""
@author: robotcator
@date: 20210404
"""
import torch
from .sym_quant import FakeLinearQuantization


BN = (torch.nn.BatchNorm1d, 
      torch.nn.BatchNorm2d, 
      torch.nn.BatchNorm3d)


def quant_activation(module, input, output):
    if hasattr(module, "activation_quant"):
        output_type = output.dtype
        output_qunt = module.activation_quant
        output.data = output_qunt(output)
        output = output.to(output_type)


def quant_weight(module, input):
    module.weight_origin.data.copy_(module.weight.data)
    if hasattr(module, "weight_quant"):
        weight_quant_module = module.weight_quant
        module.weight.data = weight_quant_module(module.weight)


def register_quantization_hook(model,
                               bits=8,
                               mode='SYMMETRIC',
                               Quant_weight=True,
                               Quant_output=True,
                               Quant_data=False,
                               filter_name=None):
 
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            register_quantization_hook(module, bits, mode, quant_weight, quant_activation)
        else:
            if filter_name is not None and name in filter_name: continue 
            if Quant_weight and hasattr(
                module, "weight") and module.weight is not None:
                # and not isinstance(module, BN):
                module.register_buffer('weight_origin', module.weight.detach().clone())
                module.add_module("weight_quant", 
                    FakeLinearQuantization(bits, mode, ema_decay=0.0))
                module.register_forward_pre_hook(quant_weight)

            if Quant_output:
                module.add_module("activation_quant", 
                    FakeLinearQuantization(bits, mode, ema_decay=0.99))
                module.register_forward_hook(quant_activation)
    return model