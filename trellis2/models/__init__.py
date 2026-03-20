import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, device: str = 'cpu', **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        device: Target device string (e.g. 'cpu', 'cuda'). When 'cuda', weights are loaded
                directly into GPU memory via safetensors, bypassing the CPU→GPU copy.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import torch
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)

    if device != 'cpu' and torch.cuda.is_available():
        # Skip random weight init during construction — weights come from checkpoint.
        # This avoids expensive kaiming_uniform_ etc. on ~1B+ parameters.
        # Computed attributes (RoPE freqs, position embeddings) still work correctly
        # since they use torch.arange/meshgrid, not nn.init functions.
        import torch.nn.init as _init
        _orig_inits = {}
        _noop = lambda tensor, *a, **kw: tensor
        for _fn in ['uniform_', 'normal_', 'kaiming_uniform_', 'kaiming_normal_',
                     'xavier_uniform_', 'xavier_normal_', 'zeros_', 'ones_',
                     'constant_', 'orthogonal_', 'trunc_normal_']:
            if hasattr(_init, _fn):
                _orig_inits[_fn] = getattr(_init, _fn)
                setattr(_init, _fn, _noop)

        try:
            model = __getattr__(config['name'])(**config['args'], **kwargs)
        finally:
            for _fn, _orig in _orig_inits.items():
                setattr(_init, _fn, _orig)

        # Capture the mixed-precision dtype layout from construction.
        # Models like SLatFlowModel.convert_to(bfloat16) only convert self.blocks,
        # leaving input_layer/out_layer in float32. Similarly FlexiDualGridVaeEncoder
        # with convert_to_fp16() keeps input_layer float32, blocks fp16.
        # Without set_default_dtype interference, these dtypes are authoritative.
        _model_dtypes = {n: p.dtype for n, p in model.named_parameters()}

        # Load weights directly from safetensors to GPU.
        # assign=True replaces parameter tensors for fast loading without
        # double-allocation, but some safetensors files store ALL weights
        # at a single dtype (e.g. fp16 VAE), losing the mixed-precision layout.
        state_dict = load_file(model_file, device=device)
        model.load_state_dict(state_dict, strict=False, assign=True)

        # Restore mixed-precision dtype layout where safetensors disagrees
        # with the model's construction-time dtypes.
        for _name, _param in model.named_parameters():
            _expected = _model_dtypes.get(_name)
            if _expected is not None and _param.dtype != _expected:
                _param.data = _param.data.to(_expected)

        # Move any remaining CPU tensors (buffers/attrs not in safetensors) to GPU.
        model.to(device)
    else:
        model = __getattr__(config['name'])(**config['args'], **kwargs)
        state_dict = load_file(model_file)
        model.load_state_dict(state_dict, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
