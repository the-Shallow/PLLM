import torch

## Recursively find layers that match the layer name specified and return a dictionary filled with the modules that match
def find_layers(module, layers, name=''):
    pass

## loop through the layers in a model and return the global sparsity % of the model
def check_sparsity(model):
    pass

## loop through a singular layer and return it's local sparsity %
def check_layer_sparsity(module):
    pass

def return_given_alpha(alpha, sort_res, W_norm, temp_norm, pre_sum):
    threshold_cumsum = alpha * pre_sum
    sort_mask = temp_norm <= threshold_cumsum.reshape((-1,1))
    threshold = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1,keepdims=True)-1)
    W_mask = (W_norm <= threshold)
    curr_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, curr_sparsity

class WrappedGPT:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples