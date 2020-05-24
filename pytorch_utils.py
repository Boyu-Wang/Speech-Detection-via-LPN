import torch
import os

def save_checkpoint(model, optimizer, file_path, filename='checkpoint.pth',
                    is_best=None, bestname='best_model.pth', only_best=True):
    state = dict()
    state['state_dict'] = model.state_dict()
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if not only_best:
        torch.save(state, os.path.join(file_path, filename))
    if is_best:
        torch.save(state, os.path.join(file_path, bestname))

def load_checkpoint(model, file_path, filename='best_model.pth',model_entry='state_dict', optimizer=None,
                    optimizer_entry='optimizer'):
    model_weights = torch.load(os.path.join(file_path, filename))
    model.load_state_dict(model_weights[model_entry])
    if optimizer is not None:
        optimizer.load_state_dict(model_weights[optimizer_entry])
