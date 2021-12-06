import os
import torch
import pickle


def save_model_by_name(model, global_step, run=0):
    save_dir = os.path.join('checkpoints', model.name, str(run))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def load_model_by_name(model, global_step, run, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints',
                             model.name,
                             str(run),
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


def save_svd_model(model, run, mod_num):
    save_dir = os.path.join('checkpoints', 'truncated_svd', str(run))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f'{mod_num}.pk')
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def load_svd_model(run, mod_num):
    save_dir = os.path.join('checkpoints', 'truncated_svd', str(run))
    file_path = os.path.join(save_dir, f'{mod_num}.pk')
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model
