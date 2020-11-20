import logging
import sys
import pickle
import torch

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def save_object(obj, object_type, timestamp, config):
    filename = config.model.name + '_' + object_type + '_' + timestamp + '.pkl'
    path = config.model.saved_model_path + filename
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_object(filename):
    # path = f'../saved_models/{filename}'
    # Using absolute path here atm
    path = filename
    with open(path, 'rb') as file:
        loaded_obj = pickle.load(file)
        log.info('Loaded object from local path...')
    return loaded_obj


def save(model, optimizer, epoch, metrics, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, config.model.save_model_path + config.experiment.name + "-epoch-" + str(epoch) + ".pth")


def load(model, optimizer, config=None):
    checkpoint = torch.load(config.model.save_model_path + config.trainer.load_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])