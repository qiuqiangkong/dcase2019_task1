import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, load_scalar, 
    get_subdir, get_sources, get_classes_num)
from data_generator import DataGenerator
from models import Cnn_9layers
from losses import nll_loss
from evaluate import Evaluator
from pytorch_utils import move_data_to_gpu
import config


def train(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      subtask: 'a' | 'b' | 'c', corresponding to 3 subtasks in DCASE2019 Task1
      data_type: 'development'
      workspace: string, directory of workspace
      model_type: string, e.g. 'Cnn_9layers'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subtask = args.subtask
    data_type = args.data_type
    workspace = args.workspace
    model_type = args.model_type
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    max_iteration = 10      # Number of mini-batches to evaluate on training data
    reduce_lr = True
    
    sources = get_sources(subtask)
    classes_num = get_classes_num(subtask)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    sub_dir = get_subdir(subtask, data_type)
    
    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 
        'fold1_train.csv')
        
    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 
        'fold1_evaluate.csv')
                
    feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(sub_dir))
    create_folder(checkpoints_dir)
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(sub_dir))
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    if cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path, 
        train_csv=train_csv, 
        validate_csv=validate_csv, 
        classes_num=classes_num, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)
    
    train_bgn_time = time.time()
    iteration = 0
    
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        if iteration % 200 == 0:

            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            evaluator.evaluate(
                data_type='train', 
                submission_path=None, 
                sources=['a', 'b', 'c'], 
                max_iteration=None, 
                verbose=False)
            
            evaluator.evaluate(
                data_type='validate', 
                sources=['a', 'b', 'c'], 
                submission_path=None, 
                max_iteration=None, 
                verbose=False)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'target']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)
        
        # Train
        model.train()
        batch_output = model(batch_data_dict['feature'])
        
        # loss
        loss = nll_loss(batch_output, batch_data_dict['target'])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 5000:
            break
            
        iteration += 1
        

def inference_validation(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      subtask: 'a' | 'b' | 'c', corresponding to 3 subtasks in DCASE2019 Task1
      data_type: 'development'
      workspace: string, directory of workspace
      model_type: string, e.g. 'Cnn_9layers'
      iteration: int
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subtask = args.subtask
    data_type = args.data_type
    workspace = args.workspace
    model_type = args.model_type
    iteration = args.iteration
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    sources = get_sources(subtask)
    classes_num = get_classes_num(subtask)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    sub_dir = get_subdir(subtask, data_type)
    
    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 
        'fold1_train.csv')
        
    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 
        'fold1_evaluate.csv')
                
    feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    model_path = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(sub_dir), '{}_iterations.pth'.format(iteration))
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(sub_dir))
    create_logging(logs_dir, 'w')
    logging.info(args)
    
    submission_path = os.path.join(workspace, 'submissions', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(sub_dir), 'md_{}_iters'.format(iteration), 'submission.csv')
        
    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path, 
        train_csv=train_csv, 
        validate_csv=validate_csv, 
        classes_num=classes_num, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)
    
    if subtask in ['a', 'c']:
        evaluator.evaluate(data_type='validate', 
            submission_path=submission_path, sources=['a'], verbose=True)
    elif subtask == 'b':
        evaluator.evaluate(data_type='validate', 
            submission_path=submission_path, sources=['a'], verbose=True)
        evaluator.evaluate(data_type='validate', 
            submission_path=submission_path, sources=['b'], verbose=True)
        evaluator.evaluate(data_type='validate', 
            submission_path=submission_path, sources=['c'], verbose=True)
        evaluator.evaluate(data_type='validate', 
            submission_path=submission_path, sources=['a', 'b', 'c'], verbose=True)
    
    if visualize:
        evaluator.visualize(data_type='validate', sources=['a'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True)
    parser_train.add_argument('--data_type', type=str, choices=['development', 'evaluation'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True)
    parser_inference_validation.add_argument('--data_type', type=str, choices=['development'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True)
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False)
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)

    else:
        raise Exception('Error argument!')