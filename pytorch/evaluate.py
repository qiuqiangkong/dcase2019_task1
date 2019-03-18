import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import sed_eval

from utilities import get_filename, inverse_scale
from pytorch_utils import forward
import config


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          cuda: bool
          verbose: bool
        '''
        
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        self.idx_to_lb = config.idx_to_lb

    def evaluate(self, data_type, sources, submission_path, max_iteration=None, verbose=False):
        '''Evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          sources: list of devices
          submission_path: string
          max_iteration: None | int, maximum iteration to run to speed up evaluation
        '''

        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            sources=sources, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
            
        logprob = output_dict['output']
        target = output_dict['target']
        
        # Evaluate
        confusion_matrix = metrics.confusion_matrix(
            y_true=np.argmax(target, axis=-1), 
            y_pred=np.argmax(logprob, axis=-1), 
            labels=None)
        
        classwise_accuracy = np.diag(confusion_matrix) \
            / np.sum(confusion_matrix, axis=-1)
        
        logging.info('Devices: {}. Data type: {}. Accuracy:'.format(
            sources, data_type))
        
        if verbose:
            classes_num = len(classwise_accuracy)
            for n in range(classes_num):
                logging.info('{:<20}{:.3f}'.format(self.labels[n], 
                    classwise_accuracy[n]))
        
        logging.info('{:<20}{:.3f}'.format('Avg.', np.mean(classwise_accuracy)))
            
    def visualize(self, data_type, sources, max_iteration=None):

        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        frames_num = config.frames_num
        labels = config.labels
        idx_to_lb = config.idx_to_lb
        
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            sources=sources, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_input=True, 
            return_target=True)

        rows_num = 3
        cols_num = 4
        classes_num = output_dict['target'].shape[1]
        
        fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))

        for k in range(classes_num):
            for n, audio_name in enumerate(output_dict['audio_name']):
                if output_dict['target'][n, k] == 1:
                    title = idx_to_lb[k]
                    axs[k // cols_num, k % cols_num].set_title(title, color='r')
                    logmel = inverse_scale(output_dict['feature'][n], self.data_generator.scalar['mean'], self.data_generator.scalar['std'])
                    axs[k // cols_num, k % cols_num].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')                
                    axs[k // cols_num, k % cols_num].set_xticks([0, frames_num])
                    axs[k // cols_num, k % cols_num].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                    axs[k // cols_num, k % cols_num].xaxis.set_ticks_position('bottom')
                    axs[k // cols_num, k % cols_num].set_ylabel('Mel bins')
                    axs[k // cols_num, k % cols_num].set_yticks([])
                    break
        
        for k in range(classes_num, rows_num * cols_num):
            row = k // cols_num
            col = k % cols_num
            axs[row, col].set_visible(False)
            
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()