import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import datetime
import _pickle as cPickle
import sed_eval

from utilities import get_filename, inverse_scale
from pytorch_utils import forward
import config


class Evaluator(object):
    def __init__(self, model, data_generator, subtask, cuda=True):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          data_type: 'train' | 'validate'
          sources: list of devices
          statistics_path: string | None
          cuda: bool
          verbose: bool
        '''
        
        self.model = model
        self.data_generator = data_generator
        self.subtask = subtask
        self.cuda = cuda
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        self.in_domain_classes_num = len(config.labels) - 1
        self.all_classes_num = len(config.labels)
        self.idx_to_lb = config.idx_to_lb
        self.lb_to_idx = config.lb_to_idx

    def evaluate(self, data_type, source, max_iteration=None, verbose=False):
        '''Evaluate the performance. 
        
        Args: 
          max_iteration: None | int, maximum iteration to run to speed up evaluation
        '''

        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            source=source, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
            
        output = output_dict['output']
        target = output_dict['target']
        
        # Obtain probability
        if self.subtask in ['a', 'b']:
            prob = np.exp(output)
        elif self.subtask == 'c':
            prob = output
        
        y_true = np.argmax(target, axis=-1)
        y_pred = np.argmax(prob, axis=-1)
        
        if self.subtask == 'c':
            for n, class_id in enumerate(y_pred):
                if prob[n, class_id] < 0.5:
                    y_pred[n] = self.lb_to_idx['unknown']
            
        if self.subtask in ['a', 'b']:
            confusion_matrix = metrics.confusion_matrix(
                y_true, y_pred, labels=np.arange(self.in_domain_classes_num))
        elif self.subtask == 'c':
            confusion_matrix = metrics.confusion_matrix(
                y_true, y_pred, labels=np.arange(self.all_classes_num))
        
        # Evaluate
        classwise_accuracy = np.diag(confusion_matrix) \
            / np.sum(confusion_matrix, axis=-1)
        
        logging.info('Data type: {}'.format(data_type))
        logging.info('    Source: {}'.format(source))
        
        if self.subtask in ['a', 'b']:
            logging.info('    Accuracy: {:.3f}'.format(np.mean(classwise_accuracy)))
        elif self.subtask == 'c':
            logging.info('    In domain ccuracy: {:.3f}, Unknown accuracy: {:.3f}'
                ''.format(np.mean(classwise_accuracy[0 : -1]), classwise_accuracy[-1]))

        if verbose:
            classes_num = len(classwise_accuracy)
            for n in range(classes_num):
                logging.info('{:<20}{:.3f}'.format(self.labels[n], 
                    classwise_accuracy[n]))
                    
            logging.info(confusion_matrix)

        statistics = {
            'accuracy': classwise_accuracy, 
            'confusion_matrix': confusion_matrix, 
            'prob': prob}

        return statistics
            
    def visualize(self, data_type, source, max_iteration=None):

        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        frames_num = config.frames_num
        labels = config.labels
        idx_to_lb = config.idx_to_lb
        
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            source=source, 
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


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'a': [], 'b': [], 'c': []}

    def append_and_dump(self, iteration, source, statistics):
        statistics['iteration'] = iteration
        self.statistics_dict[source].append(statistics)

        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
