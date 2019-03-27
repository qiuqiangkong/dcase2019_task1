import argparse
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

from utilities import get_subdir
import config


def plot_results(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    subtask = args.subtask
    select = args.select
    
    filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 5000
    data_type = 'development'
    
    iterations = np.arange(0, max_plot_iteration, 200)
    
    def _load_stat(model_type, subtask, source):
        sub_dir = get_subdir(subtask, data_type)

        statistics_path = os.path.join(workspace, 'statistics', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        model_type, '{}'.format(sub_dir), 'validate_statistics.pickle')
        
        statistics_dict = cPickle.load(open(statistics_path, 'rb'))
        accuracy_matrix = np.array([stat['accuracy'] for stat in statistics_dict[source]])
        confusion_matrix = np.array([stat['confusion_matrix'] for stat in statistics_dict[source]])
        legend = '{}'.format(model_type)
        
        if subtask in ['a', 'b']:
            accuracy = np.mean(accuracy_matrix, axis=-1)
            results = {'accuracy': accuracy, 'legend': legend}
            print('Subtask: {}, Source: {}, Model: {} accuracy: {:.3f}'.format(subtask, source, model_type, accuracy[-1]))
            
        elif subtask == 'c':
            accuracy = np.mean(accuracy_matrix[:, 0 : -1], axis=-1)
            unknown_accuracy = accuracy_matrix[:, -1]
            results = {
                'accuracy': accuracy, 
                'unknown_accuracy': unknown_accuracy, 
                'legend': legend}
            print('Subtask: {}, Source: {}, Model: {}, accuracy: {:.3f}, Unknown accuracy: {:.3f}'.format(subtask, source, model_type, accuracy[-1], unknown_accuracy[-1]))
        
        return results
        
    if subtask == 'a':
        source = 'a'
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        lines = []
        
        results = _load_stat('Cnn_9layers_AvgPooling', subtask, source=source)
        line, = ax.plot(results['accuracy'], label=results['legend'])
        lines.append(line)
        
        ax.set_title('Device: {}'.format(source))
        ax.legend(handles=lines, loc=4)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        ax.grid(color='b', linestyle='solid', linewidth=0.2)
        ax.xaxis.set_ticks(np.arange(0, len(iterations) + 1, len(iterations) // 4))
        ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration + 1, max_plot_iteration // 4))
            
        plt.tight_layout()
        fig_path = '_tmp.png'
        plt.savefig(fig_path)
        print('Save fig to {}'.format(fig_path))
        
    elif subtask == 'b':
        sources = ['a', 'b', 'c']
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        for n in range(3):
            lines = []
            
            results = _load_stat('Cnn_9layers_AvgPooling', subtask, source=sources[n])
            line, = axs[n // 2, n % 2].plot(results['accuracy'], label=results['legend'])
            lines.append(line)
            
            axs[n // 2, n % 2].set_title('Device: {}'.format(sources[n]))
            axs[n // 2, n % 2].legend(handles=lines, loc=4)
            axs[n // 2, n % 2].set_ylim(0, 1.0)
            axs[n // 2, n % 2].set_xlabel('Iterations')
            axs[n // 2, n % 2].set_ylabel('Accuracy')
            axs[n // 2, n % 2].grid(color='b', linestyle='solid', linewidth=0.2)
            axs[n // 2, n % 2].xaxis.set_ticks(np.arange(0, len(iterations) + 1, len(iterations) // 4))
            axs[n // 2, n % 2].xaxis.set_ticklabels(np.arange(0, max_plot_iteration + 1, max_plot_iteration // 4))
            
        plt.tight_layout()
        fig_path = '_tmp.png'
        plt.savefig(fig_path)
        print('Save fig to {}'.format(fig_path))
        
    elif subtask == 'c':
        source = 'a'
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].set_title('Device: {}, known'.format(source))
        axs[1].set_title('Device: {}, unknown'.format(source))
            
        lines = []
        
        results = _load_stat('Cnn_9layers_AvgPooling', subtask, source=source)
        line, = axs[0].plot(results['accuracy'], label=results['legend'])
        line, = axs[1].plot(results['unknown_accuracy'], label=results['legend'])
        lines.append(line)
        
        for n in range(2):
            axs[n].legend(handles=lines, loc=4)
            axs[n].set_ylim(0, 1.0)
            axs[n].set_xlabel('Iterations')
            axs[n].set_ylabel('Accuracy')
            axs[n].grid(color='b', linestyle='solid', linewidth=0.2)
            axs[n].xaxis.set_ticks(np.arange(0, len(iterations) + 1, len(iterations) // 4))
            axs[n].xaxis.set_ticklabels(np.arange(0, max_plot_iteration + 1, max_plot_iteration // 4))
            
        plt.tight_layout()
        fig_path = '_tmp.png'
        plt.savefig(fig_path)
        print('Save fig to {}'.format(fig_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True)
    parser.add_argument('--select', type=int, required=True)

    args = parser.parse_args()
    
    plot_results(args)