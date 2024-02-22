"""
project1.py
Contains the functions for Project 1, including calculating the target and nontarget ERP values of a specific subject,
plotting those values by channel for a specific subject, calculating FDR-corrected and boostrapped p-values for the difference
between target and nontarget ERPs, and evaluating the prior functions across all subjects and plotting the number of subjects statistically significant
at each point in time on each channel. 
Written by Lexi Reinsborough and Ashley Heath
"""
import itertools
#%%
#import packages
import math
import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import load_p300_data
import plot_p300_erps
from mne.stats import fdr_correction

import plot_topo


#%%
def erp_by_subject(subject: int):
    """
    Description
    -----------
    Caculates the target and nontarget ERP values of a specific subject, and the standard deviation for each of those.

    Parameters
    ----------
    subject : int
        The subject (participant) in the P300 dataset whose responses will be used as input.

    Returns
    -------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Contains the mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp: PxC array of floats, where P is the numbero f samples in each epoch, and C is the number of channels
        Contains the mean values for each channel and sample position where the epoch does not correspond to a target event.
    target_std : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Contains the standard deviation of the mean values for each channel and sample position in all epochs corresponding to a target event.
    nontarget_std : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Contains the standard deviation of the mean values for each channel and sample position in all epochs not corresponding to a target event.
    erp_times : Px1 array of floats, where P is the number of samples in each epoch.
        Contains the list of times (in seconds) at which the data was sampled.
    target_epochs : ExPxC array of floats, where E is the number of epochs corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch corresponds to a target event.
    nontarget_epochs : ExPxC array of floats, where E is the number of epochs not corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch does not correspond to a target event.
    """
    #load and epoch the data using functions from previous labs
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(subject, 'P300Data/')
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)

    #filter the data into target and nontarget epochs
    target_epochs = eeg_epochs[is_target_event]  # (target_count, samples_per_epoch, channel_count)
    nontarget_epochs = eeg_epochs[~is_target_event]  # (nontarget_count, samples_per_epoch, channel_count)

    # mean response on each channel for each event
    target_erp = np.mean(target_epochs, axis=0)
    nontarget_erp = np.mean(nontarget_epochs, axis=0)

    # get standard deviation of those means
    target_std = np.std(target_epochs, axis=0) / np.sqrt(len(target_epochs))
    nontarget_std = np.std(nontarget_epochs, axis=0) / np.sqrt(len(nontarget_epochs))

    return target_erp, nontarget_erp, target_std, nontarget_std, erp_times, target_epochs, nontarget_epochs



#%%
def bootstrapERP(EEGdata, size=None):  # Steps 1-2
    """
    Description
    -----------
    Helper method for the bootstrapping process which resamples the data and draws a new mean from this sample. (Corresponds to ONE mean, and will be called multiple times.)

    Parameters
    ----------
    EEGdata : ExPxC array of floats, where E is the number of epochs being analyzed, P is the number of samples in each epoch and C is the number of channels.
        The raw (not mean) EEG data which the function will randomly sample in order to create new means.
    size: int, optional
        The number of trials which will be used to make up the new sample before averaging. By default, None, and the function uses samples equal to the number of epochs.

    Returns
    -------
    EEG0.mean(0): PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels.
        the new ERP mean calculated from the resampled data.
    """
    ntrials = len(EEGdata)             # Get the number of trials
    if size == None:                   # Unless the size is specified,
        size = ntrials                 # ... choose ntrials
    i = np.random.randint(ntrials, size=size)    # ... draw random trials,
    EEG0 = EEGdata[i]                  # ... create resampled EEG,
    return EEG0.mean(0)                # ... return resampled ERP.
                                       # Step 3: Repeat 3000 times

def bootstrap_iter(target_epochs, nontarget_epochs):
    """
    Description
    -----------
    Helper method for the bootstrapping process which recombines target and nontarget epochs, performs the resampling process from bootstrapERP twice - once
    with size equal to the length of target epochs, and with size equal to the length of nontarget epochs. Then calculates the absolute value of the difference
    between these two values.
    Parameters
    ----------
    target_epochs : ExPxC array of floats, where E is the number of epochs corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch corresponds to a target event.
    nnontarget_epochs : ExPxC array of floats, where E is the number of epochs not corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch does not correspond to a target event.
    Returns
    -------
    abs(mean_difference): PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels.
        The absolute value of the difference between the mean ERP of the target epochs, and the mean ERP of the nontarget epochs.
    """
    #merge the two sets of epochs into a single distribution
    recombined_eeg = np.vstack((target_epochs, nontarget_epochs))

    #resample and calculte the mean for target and nontarget using the helper method, and then get the difference
    mean_target = bootstrapERP(recombined_eeg, size=len(target_epochs))
    mean_nontarget = bootstrapERP(recombined_eeg, size=len((nontarget_epochs)))
    mean_difference = mean_target - mean_nontarget
    return abs(mean_difference)

def bootstrap(target_erp, nontarget_erp, target_epochs, nontarget_epochs):
    """
    Description
    -----------
    Main bootstrapping method which runs the previous helper methods 3000 times to amass 3000 means of the difference between target and nontarget epochs.
    Then calculates what proportion of these bootstrapped values each real difference is larger than, and returns 1-this difference, the p-value

    Parameters
    ----------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values where the epoch doesn't correspond to a target event.
    target_epochs : ExPxC array of floats, where E is the number of epochs corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch corresponds to a target event.
    nontarget_epochs : ExPxC array of floats, where E is the number of epochs not corresponding to a target event, P is the number of samples in each epoch and C is the number of channels.
        Contains the raw EEG data (not the mean) for each channel and sample position for all epochs where the epoch does not correspond to a target event.
    Returns
    -------
    1-p_values : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Returns the true p values by channel and time. Each value is the probability of witnessing results as extreme as
        the actual values assuming the null hypothesis is true and there is no difference between ERPs for target
        and nontarget ERPs. Includes correction for false discovery rate.

    """
    #calculate the 3000 individual bootstrapped means and real difference between target and nontarget ERPs
    bootstrapped_diffs = np.array([bootstrap_iter(target_epochs, nontarget_epochs) for _ in range(3000)])
    real_diffs = abs(target_erp - nontarget_erp)
    
    
    
    num_bootstrappings, num_times, num_channels = np.shape(bootstrapped_diffs)
    
    p_values = np.zeros([num_times, num_channels])
    
    #calculate the proportion of the bootstrapped means which each real value is larger than.
    for time_index in range(num_times):
        for channel_index in range(num_channels):
            bootstrapped_trial = bootstrapped_diffs[:, time_index, channel_index]
            sorted_trial = np.sort(bootstrapped_trial)
            p_values[time_index, channel_index] = np.searchsorted(sorted_trial, real_diffs[time_index, channel_index]) / num_bootstrappings
    
    # the value we have so far is the proportion of bootstrapped means which the real value is larger than.
    # We want the opposite of this, the probability that the results could be random, so we subtract from 1,
    # and correct for false discovery rate.
    _ , corrected_pvals = fdr_correction(1-p_values)
    return corrected_pvals

def plot_erps_and_stats(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, p_values, subject=3):
    """
    Description
    -----------
    Function to plot the amplitude over time per channel, 95% confidence intervals, and p-values for a
    specific subject using the provided ERPs, std errors, and p-values.

    Parameters
    ----------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values where the epoch doesn't correspond to a target event.
    target_std : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Contains the standard deviation of the mean values for each channel and sample position in all epochs corresponding to a target event.
    nontarget_std : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Contains the standard deviation of the mean values for each channel and sample position in all epochs not corresponding to a target event.
    erp_times : Px1 array of floats, where P is the number of samples in each epoch
        The time offsets at which any given sample in eeg_epochs occurred relative to its corresponding event onset, along dimension 1.
    p_values : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        The p-values of the difference between target and nontarget ERPs at each channel at each time point. Each value is the probability of 
        witnessing results as at least as extreme as the actual data compared to the distribution that would be present if the null hypothesis was true.
    subject : int, optional
        index of the subject which the data comes from. used only for labeling plot and filename of output. The default is 3.

    Returns
    -------
    None

    """

    # transpose the erp data to plot, matches average at that sample time to the size of the time array
    target_erp_transpose = np.transpose(target_erp)
    nontarget_erp_transpose = np.transpose(nontarget_erp)

    # get channel count
    channel_count = len(target_erp_transpose)  # same as if nontargets were used

    # plot ERPs for events for each channel
    figure, channel_plots = plt.subplots(math.ceil(channel_count / 3), 3, figsize=(10, 6))
    
    for i in range(3 - channel_count % 3):
        channel_plots[-1][2 - i].remove()  # only 8 channels, 9th plot unnecessary

    #print(np.shape(rejection_array))
    for channel_index in range(channel_count):

        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        channel_plot = channel_plots[row_index][column_index]

        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')

        # plot target and nontarget erp data in the subplot
        target_channel_data = target_erp_transpose[channel_index]
        target_95 = target_std.T[channel_index] * 2
        target_handle, = channel_plot.plot(erp_times, target_channel_data)
        channel_plot.fill_between(erp_times, target_channel_data + target_95,
                                  target_channel_data - target_95, alpha=0.2)
        # https://stackoverflow.com/questions/26217687/combined-legend-entry-for-plot-and-fill-between
        nontarget_channel_data = nontarget_erp_transpose[channel_index]
        nontarget_95 = nontarget_std.T[channel_index] * 2
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp_transpose[channel_index])
        channel_plot.fill_between(erp_times, nontarget_channel_data + nontarget_95,
                                  nontarget_channel_data - target_95, alpha=0.2)
        
        dot_times = np.array(erp_times)[p_values[:, channel_index] < 0.05]
        channel_plot.scatter(dot_times, np.zeros(len(dot_times)), c="#000000")

        # workaround for legend to only display each entry once
        if channel_index == 0:
            target_handle.set_label('Target')
            nontarget_handle.set_label('Nontarget')

        # label each plot's axes and channel number
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time from flash onset (s)')
        channel_plot.set_ylabel('Voltage (μV)')

    # formatting
    figure.suptitle(f'P300 Speller S{subject} Training ERPs')
    figure.legend(loc='lower right', fontsize='xx-large')  # legend in space of nonexistent plot 9
    figure.tight_layout()  # stop axis labels overlapping titles

    # save image
    plt.savefig(f'P300_S{subject}_stat_plots.png')  # save as image


#%%
def evaluate_across_subjects():
    """
    Description
    -----------
    Function which calculates the bootstrapped p-values for each subject in the P300 dataset, and then uses these values to 
    create a graph of how many subjects in the P300 dataset are statistically significant at each point.
    
    Parameters
    ----------
    
    Returns
    -------
    None.

    """
    rejection_array = []
    erps_array = []
    
    #Run the bootstrapping functions on each individual subject in the P300 dataset
    for subject_index in range(3, 11):
        target_erp, nontarget_erp, target_std, nontarget_std, erp_times, target_epochs, nontarget_epochs = erp_by_subject(subject_index)
        p_values = bootstrap(target_erp, nontarget_erp, target_epochs, nontarget_epochs)
        rejection_array.append(0.05 > p_values.T)
        plot_erps_and_stats(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, p_values, subject_index)
        erps_array.append(erp_times)
    rejection_array = np.array(rejection_array)
    erps_array = np.array(erps_array)
    
    #Calculate the number of subjects for which the null hypothesis would be rejected at the 95% significance level
    #for each time point on each channel
    num_rejections_by_channel = np.sum(rejection_array, axis=0)
    print(num_rejections_by_channel.shape)
    
    # get channel count
    channel_count = len(rejection_array[0])
    
    #set up plots
    figure, channel_plots = plt.subplots(math.ceil(channel_count / 3), 3, figsize=(10, 6))
    for i in range(3 - channel_count % 3):
        channel_plots[-1][2 - i].remove()  # only 8 channels, 9th plot unnecessary
        
    #plot the results for each channel
    for channel_index in range(channel_count):
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        channel_plot = channel_plots[row_index][column_index]
        channel_rejections = num_rejections_by_channel[channel_index]
        channel_erps = erps_array[channel_index]
        channel_plot.plot(channel_erps,channel_rejections)
        
        #set up labels 
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time from (s)')
        channel_plot.set_ylabel('# subjects significant')
    
    figure.suptitle('Significant Subjects by Channel')
    figure.tight_layout()  # stop axis labels overlapping titles
    plt.savefig('P300_Significant_Subjects.png')  # save as image


def spatial_map():
    """
    Description
    -----------
    Function which plots the spatial map of the median values of each channel of each subject for the target events,
    across the N2 and P3b time periods, then saves that to scalp_maps.png.

    Parameters
    ----------

    Returns
    -------
    None.

    """
    matplotlib.use('Agg')
    fig, axes = plt.subplots(figsize=(10, 7), nrows=4, ncols=4, layout="constrained")
    ordering = ['Fz', 'PO7', 'P3', 'P4', 'PO8', 'Cz',   'Oz', 'Pz',]
    fig.suptitle(', '.join(ordering))
    for subject_index in range(3, 11):
        plot_y, plot_x = divmod((subject_index - 3)*2, 4)
        target_erp, nontarget_erp, target_std, nontarget_std, erp_times, target_epochs, nontarget_epochs = erp_by_subject(subject_index)
        target_median = np.median(target_epochs, axis=0)# median across
        p3b = (.39, .52) # between 390 ms and 520 ms
        n2 = (.21, .27) # between 210 ms and 270 ms
        p3b_target = np.median(target_median[((erp_times > p3b[0])&(erp_times < p3b[1])),:], axis=0) #
        n2_target = np.median(target_median[((erp_times > n2[0])&(erp_times < n2[1])),:], axis=0)
        plot_topo.plot_topo(ordering, n2_target, ax=axes[plot_y][plot_x])
        plot_topo.plot_topo(ordering, p3b_target, ax = axes[plot_y][plot_x+1])
        axes[plot_y][plot_x].set_title(f'Subject {subject_index} N2')
        axes[plot_y][plot_x+1].set_title(f'Subject {subject_index} P3b')
    fig.savefig(f'scalp_maps.png')