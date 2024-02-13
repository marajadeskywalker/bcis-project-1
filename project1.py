import math

import numpy as np
from matplotlib import pyplot as plt

import load_p300_data
import plot_p300_erps

def erp_by_subject(subject: int):
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(subject, 'P300Data/')
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_epochs = eeg_epochs[is_target_event]  # (target_count, samples_per_epoch, channel_count)
    nontarget_epochs = eeg_epochs[~is_target_event]  # (nontarget_count, samples_per_epoch, channel_count)

    # mean response on each channel for each event
    target_erp = np.mean(target_epochs, axis=0)
    nontarget_erp = np.mean(nontarget_epochs, axis=0)
    target_std = np.std(target_epochs, axis=0) / np.sqrt(len(target_epochs))
    nontarget_std = np.std(nontarget_epochs, axis=0) / np.sqrt(len(nontarget_epochs))

    return target_erp, nontarget_erp, target_std, nontarget_std, erp_times


def plot_erp_intervals(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, subject=3):
    """
    Description
    -----------
    Transforms mean channel data into 8 matplotlib plots and saves them in a .png image.

    Parameters
    ----------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values where the epoch doesn't correspond to a target event.
     erp_times : Px1 array of floats, where P is the number of samples in each epoch
        The time offsets at which any given sample in eeg_epochs occurred relative to its corresponding event onset, along dimension 1.
    subject : int, optional
        index of the subject which the data comes from. used only for labeling plot and filename of output. The default is 3.

    Returns
    -------
    None.

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
        nontarget_channel_data = nontarget_erp_transpose[channel_index]
        nontarget_95 = nontarget_std.T[channel_index] * 2
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp_transpose[channel_index])
        channel_plot.fill_between(erp_times, nontarget_channel_data + nontarget_95,
                                  nontarget_channel_data - target_95, alpha=0.2)

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
    plt.savefig(f'P300_S{subject}_channel_plots.png')  # save as image