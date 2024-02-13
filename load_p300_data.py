# %% header
"""
load_p300_data.py
Author: Lexi Reinsborough

Lab 1 module for manipulating the P300 speller dataset from
Guger, 2009 http://bnci-horizon-2020.eu/database/data-sets

functions for loading the training data, plotting it and saving images, and determining information
about the target string and typing speed

"""
import matplotlib.pyplot as plt
import loadmat


# %% load training
def load_training_eeg(subject, data_directory):
    """
    loads the data from the correct .mat file based on the directory specified and the subject ID
    Parameters
    ----------
    subject: int
    the subject id of the dataset

    data_directory: str
    the base directory of the dataset

    Where T is the number of samples:
    Returns
    -------
    eeg_time: ndarray[float]
    the time elapsed when each data point was recorded in seconds, size Tx1
    eeg_data: ndarray[float]
    the EEG data for each channel, in μV, size 8xT
    rowcol_id: ndarray[int]
    the row and column id of each data point, 1-6 = columns 1-6, 7-12 = rows 1-6, size Tx1
    is_target: ndarray[bool]
    True if the flashing row or column is the target row or column, False otherwise, size Tx1

    """
    data_file = f'{data_directory}s{subject}.mat'
    data = loadmat.loadmat(data_file)
    train_data = data[f's{subject}']['train']  # this is where the data are located

    eeg_time = train_data[0]  # row 0 is time elapsed
    eeg_data = train_data[1:9]  # rows 1-8 are channel readings
    rowcol_id = train_data[9].astype(int)  # row 9 is the id of the row or column currently flashing, or 0 if none
    is_target = train_data[10].astype(bool)  # row 10 is whether the current row or column is the correct one
    return eeg_time, eeg_data, rowcol_id, is_target


def plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject):
    """
       plots the given data in 3 stacked plots for 5 seconds starting 1 second before the first event.

       Where T is the number of samples:
       Parameters
       ----------
       eeg_time: ndarray[float]
       the time elapsed when each data point was recorded in seconds, size Tx1
       eeg_data: ndarray[float]
       the EEG data for each channel, in μV, size 8xT
       rowcol_id: ndarray[int]
       the row and column id of each data point, 1-6 = columns 1-6, 7-12 = rows 1-6, size Tx1
       is_target: ndarray[bool]
       True if the flashing row or column is the target row or column, False otherwise, size Tx1
       subject: int
       the subject id of the dataset

       Returns
       -------
       None

       """
    figure, (rowcol_plot, target_plot, volt_plot) = plt.subplots(nrows=3, ncols=1, sharex='all')  # create subplots
    start_time = 0
    for (time, id_value) in zip(eeg_time, rowcol_id):
        if id_value != 0:
            # the first time a row or column flashes, we save that timestamp and start the plot a second before
            start_time = time - 1
            break

    rowcol_plot.set_xlim([start_time, start_time + 5])  # only have to set xlim once, because sharex is on

    # set labels, plot data, format figure
    for plot in (rowcol_plot, target_plot, volt_plot):
        plot.set_xlabel('Time (s)')
    rowcol_plot.set_ylabel('row/col ID')
    rowcol_plot.plot(eeg_time, rowcol_id)
    target_plot.set_ylabel('matches target?')
    target_plot.plot(eeg_time, is_target)
    volt_plot.set_ylabel('Voltage (μV)')
    volt_plot.set_ylim([-25, 25])
    volt_plot.plot(eeg_time, eeg_data.T)
    figure.tight_layout()

    # switch to correct figure and save image
    plt.figure(figure)
    plt.savefig(f'P300_S{subject}_training_rawdata.png')


# %% load and plot all
def load_and_plot_all(subjects, data_directory):
    """
    loads the data from the correct .mat file based on the directory specified and the subject ID
    Parameters
    ----------
    subjects: list[int]
    the subject ids of the datasets

    data_directory: str
    the base directory of the data
    """
    for subject in subjects:
        plot_raw_eeg(*load_training_eeg(subject, data_directory), subject)


# %% decode
def decode_message(eeg_time, rowcol_id, is_target):
    """
    decodes the message from the row/col ID data and the truth data

    Where T is the number of samples:
    Parameters
    ----------
    eeg_time: ndarray[float]
    the time elapsed when each data point was recorded in seconds, size Tx1
    rowcol_id: ndarray[int]
    the row and column id of each data point, 1-6 = columns 1-6, 7-12 = rows 1-6, size Tx1
    is_target: ndarray[bool]
    True if the flashing row or column is the target row or column, False otherwise, size Tx1

    Returns
    -------
    target_string: str
    the decoded message
    cpm: float
    the number of characters the subject typed per minute
    """

    speller_grid = "ABCDEF GHIJKL MNOPQR STUVWX YZ0123 456789".split()  # lookup table based on Guger speller grid
    # if target is True, it's a correct row/col
    correct_rowcol_ids = [rowcol for rowcol, target in zip(rowcol_id, is_target) if target]
    target_string = ""
    while correct_rowcol_ids:
        # in each group of 120 there's two correct, the smaller one is the column, the larger is the row
        col, row = sorted(set(correct_rowcol_ids[:120]))
        target_string += speller_grid[row - 7][col - 1]
        correct_rowcol_ids = correct_rowcol_ids[120:]
    # collect all the times that any row or column was flashing, and find the timespan between the first and last
    active_times = [time for time, id_value in zip(eeg_time, rowcol_id) if id_value != 0]
    time_span = active_times[-1] - active_times[0]
    # characters per minute is number of characters divided by minutes elapsed
    cpm = len(target_string) / (time_span / 60)
    return target_string, cpm
