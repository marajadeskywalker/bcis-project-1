# %% header
"""
test_load_p300_data.py
Author: Lexi Reinsborough

test file for load_p300_data.py, manipulating the P300 speller dataset from
Guger, 2009 http://bnci-horizon-2020.eu/database/data-sets

tests data extraction, plots an example plot, then calls functions from load_p300_data.py for each of the subjects
tested on the P300 row/column speller

"""

# %% create variables
data_directory = 'P300Data/'
subject = 3
data_file = f'{data_directory}s{subject}.mat'

# %% importing
import matplotlib.pyplot as plt
import loadmat

data = loadmat.loadmat(data_file)
train_data = data[f's{subject}']['train']  # this is where the data are located

# %% extracting
eeg_time = train_data[0]  # row 0 is time elapsed
eeg_data = train_data[1:9]  # rows 1-8 are channel readings
rowcol_id = train_data[9].astype(int)  # row 9 is the id of the row or column currently flashing, or 0 if none
is_target = train_data[10].astype(bool)  # row 10 is whether the current row or column is the correct one

# %% plotting
plt.close('all')
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

# %% test load_training_eeg and plot_raw_eeg functions
import load_p300_data

data = load_p300_data.load_training_eeg(subject, data_directory)
load_p300_data.plot_raw_eeg(*data, subject)

# %% test load_and_plot_all function
load_p300_data.load_and_plot_all([*range(3, 11)], data_directory)

# %% print docstrings
import inspect

for func in [load_p300_data.load_and_plot_all,
             load_p300_data.load_training_eeg,
             load_p300_data.plot_raw_eeg,
             load_p300_data.decode_message]:
    print(func.__name__)  # function name
    print(inspect.getdoc(func))  # function docstring
    print('\n\n\n')

# %% decode message
"""
by extracting all the row/col ids where is_target is true, we can see the correct rows and columns that were recorded.
all the participants seem to have gotten the string "LUKAS" (the Guger paper describes some participants getting 
"LUCAS", but the third character is 5, 8, or column 5, row 2, which is definitely 'K' not 'C') 
the Guger paper says the character changes after 15 flashes for each row and column,
but from looking at the data it seems to be 60 each, or 120 total,
unless the string was meant to be LLLLUUUUKKKKAAAASSSS.
participants 3 and 5 seem to have been given a rate of 1.64 characters per minute, participant 4 had 1.77,
and participants 6-10 had a rate of 3.52 characters per minute.
"""
for subject_id in range(3, 11):
    eeg_time_n, _, rowcol_id_n, is_target_n = load_p300_data.load_training_eeg(subject_id, data_directory)
    print(subject_id, *load_p300_data.decode_message(eeg_time_n, rowcol_id_n, is_target_n))
