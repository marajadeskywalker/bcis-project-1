#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:17:08 2024

test_plot_p300_erps.py

@authors: Claire Leahy and Lexi Reinsborough

test_plot_p300_erps.py is the test script for Lab02 for BCIs S24. This script first loads relevant functions from the module script as well as a helper script from Lab01, load_p300_data.py. The vast majority of the script involves loading, extracting, and plotting data related to subject 3 using the written functions. Specific functionality of each function is detailed in the module script. At the end of the script, the data are evaluated for subjects 3-10, and a series of questions about the event-related potentials (ERPs) and their signficance in the electroencephalography (EEG) data are answered.

Sources and collaborators: N/A

"""


#%% Part 1: Load the Data

# import functions
from load_p300_data import load_training_eeg
from plot_p300_erps import get_events, epoch_data, get_erps, plot_erps

# load training data from subject 3
eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg() # default subject 3, P300Data directory

#%% Part 2: Extract the Event Times and Labels

# call get_events to identify samples where events occurred and if the event was a target event
event_sample, is_target_event = get_events(rowcol_id, is_target)

#%% Part 3: Extract the Epochs

# call epoch_data to get the EEG data over the epoch and the corresponding times
eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=-0.5, epoch_end_time=1)

#%% Part 4: Calculate the ERPs

# call get_erps to calculate mean EEG signals for the target and nontarget events
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)

#%% Part 5: Plot the ERPs

# call plot_erps to plot the ERP data
plot_erps(target_erp, nontarget_erp, erp_times)

#%% Part 6: Discuss the ERPs

# run code for subjects 3-10
for subject_index in range(3,11):
    
    # call relevant functions
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_index) # default path
    event_sample, is_target_event = get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=-0.5, epoch_end_time=1)
    target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
    plot_erps(target_erp, nontarget_erp, erp_times, subject_index)
    
"""

1. Why do we see repeated up-and-down patterns on many of the channels?
   
    Brainwaves have a general set frequency (for multiple waves that sum, each at a different frequency). Consequently, there will likely be some repetitive nature of baseline activity. Additionally, the epochs themselves often occur in evenly spaced intervals for large periods of time, where there may often be overlap with pre- and post-event activity, suggesting that voltages before and after the events are likely not completely isolated. Variation (i.e. peaks) may occur due to some change in activity, whether related to the event itself, related thoughts, motion, or artifacts.
    
2. Why are they more pronounced in some channels but not others?

    Consistency may have occurred in some channels more than others due to the relevance, or lack thereof, to the assignment at hand. For example, the central channel is located in an area that may be more closely linked to motor control compared with language, so while the proximity of the electrodes causes the overall activity to likely have extensive effects, behavior that is less concerned with the task may be more rhythmic.

3. Why does the voltage on some of the channels have a positive peak around half a second after a target flash?
    
    At the foundational level, this experiment is using a P300 Speller, indicating that there is likely a "positive-going" peak around 300ms; however, the figure of 300ms was an early estimate, and in real world data the peak tends to occur closer to 500ms (or 0.5s) after the event (several of the subjects exhibit peaks as early as a quarter of a second after the event). We expect this behavior from the channels which are more involved in the processing of the information presented (target or nontarget); peaks are pretty strongly limited to targets, as nontarget EEG data appears to follow a consistent pattern through the event occurrence. 
    
    This positive peak is most noticeable for subject 3 and occurs in channels 1, 2, 3, 5, and 6 (0-based). Subject 5 has a noticeable negative peak around 0.5 seconds after the event, however, on all channels except channel 4, which appears to be some sort of artifact. Subjects 8 and 10 also observe that negative peak around the same time for all channels except channel 7. Post-synaptic potentials, the voltage measured by EEG is measured as it serves to likely initiate (excite) an action potential. The peaks were almost always associated with the targets rather than the nontargets, suggesting the activation of neurons related to language or concentration.
    
4. Which EEG channels (e.g., Cz) do you think these (the ones described in the last question) might be and why?

    Visual processing of the flashes likely occurred in the occipital lobe, which only corresponded to a single channel. The frontal lobe is involved in concentrating and thinking, which would be necessary for the spelling to occur; only one channel was placed on the frontal lobe. Five of the channels are involved with the parietal lobe, however, which corresponds to the number of channels exhibiting peaks for subject 3, which could make sense seeing as the parietal lobe is often involved with language and attention. Because of the poor spatial resolution of EEG, many of the electrodes are likely to measure similar occurrences, and given the proximity of the parietal lobe electrodes to one another, they could be observing the behavior of the neurons near each other's electrodes. Overall, it is somewhat likely that the channels that observe the peaks include P3, Pz, P4, Po7, and Po8.
    
5. Describe how running the code on multiple subjects influenced your answers.

    Using the data from all of the subjects permitted an overarching view of the general behavior. For each subject, the data produced was unique; however, some general patterns could be obsered, such as the positive or negative peaks that occurred at specific points in time relative to the event.

"""
    