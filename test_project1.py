"""
test_project1.py
Calls the functions written in project1.py for testing purposes
Written by Lexi Reinsborough and Ashley Heath
"""
#import packages
from matplotlib import pyplot as plt
import numpy as np
import project1

#test functions on subject 3
# target_erp, nontarget_erp, target_std, nontarget_std, erp_times, target_epochs, nontarget_epochs = project1.erp_by_subject(3)
# p_values = project1.bootstrap(target_erp, nontarget_erp, target_epochs, nontarget_epochs)
# rejection_array = project1.plot_erps_and_stats(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, p_values, subject=3)
#
# # make all the subject plots
# project1.evaluate_across_subjects()

# make the scalp maps
project1.spatial_map()