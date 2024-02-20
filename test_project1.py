"""
test_project1.py
Calls the functions written in project1.py for testing purposes
Written by Lexi Reinsborough and Ashley Heath
"""
#import packages
from matplotlib import pyplot as plt
import numpy as np
import project1

#run functions
target_erp, nontarget_erp, target_std, nontarget_std, erp_times, target_epochs, nontarget_epochs = project1.erp_by_subject(3)
project1.plot_erp_intervals(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, subject=3)
p_values = project1.bootstrap(target_erp, nontarget_erp, target_epochs, nontarget_epochs)
rejection_array = project1.plot_fdr_corrected_ps(target_erp, nontarget_erp, target_std, nontarget_std, erp_times, p_values, subject=3)
project1.evaluate_across_subjects()
