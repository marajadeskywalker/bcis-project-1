from matplotlib import pyplot as plt

import project1

target_erp, nontarget_erp, target_std, nontarget_std, erp_times = project1.erp_by_subject(3)

project1.plot_erp_intervals(target_erp, nontarget_erp, target_std, nontarget_std, erp_times)
