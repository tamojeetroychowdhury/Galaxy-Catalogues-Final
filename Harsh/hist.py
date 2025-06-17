from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df_bns = pd.read_csv('savedata/galData_bns_new.csv')
df_nsbh = pd.read_csv('savedata/galData_nsbh_new.csv')

num_m_frac1_bns = np.array(df_bns['Gal10'])
num_m_frac2_bns = np.array(df_bns['Gal50'])
totalGal_bns = np.array(df_bns['GalTot'])
loc_area_bns = np.array(df_bns['LocArea90'])
avg_dist_bns = np.array(df_bns['EventDist'])
needed_mass_frac = [0.1, 0.5]

num_m_frac1_nsbh = np.array(df_nsbh['Gal10'])
num_m_frac2_nsbh = np.array(df_nsbh['Gal50'])
totalGal_nsbh = np.array(df_nsbh['GalTot'])
loc_area_nsbh = np.array(df_nsbh['LocArea90'])
avg_dist_nsbh = np.array(df_nsbh['EventDist'])
needed_mass_nsbh = [0.1, 0.5]


top_num_1 = 100.0
top_num_2 = 50.0
#num_m_frac1 = [d['m_frac1'] for d in allEvents]
#num_m_frac2 = [d['m_frac2'] for d in allEvents]

fig, axi = plt.subplots(figsize=(12,12))
axi.hist(num_m_frac2_bns, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac2_bns)), 20), alpha=0.4, color='red', label='bns', linewidth=3, weights=np.ones(len(num_m_frac2_bns)) *100/ len(num_m_frac2_bns))
axi.hist(num_m_frac2_nsbh, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac2_nsbh)), 20), alpha=0.4, color='blue', label='nsbh', linewidth=3, weights=np.ones(len(num_m_frac2_nsbh)) *100/ len(num_m_frac2_nsbh))
#{}% mass coverage'.format(needed_mass_frac[1]*100)
#mean = np.mean(num_m_frac2)
#std = np.std(num_m_frac2)
axi.set_xlabel("No. of galaxies required to cover {}% mass fraction".format(needed_mass_frac[1]*100), fontsize=15)
axi.set_ylabel("% of total Events", fontsize=15)
axi.set_xscale('log')
#ax[0][1].set_yscale('symlog', linthresh=1e-3, base=2)
#legend_text = f"mean: {np.mean(num_m_frac2):.2f}, Std: {np.std(num_m_frac2):.2f}, med: {np.median(num_m_frac2):.2f}"
axi.legend(['bns (median=%.2f)' % (np.nanmedian(num_m_frac2_bns)), 
            'nsbh (median=%.2f)' % (np.nanmedian(num_m_frac2_nsbh))], loc='upper left')
plt.savefig('paper_plot.png')


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
ax[0][0].hist(num_m_frac1_bns, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac1_bns)), 20), alpha=0.4, color='red', label='bns', linewidth=3, weights=np.ones(len(num_m_frac1_bns)) *100/ len(num_m_frac1_bns))
#ax[0][0].hist(num_m_frac1_bns, histtype='step', bins=10**np.linspace(1, 5, 20), alpha=0.4, color='red', label='bns', linewidth=3)
# label='{}% mass coverage'.format(needed_mass_frac[0]*100)
# ax[0][0].hist(num_m_frac1_nsbh, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac1_nsbh)), 20), alpha=0.4, color='blue', label='nsbh', linewidth=3)
ax[0][0].hist(num_m_frac1_nsbh, histtype='step', bins=10**np.linspace(1, 5, 20), alpha=0.4, color='blue', label='nsbh', linewidth=3, weights=np.ones(len(num_m_frac1_nsbh)) *100/ len(num_m_frac1_nsbh))
ax[0][0].set_xlabel("No. of galaxies required to cover {}% mass fraction".format(needed_mass_frac[0]*100), fontsize=15)
ax[0][0].set_ylabel("% of total Events", fontsize=15)
ax[0][0].set_xscale('log')
#ax[0][0].set_yscale('symlog', linthresh=1e-3, base=2)
#legend_text = f"mean: {np.mean(num_m_frac1):.2f}, Std: {np.std(num_m_frac1):.2f}, med: {np.median(num_m_frac1):.2f}"
ax[0][0].legend(['bns (median=%.2f)' % (np.nanmedian(num_m_frac1_bns)), 
            'nsbh (median=%.2f)' % (np.nanmedian(num_m_frac1_nsbh))], loc='upper left')
# ax[0][0].legend(['bns (mean=%.2f, std=%.2f)' % (np.mean(num_m_frac1_bns), np.std(num_m_frac1_bns)), 
#             'nsbh (mean=%.2f, std=%.2f)' % (np.mean(num_m_frac1_nsbh), np.std(num_m_frac1_nsbh))], loc="upper left")
#ax[0][0].legend([legend_text])
ax[0][1].hist(num_m_frac2_bns, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac2_bns)), 20), alpha=0.4, color='red', label='bns', linewidth=3, weights=np.ones(len(num_m_frac2_bns)) *100/ len(num_m_frac2_bns))
ax[0][1].hist(num_m_frac2_nsbh, histtype='step', bins=np.logspace(np.log10(1),np.log10(np.max(num_m_frac2_nsbh)), 20), alpha=0.4, color='blue', label='nsbh', linewidth=3, weights=np.ones(len(num_m_frac2_nsbh)) *100/ len(num_m_frac2_nsbh))
#{}% mass coverage'.format(needed_mass_frac[1]*100)
#mean = np.mean(num_m_frac2)
#std = np.std(num_m_frac2)
ax[0][1].set_xlabel("No. of galaxies required to cover {}% mass fraction".format(needed_mass_frac[1]*100), fontsize=15)
ax[0][1].set_ylabel("% of total Events", fontsize=15)
ax[0][1].set_xscale('log')
#ax[0][1].set_yscale('symlog', linthresh=1e-3, base=2)
#legend_text = f"mean: {np.mean(num_m_frac2):.2f}, Std: {np.std(num_m_frac2):.2f}, med: {np.median(num_m_frac2):.2f}"
ax[0][1].legend(['bns (median=%.2f)' % (np.nanmedian(num_m_frac2_bns)), 
            'nsbh (median=%.2f)' % (np.nanmedian(num_m_frac2_nsbh))], loc='upper left')
# ax[0][1].legend(['bns (mean=%.2f, std=%.2f)' % (np.mean(num_m_frac2_bns), np.std(num_m_frac2_bns)), 
#             'nsbh (mean=%.2f, std=%.2f)' % (np.mean(num_m_frac2_nsbh), np.std(num_m_frac2_nsbh))], loc="upper left")

top_frac_1_bns = [(top_num_1/d) if(d>100) else 1 for d in totalGal_bns]
top_frac_1_nsbh = [(top_num_1/d) if(d>100) else 1 for d in totalGal_nsbh]
top_frac_2_bns = [(top_num_2/d) if(d>50) else 1 for d in totalGal_bns]
top_frac_2_nsbh = [(top_num_2/d) if(d>50) else 1 for d in totalGal_nsbh]
#width = 0.5
#x = np.linspace(1, len(top_100_frac), len(top_100_frac))
#ax[1][0].bar(x, top_100_frac, alpha=0.4, color='blue', align='edge')
ax[1][0].hist(top_frac_1_bns, bins=10, alpha=0.5, color='green', histtype='step', linewidth=3, weights=np.ones(len(top_frac_1_bns)) *100/ len(top_frac_1_bns))
ax[1][0].hist(top_frac_1_nsbh, bins=10, alpha=0.5, color='purple', histtype='step', linewidth=3, weights=np.ones(len(top_frac_1_nsbh)) *100/ len(top_frac_1_nsbh))
ax[1][0].set_ylabel("% of Events", fontsize=15)
ax[1][0].set_xlabel("Fraction of mass covered by top 100 galaxies", fontsize=15)
ax[1][0].set_yscale('symlog')
#ax[1][0].set_xticks(x + width/2)
#ax[1][0].set_xticklabels(x)
#ax[1][0].set_xscale('log')
#ax[1][0].legend(['bns (mean=%.2f, std=%.2f)' % (np.mean(top_frac_1_bns), np.std(top_frac_1_bns)), 
            #'nsbh (mean=%.2f, std=%.2f)' % (np.mean(top_frac_1_nsbh), np.std(top_frac_1_nsbh))])
ax[1][0].legend(['bns' , 'nsbh' ], loc='upper right')
ax[1][1].hist(top_frac_2_bns, bins=10, alpha=0.5, color='green', histtype='step', linewidth=3, weights=np.ones(len(top_frac_2_bns)) *100/ len(top_frac_2_bns))
ax[1][1].hist(top_frac_2_nsbh, bins=10, alpha=0.5, color='purple', histtype='step', linewidth=3, weights=np.ones(len(top_frac_2_nsbh)) *100/ len(top_frac_2_nsbh))
#ax[1][1].bar(x, top_50_frac, alpha=0.4, color='red', align='edge')
#mean = np.mean(num_m_frac2)
#std = np.std(num_m_frac2)
ax[1][1].set_xlabel("Fraction of mass covered by top 50 galaxies", fontsize=15)
ax[1][1].set_ylabel("% of Events", fontsize=15)
#ax[1][1].set_xscale('log')
ax[1][1].set_yscale('symlog')
#ax[1][1].set_xticks(x + width/2)
#ax[1][1].set_xticklabels(x)
ax[1][1].legend(['bns' , 'nsbh' ], loc='upper right')
# ax[1][1].legend(['bns (mean=%.2f, std=%.2f)' % (np.mean(top_frac_2_bns), np.std(top_frac_2_bns)), 
#             'nsbh (mean=%.2f, std=%.2f)' % (np.mean(top_frac_2_nsbh), np.std(top_frac_2_nsbh))])

plt.savefig('savedata/complete/hist_comb_last.png')
