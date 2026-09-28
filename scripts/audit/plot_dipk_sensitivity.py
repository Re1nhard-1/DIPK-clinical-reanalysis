from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'method_screen'/'DIPK'/'sensitivity'
w=pd.read_csv(P/'within_stratum_auc.csv');a=pd.read_csv(P/'adjusted_associations.csv')
w=w[w.target.eq('observed')].reset_index(drop=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1]})
labels=['GSE25055: within ER\n301 patients, 56 pCR','GSE25055: within PAM50\n306 patients, 57 pCR','GSE32646: within ER\n115 patients, 27 pCR']
for i,row in w.iterrows():
    y=2-i
    axes[0].errorbar(row.estimate,y,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',color='#16737e',capsize=4,lw=2)
    axes[0].text(row.estimate,y+.19,f'{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]',ha='center',fontsize=9)
axes[0].set_yticks([2,1,0],labels);axes[0].set_xlim(.4,.9);axes[0].set_ylim(-.5,2.5)
axes[0].axvline(.5,color='#888888',ls='--',lw=1)
axes[0].set_xlabel('Within-stratum AUC (pair weighted)')
axes[0].set_title('A  Discrimination within background groups',loc='left',fontsize=11,pad=15)
for i,row in a.iterrows():
    y=2-i
    axes[1].errorbar(row.score_OR,y,xerr=[[row.score_OR-row.ci_low],[row.ci_high-row.score_OR]],fmt='o',color='#aa5d35',capsize=4,lw=2)
    axes[1].text(row.score_OR,y+.19,f'{row.score_OR:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]',ha='center',fontsize=9)
axes[1].set_yticks([2,1,0],['ER + age','PAM50 + age','ER + age']);axes[1].set_xscale('log');axes[1].set_xlim(.8,4.2);axes[1].set_ylim(-.5,2.5)
axes[1].set_xticks([1,2,3,4],[1,2,3,4]);axes[1].minorticks_off()
axes[1].axvline(1,color='#888888',ls='--',lw=1)
axes[1].set_xlabel('Adjusted pCR odds ratio per 1 SD score')
axes[1].set_title('B  Conditional association remains',loc='left',fontsize=11,pad=15)
fig.suptitle('Clinical-background sensitivity of saved DIPK predictions',fontsize=15,x=.06,ha='left',y=.99)
fig.subplots_adjust(left=.235,right=.975,bottom=.25,top=.82,wspace=.36)
fig.text(.06,.09,'Observed pathological response only. A: 2,000 patient bootstrap draws; percentile 95% intervals.\nB: exploratory logistic models; Wald 95% intervals. Association does not establish externally validated predictive gain.',fontsize=9,color='#454545',va='center')
fig.savefig(P/'DIPK_background_sensitivity.png',dpi=180)
fig.savefig(P/'DIPK_background_sensitivity.pdf')
print(P/'DIPK_background_sensitivity.png')
