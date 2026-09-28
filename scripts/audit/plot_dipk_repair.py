from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
P=Path(__file__).resolve().parent/'method_screen'/'DIPK'
r=json.loads((P/'label_repair_results.json').read_text(encoding='utf-8'))
rc=pd.read_csv(P/'R_crosscheck.csv',index_col=0)
x=pd.read_csv(P/'GSE25055_fixed_prediction_crosswalk.csv');x=x[x.observed_pcr_rd.isin(['pCR','RD'])]
assert len(x)==306
keys=['same_306_stored_DLDA30','same_306_observed_pathology']; labels=['DLDA30 prediction labels','Observed pathology'];colors=['#8d8ea6','#087e8b']
for i,k in enumerate(keys):
    for field in ['auc','mean_difference','welch_p']:assert np.isclose(rc.iloc[i][field],r[k][field],rtol=1e-9,atol=1e-14)
    y=(x.archive_group if i==0 else x.observed_pcr_rd).eq('pCR')
    assert abs(roc_auc_score(y,x.score)-r[k]['auc'])<1e-12
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(11.5,5.7),gridspec_kw={'width_ratios':[1,1.25]})
fig.subplots_adjust(left=.08,right=.965,bottom=.24,top=.76,wspace=.5)
fig.text(.06,.94,'DIPK: correcting the evaluation label changes measured discrimination',fontsize=15,weight='bold')
fig.text(.06,.885,'GSE25055 | Same 306 patients and saved scores | 90 labels differ (29.4%)',color='#44546a')
for i,k in enumerate(keys):
    y=(x.archive_group if i==0 else x.observed_pcr_rd).eq('pCR');fpr,tpr,_=roc_curve(y,x.score)
    ax[0].plot(fpr,tpr,color=colors[i],lw=2,label=f'{labels[i]} (AUC {r[k]["auc"]:.3f})')
    ci=r['bootstrap']['percentile_95'][['auc_DLDA30','auc_observed'][i]];a=r[k]['auc']
    ax[1].errorbar(a,1-i,xerr=[[a-ci[0]],[ci[1]-a]],fmt='o',color=colors[i],capsize=5,ms=8,lw=2)
    ax[1].text(a,1-i+.22,f'{a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]',ha='center',fontsize=11,color=colors[i])
ax[0].plot([0,1],[0,1],'--',color='#b5bec4',lw=1)
ax[0].set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1))
ax[0].set_title('A  ROC curves',loc='left',weight='bold',fontsize=12,pad=12)
ax[0].legend(loc='lower right',fontsize=8,frameon=False)
ax[1].set(yticks=[1,0],yticklabels=['DLDA30','Observed'],ylim=(-.5,1.6),xlim=(.6,.96),xlabel='AUC with percentile 95% interval')
ax[1].set_title('B  Fixed predictions, different targets',loc='left',weight='bold',fontsize=12,pad=12)
ci=r['bootstrap']['percentile_95']['auc_delta_observed_minus_DLDA30']
fig.text(.61,.125,f'Paired difference: {r["auc_delta"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]',ha='left',fontsize=10,weight='bold')
for a in ax:a.spines[['top','right']].set_visible(False)
fig.text(.06,.087,'2,000 paired patient bootstrap draws, seed 20260926. Four patients with missing observed response excluded from both targets.',fontsize=9,color='#44546a')
fig.text(.06,.047,'Saved predictions recalculated; no retraining. Scores are negative predicted LN IC50. Corrected association remains; this is not a clinical-use validation.',fontsize=9,color='#44546a')
for ext in ['png','pdf']:fig.savefig(P/f'DIPK_label_repair.{ext}',dpi=180,facecolor='white')
plt.close(fig)
manifest={'crosscheck':'Base R direct pairwise AUC and Welch test agree; sklearn AUC also agrees. No independent reviewer implied.',
 'inputs_unchanged':{},'figure':'DIPK_label_repair.png/.pdf','scope':'Saved-result recalculation only'}
base=P.parent.parent
for name,h in r['input_hashes'].items():
    actual=hashlib.sha256((base/name).read_bytes()).hexdigest();assert actual==h;manifest['inputs_unchanged'][name]=True
(P/'recalculation_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('R pairwise AUC, Welch test, sklearn AUC, and input hashes verified.')
