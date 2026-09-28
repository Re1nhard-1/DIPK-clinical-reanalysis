from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).resolve().parent/'method_screen'/'DIPK'/'third_cohort'
r=pd.read_csv(P/'patient_level_results.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(13.5,5),gridspec_kw={'width_ratios':[1.15,1]})
colors=['#246675','#79a9b0','#bd7045','#dbc7af']
left=0
for n,color in zip([130,100,29,19],colors):
 ax[0].barh(1,n,left=left,height=.34,color=color)
 ax[0].text(left+n/2,1,str(n),ha='center',va='center',color='white' if color in colors[:3] else '#222222',fontsize=10)
 left+=n
for n,left,color in [(188,0,'#246675'),(60,188,'#bfcacc')]:
 ax[0].barh(0,n,left=left,height=.34,color=color)
 ax[0].text(left+n/2,0,str(n),ha='center',va='center',color='white' if left==0 else '#222222')
ax[0].text(94,-.38,'Linked to GSE25055',ha='center',fontsize=9)
ax[0].text(218,-.38,'Not linked',ha='center',fontsize=9)
ax[0].set_yticks([1,0],['278 array records','248 source\npatient identifiers'])
ax[0].set_xlim(0,290);ax[0].set_ylim(-.62,1.65);ax[0].set_xlabel('Records or identifiers, as labeled')
ax[0].legend(handles=[Patch(color=c,label=t) for c,t in zip(colors,['MAQC training','MAQC validation','Replicate flag','Outlier flag'])],
 loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=8,frameon=False)
ax[0].set_title('A  Arrays are not independent patients',loc='left',fontsize=11,pad=30)
labels=['Patient-mean scores\n248 patients, 50 pCR','Original MAQC subset\n230 patients, 48 pCR','Patient means, not linked\n60 patients, 18 pCR']
for i,z in r.iterrows():
 y=2-i
 ax[1].errorbar(z.auc,y,xerr=[[z.auc-z.ci_low],[z.ci_high-z.auc]],fmt='o',color='#246675',capsize=4,lw=2)
 ax[1].text(z.auc,y+.22,f'{z.auc:.3f} [{z.ci_low:.3f}, {z.ci_high:.3f}]',ha='center',fontsize=9)
ax[1].set_yticks([2,1,0],labels);ax[1].set_ylim(-.5,2.55);ax[1].set_xlim(.45,.9)
ax[1].axvline(.5,ls='--',color='#888888',lw=1);ax[1].set_xlabel('Observed-response AUC')
ax[1].set_title('B  Patient-level sensitivity analyses',loc='left',fontsize=11,pad=30)
fig.suptitle('GSE20194: correct outcome labels, repeated arrays and shared patient identifiers',x=.04,y=.985,ha='left',fontsize=14)
fig.subplots_adjust(left=.16,right=.97,top=.74,bottom=.23,wspace=.7)
fig.text(.04,.06,'Thirty extra arrays represent repeated source IDs; 188 of 248 IDs link to GSE25055.\nMAQC flags describe the original study, not DIPK training. Intervals: 2,000 patient bootstrap draws. Unlinked IDs do not prove independence.',fontsize=9,color='#444444')
fig.savefig(P/'DIPK_GSE20194_identity.png',dpi=180)
fig.savefig(P/'DIPK_GSE20194_identity.pdf')
