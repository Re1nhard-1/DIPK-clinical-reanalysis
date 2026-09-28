o <- 'work/treatment_response/method_screen/DIPK/third_cohort'
sets <- read.csv(file.path(o,'patient_analysis_sets.csv'))
results <- read.csv(file.path(o,'patient_level_results.csv'))
draws <- read.csv(file.path(o,'patient_bootstrap.csv'))
checks <- data.frame()
for (i in seq_len(nrow(results))) {
  r <- results[i,];d <- sets[sets$analysis==r$analysis,]
  stopifnot(!anyDuplicated(d$patient_key),nrow(d)==r$n,sum(d$observed_pcr)==r$pcr)
  diff <- outer(d$score[d$observed_pcr==1],d$score[d$observed_pcr==0],'-')
  value <- mean((diff>0)+.5*(diff==0))
  v <- draws$auc[draws$analysis==r$analysis]
  stopifnot(length(v)==2000,sum(!is.na(v))==r$valid_draws)
  ci <- quantile(v,c(.025,.975),na.rm=TRUE,type=7)
  stopifnot(abs(value-r$auc)<1e-12,max(abs(ci-c(r$ci_low,r$ci_high)))<1e-12)
  checks <- rbind(checks,data.frame(analysis=r$analysis,n=nrow(d),pcr=sum(d$observed_pcr),
    direct_pair_AUC=value,AUC_difference=value-r$auc,max_interval_difference=max(abs(ci-c(r$ci_low,r$ci_high)))))
}
arrays <- read.csv(file.path(o,'GSE20194_verified_array_crosswalk.csv'))
patient <- read.csv(file.path(o,'GSE20194_patient_means.csv'))
mu <- aggregate(score~patient_key,arrays,mean)
join <- merge(mu,patient,by='patient_key',suffixes=c('_R','_Python'))
stopifnot(nrow(join)==248,max(abs(join$score_R-join$score_Python))<1e-12)
write.csv(checks,file.path(o,'R_unit_crosscheck.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(o,'R_session.txt'))
print(checks,row.names=FALSE)
