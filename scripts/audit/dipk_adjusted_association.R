args <- commandArgs(trailingOnly=TRUE)
p <- if (length(args)) args[[1]] else "work/treatment_response"
o <- file.path(p,"method_screen","DIPK","sensitivity")
d <- read.csv(file.path(o,"analysis_input.csv"),stringsAsFactors=FALSE)
checks <- data.frame()
auc <- function(y,s) {
  if (!sum(y==1) || !sum(y==0)) return(NA_real_)
  z <- outer(s[y==1],s[y==0],"-")
  mean((z>0)+0.5*(z==0))
}
within <- function(y,s,g) {
  groups <- split(seq_along(y),g)
  numerator <- denominator <- 0
  for (ix in groups) {
    pairs <- sum(y[ix]==1)*sum(y[ix]==0)
    if (pairs>0) {
      numerator <- numerator+auc(y[ix],s[ix])*pairs
      denominator <- denominator+pairs
    }
  }
  numerator/denominator
}
a <- read.csv(file.path(o,"subgroup_auc.csv"))
for (i in seq_len(nrow(a))) {
  r <- a[i,]
  s <- d[d$cohort==r$cohort & d[[r$grouping]]==r$stratum,]
  checks <- rbind(checks,data.frame(kind="subgroup",row=i,difference=auc(s$y,s$score)-r$auc))
}
a <- read.csv(file.path(o,"within_stratum_auc.csv"))
for (i in seq_len(nrow(a))) {
  r <- a[i,]
  s <- d[d$cohort==r$cohort & d[[r$grouping]]!="unknown",]
  ob <- within(s$y,s$score,s[[r$grouping]])
  st <- within(s$stored,s$score,s[[r$grouping]])
  value <- switch(r$target,observed=ob,DLDA30=st,observed_minus_DLDA30=ob-st)
  checks <- rbind(checks,data.frame(kind="within",row=i,difference=value-r$estimate))
}
stopifnot(max(abs(checks$difference))<1e-12)
write.csv(checks,file.path(o,"R_auc_crosscheck.csv"),row.names=FALSE)

fits <- coefficients <- data.frame()
for (cohort in c("GSE25055","GSE32646")) {
  for (grouping in if(cohort=="GSE25055") c("er","pam50") else "er") {
    s <- d[d$cohort==cohort & d[[grouping]]!="unknown",]
    score_mean <- mean(s$score); score_sd <- sd(s$score)
    s$score_z <- (s$score-score_mean)/score_sd
    s$background <- factor(s[[grouping]])
    warnings <- character()
    fit <- withCallingHandlers(glm(y~background+age10+score_z,data=s,family=binomial()),
       warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
    cf <- coef(summary(fit))
    stopifnot(fit$converged,fit$rank==length(coef(fit)),all(is.finite(cf)))
    id <- paste(cohort,grouping,sep="_")
    ci <- cf["score_z",1]+c(-1,1)*qnorm(.975)*cf["score_z",2]
    fits <- rbind(fits,data.frame(model=id,n=nrow(s),pcr=sum(s$y),
      score_mean=score_mean,score_sd=score_sd,beta=cf["score_z",1],se=cf["score_z",2],
      score_OR=exp(cf["score_z",1]),ci_low=exp(ci[1]),ci_high=exp(ci[2]),
      wald_p=cf["score_z",4],converged=fit$converged,rank=fit$rank,
      min_fitted=min(fitted(fit)),max_fitted=max(fitted(fit)),
      iterations=fit$iter,warnings=paste(warnings,collapse="; ")))
    coefficients <- rbind(coefficients,data.frame(model=id,term=rownames(cf),
      estimate=cf[,1],se=cf[,2],row.names=NULL))
    X <- as.data.frame(model.matrix(fit),check.names=FALSE)
    write.csv(cbind(GSM=s$GSM,y=s$y,X),file.path(o,paste0(id,"_design.csv")),row.names=FALSE)
  }
}
write.csv(fits,file.path(o,"adjusted_associations.csv"),row.names=FALSE)
write.csv(coefficients,file.path(o,"R_coefficients.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(o,"R_session.txt"))
print(fits,row.names=FALSE)
