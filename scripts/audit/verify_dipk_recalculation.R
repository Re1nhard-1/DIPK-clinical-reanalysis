p <- 'work/treatment_response/method_screen/DIPK'
x <- read.csv(file.path(p,'GSE25055_fixed_prediction_crosswalk.csv'),stringsAsFactors=FALSE)
x <- x[x$observed_pcr_rd %in% c('pCR','RD'),]
stopifnot(nrow(x)==306,!anyDuplicated(x$GSM))
metric <- function(label) {
  y <- label=='pCR'; s <- x$score; n <- sum(y); m <- sum(!y)
  differences <- outer(s[y],s[!y],'-')
  c(auc=mean((differences>0)+0.5*(differences==0)),
    mean_difference=mean(s[y])-mean(s[!y]),
    welch_p=t.test(s[y],s[!y],var.equal=FALSE)$p.value)
}
out <- rbind(DLDA30=metric(x$archive_group),observed=metric(x$observed_pcr_rd))
write.csv(out,file.path(p,'R_crosscheck.csv'))
capture.output(sessionInfo(),file=file.path(p,'R_crosscheck_session.txt'))
print(out)
