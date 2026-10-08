p <- 'work/treatment_response'
src <- file.path(p,'vaen_source/Figure/Figure5/GSE25055/03.pCR.R')
x <- read.csv(file.path(p,'GSE25055_label_crosswalk.csv'),stringsAsFactors=FALSE)
expr <- as.list(parse(src))
txt <- vapply(expr,function(e)paste(deparse(e),collapse=' '),character(1))
rename <- which(grepl('dlda30_prediction:',txt,fixed=TRUE))[1:2]
fac <- which(grepl('factor(dat[, 2]',txt,fixed=TRUE))[1]
stopifnot(length(rename)==2,!anyNA(rename),!is.na(fac))
scope <- new.env(parent=baseenv())
scope$dat <- data.frame(Response=rep(NA_real_,nrow(x)),pCR=x$raw_selected_characteristic,stringsAsFactors=FALSE)
for (i in c(rename,fac)) eval(expr[[i]],envir=scope)
labels <- as.character(scope$dat[,2])
stopifnot(identical(labels,x$dlda30_prediction),sum(labels=='pCR')==122,sum(labels=='RD')==188)
write.csv(data.frame(GSM=x$GSM,original_snippet_label=labels),file.path(p,'original_snippet_labels.csv'),row.names=FALSE)
print(table(labels))
