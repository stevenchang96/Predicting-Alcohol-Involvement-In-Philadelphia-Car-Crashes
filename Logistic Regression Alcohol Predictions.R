install.packages("aod")
install.packages("ggplot2")
install.packages("rms")
install.packages("gmodels")
install.packages("ROCR")
library(aod)
library(ggplot2)
library(rms)
library(gmodels)
library(ROCR)

dataset <- read.csv("Logistic Regression Data.csv")
head(dataset)
summary(dataset)

#Exploratory analyses

#2a
DRINKING_D.tab <- table(dataset$DRINKING_D)
table(DRINKING_D.tab)
prop.table(DRINKING_D.tab)

#2b
CrossTable(dataset$FATAL_OR_M, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$OVERTURNED, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$CELL_PHONE, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$SPEEDING, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$AGGRESSIVE, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$DRIVER1617, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(dataset$DRIVER65PLUS, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE)

#Cross-tabulation like above with Chi-squared included
CrossTable(dataset$FATAL_OR_M, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$OVERTURNED, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$CELL_PHONE, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$SPEEDING, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$AGGRESSIVE, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$DRIVER1617, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)
CrossTable(dataset$DRIVER65PLUS, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.chisq = FALSE, chisq = TRUE)

#2c - mean and standard deviations of 2 continuous variables for alcohol and non-alcohol related crashes
tapply(dataset$PCTBACHMOR, dataset$DRINKING_D, mean)
tapply(dataset$PCTBACHMOR, dataset$DRINKING_D, sd)

tapply(dataset$MEDHHINC, dataset$DRINKING_D, mean)
tapply(dataset$MEDHHINC, dataset$DRINKING_D, sd)
#tests for significance in above 2 variables across 2 crash categories
t.test(dataset$PCTBACHMOR~dataset$DRINKING_D)
t.test(dataset$MEDHHINC~dataset$DRINKING_D)

#2d - Pearson correlations
correlation <- dataset[c(4:10, 12:13)]
cor(correlation, method = "pearson")

#3 - Logistic regression
logit <- glm(DRINKING_D ~ FATAL_OR_M + OVERTURNED + CELL_PHONE + SPEEDING + AGGRESSIVE + DRIVER1617 + DRIVER65PLUS
             + PCTBACHMOR + MEDHHINC, data = dataset, family = "binomial")
summary(logit)

exp(cbind(OR = coef(logit), confint(logit)))

logitoutput <- summary(logit)
logitcoeffs <- logitoutput$coefficients
logitcoeffs

or_ci <- exp(cbind(OR = coef(logit), confint(logit)))

finallogitoutput <- cbind(logitcoeffs, or_ci)
finallogitoutput

#Sensitivity, specificity, misclassification
fit <- logit$fitted
fit.binary = (fit>=0.02)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.03)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.05)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.07)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.08)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.09)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.1)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.15)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.2)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

fit.binary = (fit>=0.5)
CrossTable(fit.binary, dataset$DRINKING_D, prop.r = FALSE, prop.t = FALSE, prop.c = FALSE, prop.chisq = FALSE)

#Generate ROC curve
a <- cbind(dataset$DRINKING_D, fit)
colnames(a) <- c("labels", "predictions")
head(a)
roc <- as.data.frame(a)
pred <- prediction(roc$predictions, roc$labels)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf)
abline(a = 0, b = 1)

#Calculate sensitivity, specificity, and optimal cutoff
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

print(opt.cut(roc.perf, pred))

#Calculate area under curve
auc.perf = performance(pred, measure ="auc")
auc.perf@y.values

#Re-runs logistic regression without continuous predictors
logit_2 <- glm(DRINKING_D ~ FATAL_OR_M + OVERTURNED + CELL_PHONE + SPEEDING + AGGRESSIVE + DRIVER1617 + DRIVER65PLUS, 
               data = dataset, family = "binomial")
summary(logit_2)

exp(cbind(OR = coef(logit_2), confint(logit_2)))

logitoutput_2 <- summary(logit_2)
logitcoeffs_2 <- logitoutput_2$coefficients
logitcoeffs_2

or_ci_2 <- exp(cbind(OR = coef(logit_2), confint(logit_2)))

finallogitoutput_2 <- cbind(logitcoeffs_2, or_ci_2)
finallogitoutput_2