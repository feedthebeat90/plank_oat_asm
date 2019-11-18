library(tidyverse)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Two sources of training data
## 1: Coauthor Sahar's hand-coded data
#
# load("influence0204.RData")
# coded1 = influence0204 %>% group_by(ACN_matrixName) %>% 
#   filter(!is.na(cfscore)) %>% dplyr::select(ACN_matrixName, DIME_contributorname)
# colnames(coded1) = c("amicus", "bonica")
# 
## 2: Mechanical Turk hand-codes
# matches = read.csv("mturk_results4.csv")
# matches = matches %>% filter(bonica_orgname != "")
# match_bonicaname = strsplit(as.character(matches$bonica_orgname), "|", fixed=TRUE)
# match_bonicaname2 = sapply(1:length(match_bonicaname),
#                            FUN=function(x) cbind(as.character(matches$amics_orgname[x]), match_bonicaname[[x]]))
# match_bonicaname2 = do.call(rbind, match_bonicaname2)
# mturk = data.frame(match_bonicaname2)
# colnames(mturk) = c("amicus", "bonica")
# 
## Then merge them together:
# handcoded = rbind(data.frame(coded1), mturk)
# handcoded$match = 1
# handcoded = handcoded[!is.na(handcoded$amicus),]
# handcoded = handcoded[!is.na(handcoded$bonica),]
# 
# handcoded_vec = paste(handcoded$amicus, handcoded$bonica, sep="_")
# 
# 
# gdata::keep(handcoded, handcoded_vec, sure=T)
#
#save(handcoded, handcoded_vec, file="trainingset_stringmatch.RData")
#
#write.csv(handcoded, file="handcoded.csv")

load("trainingset_stringmatch.RData")
# or
# read.csv("handcoded.csv")

## A training set needs positive and negative cases, though
## To find some some nonmatches, just permute the positive cases

## Question to look into (if time allows):
## Optimal ratio of positive to negative cases?

handcoded_vec = paste(handcoded$amicus, handcoded$bonica, sep="_")

tmp = handcoded
tmp$amicus = sample(tmp$amicus, replace=F)
temp_vec = paste(tmp$amicus, tmp$bonica, sep="_")
tmp = tmp[!temp_vec %in% handcoded_vec,]
tmp$match = 0

tmp2 = handcoded
tmp2$amicus = sample(tmp2$amicus, replace=F)
temp2_vec = paste(tmp2$amicus, tmp2$bonica, sep="_")
tmp2 = tmp2[!temp2_vec %in% handcoded_vec,]
tmp2$match = 0

tmp_full = rbind(tmp, tmp2)
tmp_full = tmp_full[!duplicated(tmp_full),]
colnames(tmp_full) = c("amicus", "bonica", "match")

## Full Training set, positive and negative cases
train = rbind(handcoded, tmp_full)
train$amicus = tolower(train$amicus) # note that I preprocess the strings
train$bonica = tolower(train$bonica)

## Calculate covariates
## Some of this takes time
library(stringdist)

methods = c("osa", "lcs", "qgram", "cosine", "jaccard", "jw", "soundex")

## This function creates the features: a series of deterministic string similarity metrics
stringdist_wrap = function(idx){ # idx indexes a row of train
  a = train$amicus[idx]
  b = train$bonica[idx]
  out = sapply(methods,FUN=function(x) stringdist(a, b, useBytes = FALSE, method = x,
                                   weight = c(d = 1,i = 1, s = 1, t = 1), q = 1, p = 0, bt = 0,
                                   nthread = getOption("sd_num_thread")))
  return(out)
}

df = data.frame(t(sapply(1:nrow(train), stringdist_wrap)))

train = cbind(train, df)

## Now we need to train the model. Random Forest is probably the best choice,
## though boosted trees might be good too

library(randomForest)

## rfcv is a cross-validation function.
m_cv = rfcv(trainx = train[,4:ncol(train)], trainy = factor(train$match), cv.fold=5, interaction.depth=2)
summary(m_cv) # Looks pretty good, but we've given it easy cases so far.

## Now train the model for real:
m = randomForest(x = train[,4:ncol(train)], y = factor(train$match), interaction.depth=2)


## Clean up the workspace
gdata::keep(train, m, methods, sure=T)


## Now we need to calculate the test set
## The test set, here, is the set of all possible matches
## That is, every possible combination of names from Set A and names from Set B

# This is the list of organizations that have filed amicus curiae briefs
# We would like to find as many matches for this list as possible

#load("adj_matrix_4_3_18.RData")
#amicus = rownames(dat)
#rm(dat)
#save(amicus, file="amicus_org_names.RData")
#write.csv(amicus, file="amicus_org_names.csv")

load("amicus_org_names.RData")
length(amicus) # 13939 organizations

## This is the target, the set of organizations we think contains SOME matches for the amicus data
#bonica = read.csv("bonica_orgs_reduced.csv", header=F)
#bonica = bonica$V2
#bonica = as.character(bonica)
#save(bonica, file="bonica_org_names.RData")
#write.csv(bonica, file="bonica_org_names.csv")
load("bonica_org_names.RData")


# Remove the rows we already have perfect matches for
amicus = amicus[!amicus %in% bonica]


## If we tried to make a distance matrix that is 12k by 1.3M, we'd need a much bigger computer
## Instead, let's calculate that matrix a small piece at a time
out1 = stringdistmatrix(a=amicus, b=bonica[1:1000],
                                       method=methods[1])
rownames(out1) = amicus
colnames(out1) = bonica[1:1000]
out1 = reshape2::melt(out1)
out = lapply(2:length(methods),
             FUN=function(x) reshape2::melt(stringdistmatrix(a=amicus, b=bonica[1:1000],
                                                             method=methods[x]))$value)
out = do.call(cbind, out)
out2 = cbind(out1, out)
colnames(out2) = c("amicus", "bonica", methods)

## Predict which are matches
out2$preds = predict(m, out2, type="prob")

## Now is the human-in-the-loop part
## Looking at only the most likely matches, which rows are true matches and which are not?

temp = out2[out2$preds[,2]>0.95,]
temp$id = 1:nrow(temp)
temp2 = temp[,c("id", "amicus", "bonica")]
matches = c(1,3,21,29,31,33,55,56,124,146,241,360,372,401,490)
# # note that these won't be the same for you since the model is stochastic
temp = temp[,-c(10,11)]

temp$match = 0
temp$match[matches] = 1


## So now we have found some matches. Let's add them to our training set...
train = rbind(train, temp)

## And try the whole thing again
m = randomForest(x = train[,4:ncol(train)], y = factor(train$match), interaction.depth=2)
gdata::keep(train, m, methods, amicus, bonica, sure=T)

## This time, we'll use the next set of data
out1 = stringdistmatrix(a=amicus, b=bonica[1001:2000],
                        method=methods[1])
rownames(out1) = amicus
colnames(out1) = bonica[1001:2000]
out1 = reshape2::melt(out1)
out = lapply(2:length(methods),
             FUN=function(x) reshape2::melt(stringdistmatrix(a=amicus, b=bonica[1001:2000],
                                                             method=methods[x]))$value)
out = do.call(cbind, out)
out2 = cbind(out1, out)
colnames(out2) = c("amicus", "bonica", methods)
out2$preds = predict(m, out2, type="prob")
temp = out2[out2$preds[,2]>0.95,]
temp$id = 1:nrow(temp)
temp2 = temp[,c("id", "amicus", "bonica")]
temp = temp[,-c(10,11)]
temp$match = 1 # All the matches seemed to work for me this time

## iterate again, this time using the third set of 1000
train = rbind(train, temp)
m2 = rfcv(trainx = train[,4:ncol(train)], trainy = factor(train$match), cv.fold=5, interaction.depth=2)
m = randomForest(x = train[,4:ncol(train)], y = factor(train$match), interaction.depth=2)
gdata::keep(train, m, methods, amicus, bonica, sure=T)

out1 = stringdistmatrix(a=amicus, b=bonica[2001:3000],
                        method=methods[1])
rownames(out1) = amicus
colnames(out1) = bonica[2001:3000]
out1 = reshape2::melt(out1)
out = lapply(2:length(methods),
             FUN=function(x) reshape2::melt(stringdistmatrix(a=amicus, b=bonica[2001:3000],
                                                             method=methods[x]))$value)
out = do.call(cbind, out)
out2 = cbind(out1, out)
colnames(out2) = c("amicus", "bonica", methods)
out2$preds = predict(m, out2, type="prob")
temp = out2[out2$preds[,2]>0.80,]
temp$id = 1:nrow(temp)
temp2 = temp[,c("id", "amicus", "bonica")]
#temp = temp[,-c(10,11)]
#temp$match = 1
#train = rbind(train, temp)

# I didn't find any matches from this batch
## Trying out a bigger batch size

out1 = stringdistmatrix(a=amicus, b=bonica[3001:6000],
                        method=methods[1])
rownames(out1) = amicus
colnames(out1) = bonica[3001:6000]
out1 = reshape2::melt(out1)
out = lapply(2:length(methods),
             FUN=function(x) reshape2::melt(stringdistmatrix(a=amicus, b=bonica[3001:6000],
                                                             method=methods[x]))$value)
out = do.call(cbind, out)
out2 = cbind(out1, out)
colnames(out2) = c("amicus", "bonica", methods)
out2$preds = predict(m, out2, type="prob")
temp = out2[out2$preds[,2]>0.90,] # Lowering the threshold
temp$id = 1:nrow(temp)
temp2 = temp[,c("id", "amicus", "bonica")]
temp = temp[,-c(10,11)]
temp$match = 1
temp$match[c(2,7,12) ] = 0
train = rbind(train, temp)

gdata::keep(train, m, methods, amicus, bonica, sure=T)

## By this point, I'm fairly confident that all the matches my model produces are true matches.
## So I can either go to a bigger computer and generate the remaining predictions
## Or I loop through the bonica data, 3000 or so at a time, and collect all the predictions
## (without updating the training set or the model)