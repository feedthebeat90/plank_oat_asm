setwd("D:/Dropbox/Stringmatch")
library(tidyverse)

dat = readLines("open_secrets_orgs.csv")

## Find D{0123456789}{10}
## Replace with that thing + \n

strsplit2 <- function(x,
                     split,
                     type = "remove",
                     perl = FALSE,
                     ...) {
  if (type == "remove") {
    # use base::strsplit
    out <- base::strsplit(x = x, split = split, perl = perl, ...)
  } else if (type == "before") {
    # split before the delimiter and keep it
    out <- base::strsplit(x = x,
                          split = paste0("(?<=.)(?=", split, ")"),
                          perl = TRUE,
                          ...)
  } else if (type == "after") {
    # split after the delimiter and keep it
    out <- base::strsplit(x = x,
                          split = paste0("(?<=", split, ")"),
                          perl = TRUE,
                          ...)
  } else {
    # wrong type input
    stop("type must be remove, after or before!")
  }
  return(out)
}


dat2 = strsplit2(dat, "D[0-9]{9}", type="after")
dat3 = unlist(dat2)
dat3[1] = gsub("orgName,parentName,parentID", "", dat3[1])

dat4 = strsplit(dat3, ",")
dat4 = dat4[sapply(dat4, length) == 3]
dat4 = do.call(rbind, dat4)
dat4 = data.frame(dat4)
colnames(dat4) = c("org", "parent", "parentID")

## Now I match this list to something else, preserving parentID
## It might be the orgs from comments_names
## The training data might be matches