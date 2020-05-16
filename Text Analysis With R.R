setwd("~/Documents/TextAnalysisWithR")
text.v <- scan("data/plainText/melville.txt", what ="character", sep="\n")
text.v

## Load Text File

start.v <-which(text.v=="CHAPTER 1. Loomings.")
end.v<-which(text.v=="orphan.")
start.v;end.v

# Clean Text File
length(text.v)
text.v[1]

text.v[start.v]
text.v[start.v-1]
text.v[end.v]
text.v[end.v+1]

start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
novel.lines.v


length(text.v)
length(novel.lines.v)



## Prepare Text File

novel.v<-paste(novel.lines.v, collapse=" ")
length(novel.v)
novel.v[1]

novel.lower.v<-tolower(novel.v)
novel.lower.v

moby.words.l<-strsplit(novel.lower.v,"\\W")

moby.words.l
class(novel.lower.v)
class(moby.words.l)
str(moby.words.l)
moby.words.l

moby.word.v<-unlist(moby.words.l)
not.blanks.v<-which(moby.word.v!="")
not.blanks.v
moby.word.v <-moby.word.v[not.blanks.v]
moby.word.v

moby.word.v[1:10]
moby.word.v[99986]
mypositions.v <- c(4,5,6)
moby.word.v[mypositions.v]
moby.word.v[c(4,5,6)]


## pulling out words from the text

which(moby.word.v=="whale")
moby.word.v[which(moby.word.v=="whale")]
length(moby.word.v[which(moby.word.v=="whale")])
length(moby.word.v)

whale.hits.v <- length(moby.word.v[which(moby.word.v=="whale")])
total.words.v <- length(moby.word.v)
whale.hits.v/total.words.v

length(unique(moby.word.v))

moby.freqs.t <- table(moby.word.v)
moby.freqs.t[1:10]
moby.freqs.t[11:111]
sorted.moby.freqs.t <- sort(moby.freqs.t , decreasing=TRUE)
sorted.moby.freqs.t[c(4,5,6)]
sorted.moby.freqs.t[c(1:10)]
sorted.moby.freqs.t[c(10:100)]

?class
mynums.v <- c(1:10)
plot(mynums.v)

sorted.moby.freqs.t["he"]
sorted.moby.freqs.t["she"]
sorted.moby.freqs.t["fish"]
sorted.moby.freqs.t[15]
sorted.moby.freqs.t["the"]
sorted.moby.freqs.t["him"]/sorted.moby.freqs.t["her"]
length(moby.word.v)
sum(sorted.moby.freqs.t)
sorted.moby.rel.freqs.t <- 100*(sorted.moby.freqs.t/sum(sorted.moby.freqs.t))

sorted.moby.rel.freqs.t["the"]
sorted.moby.rel.freqs.t["whale"]

plot(sorted.moby.rel.freqs.t[1:30], type="b",
     xlab="Top Thirty Words", ylab="Percentage of Full Text", xaxt ="n")
axis(1,1:10, labels=names(sorted.moby.rel.freqs.t [1:10]))

# Clearing Workspace

rm(list =ls())  

setwd("~/Documents/TextAnalysisWithR")
text.v<- scan("data/plainText/melville.txt", what ="character", sep="\n")
text.v

start.v <- which(text.v == "CHAPTER 1. Loomings.")
end.v <- which(text.v == "orphan.")
novel.lines.v <- text.v[start.v:end.v]
novel.lines.v


## Restart (with Moby Dick)

setwd("~hawc1/R Scripting/TextAnalysisWithR")
text.v<- scan("data/plainText/melville.txt", what ="character", sep="\n")
start.v <-which(text.v=="CHAPTER 1. Loomings.")
end.v<-which(text.v=="orphan.")
start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
novel.v<-paste(novel.lines.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
moby.words.l<-strsplit(novel.lower.v,"\\W")
moby.word.v<-unlist(moby.words.l)
not.blanks.v<-which(moby.word.v!="")
moby.word.v <-moby.word.v[not.blanks.v]
moby.word.v

## Quick Set Up with Clean Text from Loading to Graphing Top Word Frequency

text.v<- scan("data/plainText/morrisonsula.txt", what ="character", sep="\n")
novel.v<-paste(text.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
y.words.l<-strsplit(novel.lower.v,"\\W")
y.word.v<-unlist(y.words.l)
not.blanks.v<-which(y.word.v!="")
y.word.v <-y.word.v[not.blanks.v]
y.freqs.t <- table (y.word.v)
sorted.y.freqs.t <- sort(y.freqs.t , decreasing=TRUE)
sorted.y.freqs.t[c(1:100)]
sorted.y.rel.freqs.t <- 100*(sorted.y.freqs.t/sum(sorted.y.freqs.t))
plot(sorted.y.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty-Five Words", ylab="Percentage of Full Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.y.rel.freqs.t [1:25]))

## Jane Austen's Sense and Sensibility (Clean up exercise)

text.v<- scan("data/plainText/austen.txt", what ="character", sep="\n")
text.v
start.v <-which(text.v=="CHAPTER 1")
end.v <-which(text.v=="between themselves, or producing coolness between their husbands.")
start.v;end.v
length(text.v)
start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
text.v[start.v]
text.v[start.v-1]
text.v[end.v]
text.v[end.v+1]
novel.lines.v
length(text.v)
length(novel.lines.v)
novel.v<-paste(novel.lines.v, collapse=" ")
length(novel.v)
novel.v[1]
novel.lower.v<-tolower(novel.v)
novel.lower.v
sense.words.l<-strsplit(novel.lower.v,"\\W")
class(novel.lower.v)
class(sense.words.l)
str(sense.words.l)
sense.word.v<-unlist(sense.words.l)
not.blanks.v<-which(sense.word.v!="")
not.blanks.v
sense.word.v <-sense.word.v[not.blanks.v]
sense.word.v
sense.word.v[1:10]
sense.word.v[99986]
sense.word.v[1000]
which(sense.word.v=="sense")
length(sense.word.v[which(sense.word.v=="sense")])
length(sense.word.v)
sense.hits.v <- length(sense.word.v[which(sense.word.v=="sense")])
total.words.v <- length(sense.word.v)
sense.hits.v/total.words.v
length(unique(sense.word.v))
sense.freqs.t <- table (sense.word.v)
sense.freqs.t[1:10]
sense.freqs.t[11:111]
sorted.sense.freqs.t <- sort(sense.freqs.t , decreasing=TRUE)
sorted.sense.freqs.t[c(4,5,6)]
sorted.sense.freqs.t[c(1:10)]
sorted.sense.freqs.t[c(10:100)]
mynums.v <- c(1:10)
plot(mynums.v)
sorted.sense.freqs.t["he"]
sorted.sense.freqs.t["she"]
sorted.sense.freqs.t["no"]
sorted.sense.freqs.t["him"]/sorted.sense.freqs.t["her"]
length(sense.word.v)
sum(sorted.sense.freqs.t)
sorted.sense.rel.freqs.t <- 100*(sorted.sense.freqs.t/sum(sorted.sense.freqs.t))
sorted.sense.rel.freqs.t["the"]
sorted.sense.rel.freqs.t["marriage"]
plot(sorted.sense.rel.freqs.t[1:10], type="b",
     xlab="Top Ten Words", ylab="Percentage of Full Austen Text", xaxt ="n")
axis(1,1:10, labels=names(sorted.sense.rel.freqs.t [1:10]))

## Comparing Moby Dick to Sense and Sensibility

plot(sorted.sense.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty Five Words", ylab="Percentage of Full Austen Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.sense.rel.freqs.t [1:25]))

plot(sorted.moby.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty Five Words", ylab="Percentage of Full Melville Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.moby.rel.freqs.t [1:25]))



### Heart of Darkness code

text.v<- scan("data/plainText/conrad.txt", what ="character", sep="\n")
text.v
start.v <-which(text.v=="The Nellie, a cruising yawl, swung to her anchor without a flutter of")
end.v <-which(text.v=="sky--seemed to lead into the heart of an immense darkness.")
start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
novel.v<-paste(novel.lines.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
heart.words.l<-strsplit(novel.lower.v,"\\W")
heart.word.v<-unlist(heart.words.l)
not.blanks.v<-which(heart.word.v!="")
heart.word.v <-heart.word.v[not.blanks.v]
which(heart.word.v=="heart")
length(heart.word.v[which(heart.word.v=="heart")])
length(heart.word.v)
heart.hits.v <- length(heart.word.v[which(heart.word.v=="heart")])
total.words.v <- length(heart.word.v)
heart.hits.v/total.words.v
length(unique(heart.word.v))
heart.freqs.t <- table (heart.word.v)
heart.freqs.t[1:100]
sorted.heart.freqs.t <- sort(heart.freqs.t , decreasing=TRUE)
sorted.heart.freqs.t[c(1:10)]
sorted.heart.freqs.t[c(10:100)]
mynums.v <- c(1:10)
plot(mynums.v)
sorted.heart.freqs.t["he"]
sorted.heart.freqs.t["she"]
sorted.heart.freqs.t["him"]/sorted.heart.freqs.t["her"]
length(heart.word.v)
sum(sorted.heart.freqs.t)
sorted.heart.rel.freqs.t <- 100*(sorted.heart.freqs.t/sum(sorted.heart.freqs.t))
sorted.heart.rel.freqs.t["the"]
sorted.heart.rel.freqs.t["savage"]
plot(sorted.heart.rel.freqs.t[1:10], type="b",
     xlab="Top Ten Words", ylab="Percentage of Full Conrad Text", xaxt ="n")
axis(1,1:10, labels=names(sorted.heart.rel.freqs.t [1:10]))


# James Joyce Finnegans Wake

text.v<- scan("data/plainText/finnegans.txt", what ="character", sep="\n")
novel.v<-paste(text.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
finn.words.l<-strsplit(novel.lower.v,"\\W")
finn.word.v<-unlist(finn.words.l)
not.blanks.v<-which(finn.word.v!="")
finn.word.v <-finn.word.v[not.blanks.v]
which(finn.word.v=="virus")
length(finn.word.v[which(finn.word.v=="virus")])
length(finn.word.v)
finn.hits.v <- length(finn.word.v[which(finn.word.v=="virus")])
total.words.v <- length(finn.word.v)
finn.hits.v/total.words.v
length(unique(finn.word.v))
finn.freqs.t <- table (finn.word.v)
finn.freqs.t[1:100]
sorted.finn.freqs.t <- sort(finn.freqs.t , decreasing=TRUE)
sorted.finn.freqs.t[c(1:100)]
mynums.v <- c(1:10)
plot(mynums.v)
sorted.finn.freqs.t["he"]
sorted.finn.freqs.t["she"]
sorted.finn.freqs.t["him"]/sorted.finn.freqs.t["her"]
length(finn.word.v)
sum(sorted.finn.freqs.t)
sorted.finn.rel.freqs.t <- 100*(sorted.finn.freqs.t/sum(sorted.finn.freqs.t))
sorted.finn.rel.freqs.t["the"]
sorted.finn.rel.freqs.t["virus"]
plot(sorted.finn.rel.freqs.t[1:25], type="b",
     xlab="Top Ten Words", ylab="Percentage of Full Finnegans Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.finn.rel.freqs.t [1:25]))




## Comparing Moby Dick Sense and Sensibilty Heart of Darkness 

plot(sorted.heart.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty Five Words", ylab="Percentage of Full Conrad Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.heart.rel.freqs.t [1:25]))

plot(sorted.sense.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty Five Words", ylab="Percentage of Full Austen Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.sense.rel.freqs.t [1:25]))

plot(sorted.moby.rel.freqs.t[1:25], type="b",
     xlab="Top Twenty Five Words", ylab="Percentage of Full Melville Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.moby.rel.freqs.t [1:25]))

plot(sorted.tte67.rel.freqs.t[1:25], type="b",
     xlab="Top 25 Words", ylab="Percentage of Full TTE67 Text", xaxt ="n")
axis(1,1:25, labels=names(sorted.tte67.rel.freqs.t [1:25]))








## Chapter 4 - Tracking Time in Moby Dick

setwd("~/Documents/TextAnalysisWithR")
text.v<- scan("data/plainText/melville.txt", what ="character", sep="\n")
start.v <-which(text.v=="CHAPTER 1. Loomings.")
end.v<-which(text.v=="orphan.")
start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
novel.v<-paste(novel.lines.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
moby.words.l<-strsplit(novel.lower.v,"\\W")
moby.word.v<-unlist(moby.words.l)
not.blanks.v<-which(moby.word.v!="")
moby.word.v <-moby.word.v[not.blanks.v]
n.time.v <- seq(1:length(moby.word.v))
whales.v <- which(moby.word.v == "pip")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[whales.v] <- 1
plot(w.count.v, main="Dispersion Plot of `pip' in Moby Dick",
     xlab="Novel Time", ylab="pip", type="h", ylim=c(0,1), yaxt='n')
ahabs.v <- which(moby.word.v == "queequeg")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[ahabs.v] <- 1
plot(w.count.v, main="Dispersion Plot of `queequeg' in Moby Dick",
     xlab="Novel Time", ylab="queequeg", type="h", ylim=c(0,1), yaxt='n')









#Chapter 4 of Jockers


setwd("~/Documents/TextAnalysisWithR")
text.v<- scan("data/plainText/melville.txt", what ="character", sep="\n")
start.v <-which(text.v=="CHAPTER 1. Loomings.")
end.v<-which(text.v=="orphan.")
start.metadata.v<-text.v[1:start.v -1]
end.metadata.v<-text.v[(end.v+1):length(text.v)]
metadata.v<-c(start.metadata.v, end.metadata.v)
novel.lines.v<-text.v[start.v:end.v]
novel.v<-paste(novel.lines.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
moby.words.l<-strsplit(novel.lower.v,"\\W")
moby.word.v<-unlist(moby.words.l)
not.blanks.v<-which(moby.word.v!="")
moby.word.v <-moby.word.v[not.blanks.v]
moby.word.v
n.time.v <- seq(1:length(moby.word.v))
n.time.v
whales.v <- which(moby.word.v == "whale")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[whales.v] <- 1
plot(w.count.v, main="Dispersion Plot of `whale' in Moby Dick",
     xlab="Novel Time", ylab="whale", type="h", ylim=c(0,1), yaxt='n')
ahabs.v <- which(moby.word.v == "ahab")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[ahabs.v] <- 1
plot(w.count.v, main="Dispersion Plot of `ahab' in Moby Dick",
     xlab="Novel Time", ylab="ahab", type="h", ylim=c(0,1), yaxt='n')
chap.positions.v <- grep("^CHAPTER \\d", novel.lines.v)
novel.lines.v[chap.positions.v]
for(i in 1:length(chap.positions.v)){
  print(chap.positions.v[i])
}
for(i in 1:length(chap.positions.v)){
  print(paste("Chapter ",i, " begins at position ", chap.positions.v[i]), sep="")
}
chapter.raws.l <- list()
chapter.freqs.l <- list()
for (i in 1: length(chap.positions.v)){
  if (i != length(chap.positions.v)){
    chapter.title <- novel.lines.v[chap.positions.v[i]]
    start <- chap.positions.v[i]+1
    end <- chap.positions.v[i+1] - 1
    chapter.lines.v <- novel.lines.v[start:end]
    chapter.words.v <- tolower(paste(chapter.lines.v, collapse=" "))
    chapter.words.l <- strsplit(chapter.words.v, "\\W")
    chapter.word.v <- unlist(chapter.words.l)
    chapter.word.v <- chapter.word.v[which(chapter.word.v!="")]
    chapter.freqs.t <- table(chapter.word.v)
    chapter.raws.l[[chapter.title]] <- chapter.freqs.t
    chapter.freqs.t.rel <- 100*(chapter.freqs.t/sum(chapter.freqs.t))
    chapter.freqs.l[[chapter.title]] <- chapter.freqs.t.rel
}
}

whale.l <- lapply(chapter.freqs.l, '[', 'whale')
whale.l # frequency of whale for each chaper
chapter.freqs.l[[1]]["whale"] # freq of whale for this chapter
whales.m <- do.call(rbind, whale.l)
whales.m
ahab.l <- lapply(chapter.freqs.l, '[', 'ahab')
ahabs.m <- do.call(rbind, ahab.l)
ahabs.m
whales.v <- whales.m[,1]
ahabs.v <- ahabs.m[,1]
whales.ahabs.m <- cbind(whales.v, ahabs.v)
dim(whales.ahabs.m)
whales.ahabs.m
colnames(whales.ahabs.m) <- c("whale", "ahab")
barplot(whales.ahabs.m, beside=T, col="pink")

# Dispersion Plots Beckett's Murphy

setwd("~/Documents/TextAnalysisWithR")
text.v<- scan("data/plainText/BeckettMurphy.txt", what ="character", sep="\n")
novel.v<-paste(text.v, collapse=" ")
novel.lower.v<-tolower(novel.v)
murphy.words.l<-strsplit(novel.lower.v,"\\W")
murphy.word.v<-unlist(murphy.words.l)
not.blanks.v<-which(murphy.word.v!="")
murphy.word.v <-murphy.word.v[not.blanks.v]
n.time.v <- seq(1:length(murphy.word.v))
murphy.v <- which(murphy.word.v == "nothing")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[murphy.v] <- 1
plot(w.count.v, main="Dispersion Plot of `nothing' in Murphy",
     xlab="Novel Time", ylab="nothing", type="h", ylim=c(0,1), yaxt='n')
ahabs.v <- which(murphy.word.v == "queequeg")
w.count.v <- rep(NA,length(n.time.v))
w.count.v[ahabs.v] <- 1
plot(w.count.v, main="Dispersion Plot of `queequeg' in Moby Dick",
     xlab="Novel Time", ylab="queequeg", type="h", ylim=c(0,1), yaxt='n')



## Preparing Dataset for Chapter 6 

text.v <- scan("data/plainText/melville.txt", what="character", sep="\n")
start.v <- which(text.v == "CHAPTER 1. Loomings.")
end.v <- which(text.v == "orphan.")
novel.lines.v <-  text.v[start.v:end.v]
novel.lines.v <- unlist(novel.lines.v)
novel.lines.v <- c(novel.lines.v, "END") # Correction for second Edition.
chap.positions.v <- grep("^CHAPTER \\d", novel.lines.v)
last.position.v <-  length(novel.lines.v)
chap.positions.v  <-  c(chap.positions.v , last.position.v)
chapter.freqs.l <- list()
chapter.raws.l <- list()
for(i in 1:length(chap.positions.v)){
  if(i != length(chap.positions.v)){
    chapter.title <- novel.lines.v[chap.positions.v[i]]
    start <- chap.positions.v[i]+1
    end <- chap.positions.v[i+1]-1
    chapter.lines.v <- novel.lines.v[start:end]
    chapter.words.v <- tolower(paste(chapter.lines.v, collapse=" "))
    chapter.words.l <- strsplit(chapter.words.v, "\\W")
    chapter.word.v <- unlist(chapter.words.l)
    chapter.word.v <- chapter.word.v[which(chapter.word.v!="")] 
    chapter.freqs.t <- table(chapter.word.v)
    chapter.raws.l[[chapter.title]] <-  chapter.freqs.t
    chapter.freqs.t.rel <- 100*(chapter.freqs.t/sum(chapter.freqs.t))
    chapter.freqs.l[[chapter.title]] <- chapter.freqs.t.rel
  }
}