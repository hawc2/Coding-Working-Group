## Source - https://www.tidytextmining.com/topicmodeling.html

## Libraries
library(topicmodels)
library(tm)
library(tidyverse)
library(tidyr)
library(tidytext)
library(stringr)
library(magrittr)
library(purrr)
library(Matrix)
library(dplyr)
library(knitr)
library(ggplot2)
library(methods)
library(scales)


## Not sure if this setting is necessary
opts_chunk$set(message = FALSE, warning = FALSE, cache = TRUE)
options(width = 100, dplyr.width = 150)
theme_set(theme_light())

## Preparing Data
text <- scan("LawrenceAll.txt", what ="character", sep="\n")

ps_dtm <- VectorSource(text) %>%
  VCorpus() %>%
  DocumentTermMatrix(control = list(removePunctuation = TRUE,
                                    removeNumbers = TRUE,
                                    stopwords = TRUE))
ps_dtm

ps_tidy <- tidy(ps_dtm, matrix = "gamma")

## LDA Modeling

reviewslda <- LDA(ps_tidy, k = 5, control = list(seed = 1234))

reviewslda <- LDA(ps_dtm, k = 5, control = list(seed = 1234))

reviewslda

reviews_topics <- tidy(reviewslda, matrix = "beta")
reviews_topics

##reviews_dtm <- text.v %>% unnest_tokens(word, text.v) %>%  count(book, word) %>%  cast_dtm(book, word, n)
     
##reviews_dtm

## Visualizing LDA Models

review_top_terms <- reviews_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

review_top_terms

review_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()


## Visualizing Topic Differences

library(tidyr)

beta_spread <- reviews_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

beta_spread

beta_spread %>%
  group_by(direction = log_ratio > 0) %>%
  top_n(10, abs(log_ratio)) %>%
  ungroup() %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_col() +
  labs(y = "Log2 ratio of beta in topic 2 / topic 1") +
  coord_flip()

## Document-Topic Probabilities

reviews_topics <- tidy(reviewslda, matrix = "gamma")
reviews_topics

