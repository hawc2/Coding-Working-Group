
# Install Packages --------------------------------------------------------




## https://www.datacamp.com/community/tutorials/sentiment-analysis-R
install.packages("dplyr")
library(dplyr) #Data manipulation (also included in the tidyverse package)
library(tidytext) #Text mining
library(tidyr) #Spread, separate, unite, text mining (also included in the tidyverse package)
library(widyr) #Use for pairwise correlation

#Visualizations!
library(ggplot2) #Visualizations (also included in the tidyverse package)
library(ggrepel) #`geom_label_repel`
library(gridExtra) #`grid.arrange()` for multi-graphs
library(knitr) #Create nicely formatted output tables
library(kableExtra) #Create nicely formatted output tables
library(formattable) #For the color_tile function
library(circlize) #Visualizations - chord diagram
library(memery) #Memes - images with plots
library(magick) #Memes - images with plots (image_read)
library(yarrr)  #Pirate plot
library(radarchart) #Visualizations
library(igraph) #ngram network diagrams
library(ggraph) #ngram network diagrams


# Set Themes --------------------------------------------------------------



#Define some colors to use throughout
my_colors <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#D65E00")

#Customize ggplot2's default theme settings
#This tutorial doesn't actually pass any parameters, but you may use it again in future tutorials so it's nice to have the options
theme_lyrics <- function(aticks = element_blank(),
                         pgminor = element_blank(),
                         lt = element_blank(),
                         lp = "none")
{
  theme(plot.title = element_text(hjust = 0.5), #Center the title
        axis.ticks = aticks, #Set axis ticks to on or off
        panel.grid.minor = pgminor, #Turn the minor grid lines on or off
        legend.title = lt, #Turn the legend title on or off
        legend.position = lp) #Turn the legend on or off
}

#Customize the text tables for consistency using HTML formatting
my_kable_styling <- function(dat, caption) {
  kable(dat, "html", escape = FALSE, caption = caption) %>%
    kable_styling(bootstrap_options = c("striped", "condensed", "bordered"),
                  full_width = FALSE)
}


# Import CSV Data ---------------------------------------------------------


prince_data <- read.csv('AS.csv', stringsAsFactors = FALSE, row.names = 1)

  
  glimpse(prince_data) #Transposed version of `print()`


  #Created in the first tutorial
 # undesirable_words <- c("prince", "chorus", "repeat", "lyrics",
    ##                     "theres", "bridge", "fe0f", "yeah", "baby",
     ##                    "alright", "wanna", "gonna", "chorus", "verse",
     ##                    "whoa", "gotta", "make", "miscellaneous", "2",
     ##                    "4", "ooh", "uurh", "pheromone", "poompoom", "3121",
      ##                   "matic", " ai ", " ca ", " la ", "hey", " na ",
      ##                   " da ", " uh ", " tin ", "  ll", "transcription",
      ##                   "repeats", "la", "da", "uh", "ah")
  
  #Create tidy text format: Unnested, Unsummarized, -Undesirables, Stop and Short words
  prince_tidy <- prince_data %>%
    unnest_tokens(word, User.Review) %>% #Break the lyrics into individual words
     #  filter(!word %in% undesirable_words) %>% #Remove undesirables
    filter(!nchar(word) < 3) %>% #Words like "ah" or "oo" used in music
    anti_join(stop_words) #Data provided by the tidytext package


  glimpse(prince_tidy) #From `dplyr`, better than `str()`.

  
  

# Word Summary ------------------------------------------------------------

  
  
  word_summary <- prince_tidy %>%
    mutate(Date = ifelse(is.na(Date),"NONE", Date)) %>%
    group_by(Date, Rating) %>%
    mutate(word_count = n_distinct(word)) %>%
    select(Date, Review_Date = Date, Review_Rating = Rating, word_count) %>%
    distinct() %>% #To obtain one record per song
    ungroup()
  
  word_summary
  
  pirateplot(formula =  word_count ~ Review_Date + Review_Rating, #Formula
             data = word_summary, #Data frame
             xlab = NULL, ylab = "Rating", #Axis labels
             main = "Date", #Plot title
             pal = "google", #Color scheme
             point.o = .2, #Points
             avg.line.o = 1, #Turn on the Average/Mean line
             theme = 0, #Theme
             point.pch = 16, #Point `pch` type
             point.cex = 1.5, #Point size
             jitter.val = .1, #Turn on jitter to see the songs better
             cex.lab = .9, cex.names = .7) #Axis label size


  
  Date_Rating <- prince_data %>%
    select(Date, Rating) %>%
    group_by(Date) %>%
    summarise(Date_count = n())
  
  
  id <- seq_len(nrow(Date_Rating))
  Date_Rating <- cbind(Date_Rating, id)
  label_data = Date_Rating
  number_of_bar = nrow(label_data) #Calculate the ANGLE of the labels
  angle = 90 - 360 * (label_data$id - 0.5) / number_of_bar #Center things
  label_data$hjust <- ifelse(angle < -90, 1, 0) #Align label
  label_data$angle <- ifelse(angle < -90, angle + 180, angle) #Flip angle
  ggplot(Date_Rating, aes(x = as.factor(id), y = Date_count)) +
    geom_bar(stat = "identity", fill = alpha("purple", 0.7)) +
    geom_text(data = label_data, aes(x = id, y = Date_count + 10, label = Date, hjust = hjust), color = "black", alpha = 0.6, size = 3, angle =  label_data$angle, inherit.aes = FALSE ) +
    coord_polar(start = 0) +
    ylim(-20, 150) + #Size of the circle
    theme_minimal() +
    theme(axis.text = element_blank(),
          axis.title = element_blank(),
          panel.grid = element_blank(),
          plot.margin = unit(rep(-4,4), "in"),
          plot.title = element_text(margin = margin(t = 10, b = -10)))
  
  

#   Sentiment Analysis ----------------------------------------------------

  
  prince_nrc <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))
  reviews_joy <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
    filter(sentiment == "joy")
  
  new_file <-  reviews_joy %>%
    filter(Date != "NA") %>% #Remove Date without release dates
    count(Date, Rating)  #
  
  new_file
  
  decade_chart <-  prince_data %>%
    filter(Date != "NA") %>% #Remove Date without release dates
    count(Date, Rating)  #Get SONG count per chart level per decade. Order determines top or bottom.
  
  decade_chart
  
  library(ggplot2)
  theme_set(theme_bw())
  
  # Plot
  ggplot(decade_chart, aes(x=Rating, y=Date)) + 
    geom_point(size=3) + 
    geom_segment(aes(x=0, 
                     xend=Rating, 
                     y=2010, 
                     yend=Date)) + 
    labs(title="Hurt Locker User Reviews", 
         subtitle="Ratings over Time") + 
    theme(axis.text.x = element_text(angle=65, vjust=0.6))
  
  
  
  
  qplot(x=Rating, y=Date,
        data=decade_chart, na.rm=TRUE,
        main='Ratings over Time for Fear',
        xlab='Date', ylab='Rating')
  

  
  
  circos.clear() #Very important - Reset the circular layout parameters!
  grid.col = c("1970s" = my_colors[1], "1980s" = my_colors[2], "1990s" = my_colors[3], "2000s" = my_colors[4], "2010s" = my_colors[5], "Charted" = "grey", "Uncharted" = "grey") #assign chord colors
  # Set the global parameters for the circular layout. Specifically the gap size
  circos.par(gap.after = c(rep(5, length(unique(decade_chart[[1]])) - 1), 15,
                           rep(5, length(unique(decade_chart[[2]])) - 1), 15))
  
  chordDiagram(decade_chart, grid.col = grid.col, transparency = .2)
  title("Relationship Between Chart and Decade")
  
  
  
  
  
  
  
  
  
  ## Sentiment Groups

  
  prince_nrc <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))
  
  prince_nrc
  
  prince_nrc_sub <- prince_tidy %>%
    inner_join(get_sentiments("nrc")) %>%
    filter(!sentiment %in% c("positive", "negative"))
 
  prince_nrc_sub
  

reviews_joy <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
  filter(sentiment == "joy")
  
reviews_joy

reviews_fear <- prince_tidy %>%
  inner_join(get_sentiments("nrc"))%>%
  filter(sentiment == "fear")




prince_polarity_chart <- reviews_fear %>%
  count(sentiment, Date) %>%
  spread(sentiment, n, fill = 0) 


#Polarity by Date
plot1 <- prince_polarity_chart %>%
  ggplot( aes(Date, fill = Date)) +
  geom_col() +
  scale_fill_manual(values = my_colors[3:5]) +
  geom_hline(yintercept = 0, color = "red") +
  theme_lyrics() + theme(plot.title = element_text(size = 11)) +
  xlab(NULL) + ylab(NULL) +
  ggtitle("Polarity By Date")

plot1

prince_nrc <- prince_tidy %>%
  inner_join(get_sentiments("nrc"))

  nrc_plot <- prince_nrc %>%
    group_by(sentiment) %>%
    summarise(word_count = n()) %>%
    ungroup() %>%
    mutate(sentiment = reorder(sentiment, word_count)) %>%
    #Use `fill = -word_count` to make the larger bars darker
    ggplot(aes(sentiment, word_count, fill = -word_count)) +
    geom_col() +
    guides(fill = FALSE) + #Turn off the legend
    theme_lyrics() +
    labs(x = NULL, y = "Word Count") +
    scale_y_continuous(limits = c(0, 15000)) + #Hard code the axis limit
    ggtitle("Review Sentiment") +
    coord_flip()
  

  nrc_plot
  
  
  
  
  bing_plot <- prince_bing %>%
    group_by(sentiment) %>%
    summarise(word_count = n()) %>%
    ungroup() %>%
    mutate(sentiment = reorder(sentiment, word_count)) %>%
    ggplot(aes(sentiment, word_count, fill = sentiment)) +
    geom_col() +
    guides(fill = FALSE) +
    theme_lyrics() +
    labs(x = NULL, y = "Word Count") +
    scale_y_continuous(limits = c(0, 8000)) +
    ggtitle("Review Sentiment") +
    coord_flip()
  
  bing_plot
  
  
  prince_nrc <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))
  
  options(scipen=999)  # turn-off scientific notation like 1e+48
  library(ggplot2)
  theme_set(theme_bw())  # pre-set the bw theme.
  ##data("reviews", package = "ggplot2")
  # midwest <- read.csv("http://goo.gl/G1K41K")  # bkup data source
  
 
  # Scatterplot
  gg <- ggplot(prince_nrc, aes(x=Rating, y=Date)) + 
    geom_point(aes(col=sentiment, size=)) + 
    geom_smooth(method="loess", se=F) + 
    xlim(c(0, 0.1)) + 
    ylim(c(0, 500000)) + 
    labs(subtitle="Area Vs Population", 
         y="Population", 
         x="Area", 
         title="Scatterplot", 
         caption = "Source: midwest")
  
  plot(gg)
  
  prince_polarity_chart <- prince_bing %>%
    count(sentiment, Date) %>%
    spread(sentiment, n, fill = 0) %>%
    mutate(polarity = positive - negative,
           percent_positive = positive / (positive + negative) * 100)
  
  #Polarity by Date
  plot1 <- prince_polarity_chart %>%
    ggplot( aes(Rating, polarity, fill = Date)) +
    geom_col() +
    scale_fill_manual(values = my_colors[3:5]) +
    geom_hline(yintercept = 0, color = "red") +
    theme_lyrics() + theme(plot.title = element_text(size = 11)) +
    xlab(NULL) + ylab(NULL) +
    ggtitle("Polarity By Date")
  
  plot1
  
  #Percent positive by Date
  plot2 <- prince_polarity_chart %>%
    ggplot( aes(chart_level, percent_positive, fill = chart_level)) +
    geom_col() +
    scale_fill_manual(values = c(my_colors[3:5])) +
    geom_hline(yintercept = 0, color = "red") +
    theme_lyrics() + theme(plot.title = element_text(size = 11)) +
    xlab(NULL) + ylab(NULL) +
    ggtitle("Percent Positive By Chart Level")
  
  grid.arrange(plot1, plot2, ncol = 2)
  
  prince_bing <- prince_tidy %>%
    inner_join(get_sentiments("bing"))
  

  
  

#   Sentiment Analysis Continued ------------------------------------------

  
  
  
  
  ##
  
  prince_nrc <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))
  reviews_joy <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
    filter(sentiment == "joy")
  
  new_file <-  reviews_joy %>%
    filter(Date != "NA") %>% #Remove Date without release dates
    count(Date, Rating)  #
  
  
  new_file
  
  
  prince_nrc <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))
  prince_nrc
  prince_nrc_sub <- prince_tidy %>%
    inner_join(get_sentiments("nrc")) %>%
    filter(!sentiment %in% c("negative", "positive"))
  
  prince_nrc_sub
  
  anger_num <- prince_nrc_sub %>%
    count(sentiment)
  
  reviews_joy <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
    filter(sentiment == "joy")
  reviews_joy
  
  reviews_polarity <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
    filter(sentiment == "positive", "negative")
  
  install.packages("summarytools")
  library(summarytools)
  
  reviews_sadness <- prince_nrc %>%
    filter(sentiment == "sadness") %>%
    group_by(word) %>%
    summarize(freq = mean(freq))
  
  reviews_sadness
  
  reviews_joy
  datejoy <- reviews_joy %>% count(Date, sentiment)
  datejoy
  
  reviews_anger <- prince_tidy %>%
    inner_join(get_sentiments("nrc"))%>%
    filter(sentiment == "anger")
  reviews_anger
  
  dateanger <- reviews_anger %>% count(Date, sentiment)
  dateanger
  
  new_file <-  prince_nrc_sub %>%
    count(Date, sentiment)
  
  new_file
  
  new_file2 <- prince_nrc_sub %>%
    count(Date, Rating)  #
  new_file2
  
  options(scipen=999)  # turn-off scientific notation like 1e+48
  library(ggplot2)
  theme_set(theme_bw())  # pre-set the bw theme.
  ##data("reviews", package = "ggplot2")
  # midwest <- read.csv("http://goo.gl/G1K41K")  # bkup data source
  
  
  # Scatterplot
  gg <- ggplot(new_file, aes(x=Date, y=n)) + 
    geom_point(aes(col=Date, size=n)) + 
    geom_smooth(method="loess", se=F) + 
    xlim(c(2012, 2018)) + 
    ylim(c(0, 5)) + 
    labs(subtitle="Rating by Year", 
         y="Rating", 
         x="Year", 
         title="Scatterplot", 
         caption = "Hurt Locker Joyful User Reviews")
  
  plot(gg)
  
  library(ggplot2)
  theme_set(theme_classic())
  
  ##
  
  new_file <-  prince_nrc_sub %>%
    count(Date, sentiment)
  
  new_file
  
  
  p <-ggplot(new_file, aes(Date, n))
  p +geom_bar(stat = "identity", aes(fill = sentiment))
  
  # Histogram on a Continuous (Numeric) Variable
  
  
  
  
  p <-ggplot(new_file, aes(Date, n))
  p +geom_bar(stat = "identity", aes(fill = sentiment), position = "dodge")
  
  
  p <-ggplot(new_file, aes(Date, n))
  p +geom_bar(stat = "identity", aes(fill = sentiment)) +
    xlab("Date") + ylab("Sentiment Frequency") +
    ggtitle("American Sniper User Review Affects over Time") +
    theme_bw()
  
  g <- ggplot(new_file, aes(Date, n)) + scale_fill_brewer(palette = "Spectral")
  
  
  g + geom_histogram(stat="identity", aes(fill=sentiment)) + 
    labs(title="Histogram with Hurt Locker User Reviews", 
         subtitle="Joyful Sentiments")  
  
  g + geom_histogram(aes(fill=n), 
                     bins=5, 
                     col="black", 
                     size=.1) +   # change number of bins
    labs(title="Histogram with Fixed Bins", 
         subtitle="Engine Displacement across Vehicle Classes") 
  
  

# Gutenberg Sentiment Analysis --------------------------------------------





### This all from main Tidy R package - it works, but I'm having trouble translating to a different text format input

install.packages("gutenbergr")
library(dplyr)
library(gutenbergr)


##hgwells <- gutenberg_download(c(35, 36, 5230, 159))
## Lots on how to compare corpora

gutenberg_subjects %>%
  filter(subject == "Detective and mystery stories")

lawrence <- gutenberg_download("gutenberg_subjects")

aristotle_books <- gutenberg_works(author == "Aristotle") %>%
  gutenberg_download(meta_fields = "title")

aristotle_books



library(dplyr)
library(stringr)
gutenberg_subjects %>%
  filter(subject_type == "lcsh") %>%
  count(subject, sort = TRUE)
scifi_subjects <- gutenberg_subjects %>%
  filter(str_detect(subject, "Science Fiction")) %>%
  semi_join(scifi_subjects, by = "gutenberg_id")
scifi_subjects
##scifi_metadata <- gutenberg_works() %>%
##filter(author == "Doyle, Arthur Conan") %>%
# semi_join(scifi_subjects, by = "gutenberg_id")
#sherlock_holmes_metadata
## Not run:
scifi_books <- gutenberg_download(scifi_subjects$gutenberg_id)
scifi_books




library(tidytext)

sentiments
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

library(tidyverse)

tbl <- list.files(pattern = "LawrenceAll.txt") %>% 
  map_chr(~ read_file(.)) %>% 
  data_frame(text = .)

## library(janeaustenr)
library(dplyr)
library(stringr)

tidy_books <- aristotle_books %>%
 ## group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]", 
                                                 ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

nrc_joy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

tidy_books %>%
  filter(title == "THE POETICS OF ARISTOTLE") %>%
  inner_join(nrc_joy) %>%
  count(word, sort = TRUE)



library(tidyr)

jane_austen_sentiment <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(title, index = linenumber %/% 80, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)


library(ggplot2)

ggplot(jane_austen_sentiment, aes(index, sentiment, fill = title)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~title, ncol = 2, scales = "free_x")













library(janeaustenr)
library(dplyr)
library(stringr)

original_books <- austen_books() %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) %>%
  ungroup()

original_books


library(tidytext)
tidy_books <- original_books %>%
  unnest_tokens(word, text)

tidy_books

data(stop_words)

tidy_books <- tidy_books %>%
  anti_join(stop_words)

tidy_books %>%
  count(word, sort = TRUE) 

library(ggplot2)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 600) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()
















