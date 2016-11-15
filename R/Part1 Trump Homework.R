library(readr)
library(stringr)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(reshape)
#import data
raw <- read.csv("~/Desktop/Third Semester/BA/11.8 Homework/trump and clinton/tweets.csv", comment.char="#",
                   header = TRUE,
                   stringsAsFactors = FALSE)
names(raw)
head(raw)
summary(raw)

#Subset Id, handle, and text
tweets_sub <- subset(raw,select=id:text)

###############
#### Trump#####
###############
Trump <- subset(tweets_sub,handle== "realDonaldTrump")
text_Trump <-Trump$text


#pre-processing
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words = c('am','-','a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always','among','an','and','another','any','anybody','anyone','anything','anywhere','are','area','areas','around','as','ask','asked','asking','asks','at','away','b','back','backed','backing','backs','be','became','because','become','becomes','been','before','began','behind','being','beings','best','better','between','big','both','but','by','c','came','can','cannot','case','cases','certain','certainly','clear','clearly','come','could','d','did','differ','different','differently','do','does','done','down','down','downed','downing','downs','during','e','each','early','either','end','ended','ending','ends','enough','even','evenly','ever','every','everybody','everyone','everything','everywhere','f','face','faces','fact','facts','far','felt','few','find','finds','first','for','four','from','full','fully','further','furthered','furthering','furthers','g','gave','general','generally','get','gets','give','given','gives','go','going','good','goods','got','great','greater','greatest','group','grouped','grouping','groups','h','had','has','have','having','he','her','here','herself','high','high','high','higher','highest','him','himself','his','how','however','i','if','important','in','interest','interested','interesting','interests','into','is','it','its','itself','j','just','k','keep','keeps','kind','knew','know','known','knows','l','large','largely','last','later','latest','least','less','let','lets','like','likely','long','longer','longest','m','made','make','making','man','many','may','me','member','members','men','might','more','most','mostly','mr','mrs','much','must','my','myself','n','necessary','need','needed','needing','needs','never','new','new','newer','newest','next','no','nobody','non','noone','not','nothing','now','nowhere','number','numbers','o','of','off','often','old','older','oldest','on','once','one','only','open','opened','opening','opens','or','order','ordered','ordering','orders',
               'other','others','our','out','over','p','part','parted','parting','parts','per','perhaps','place','places','point','pointed','pointing','points','possible','present','presented','presenting','presents','problem','problems','put','puts','q','quite','r','rather','really','right','right','room','rooms','s','said','same','saw','say','says','second','seconds','see','seem','seemed','seeming','seems','sees','several','shall','she','should','show','showed','showing','shows','side','sides','since','small','smaller','smallest','so','some','somebody','someone','something','somewhere','state','states','still','still','such','sure','t','take','taken','than','that','the','their','them','then','there','therefore','these','they','thing','things','think','thinks','this','those','though','thought','thoughts','three','through','thus','to','today','together','too','took','toward','turn','turned','turning','turns','two','u','under','until','up','upon','us','use','used','uses','v','very','w','want','wanted','wanting','wants','was','way','ways','we','well','wells','went','were','what','when','where','whether','which','while','who','whole','whose','why','will','with','within','without','work','worked','working','works','would','x','y','year','years','yet','you','young','younger','youngest','your','yours','z')
stop_words <- tolower(stop_words)

cleantext<- gsub("'", "", text_Trump) # remove apostrophes
cleantext <- gsub("[[:punct:]]", " ", text_Trump)  # replace punctuation with space 
cleantext <- gsub("[[:cntrl:]]", " ", text_Trump)  # replace control characters with space
cleantext <- gsub("^[[:space:]]+", "", text_Trump) # remove whitespace at beginning of documents
cleantext <- gsub("[[:space:]]+$", "", text_Trump) # remove whitespace at end of documents
cleantext <- gsub("[^a-zA-Z -]", " ", text_Trump) # allows only letters
cleantext <- tolower(text_Trump)  # force to lowercase

## get rid of blank docs
cleantext <- cleantext[cleantext != ""]  # remove ""

# tokenize on space and output as a list:
doc.list <- strsplit(cleantext, "[[:space:]]+")  # break into single word

# compute the table of terms:
term_Trump.table <- table(unlist(doc.list))
term_Trump.table <- sort(term_Trump.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term_Trump.table) %in% stop_words | term_Trump.table < 5
term_Trump.table <- term_Trump.table[!del]
term_Trump.table <- term_Trump.table[names(term_Trump.table) != ""]
vocab <- names(term_Trump.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) { 
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)
head(documents)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term_Trump.table) 

# MCMC and model tuning parameters:
K <- 10  # number of topic
G <- 3000  # run 3000 times
alpha <- 0.02  # math
eta <- 0.02 

# Fit the model:
library(lda)
set.seed(357)   # number of  virtualization 
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # to show how many mins

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))   
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x))) 

news_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
library(LDAvis)
library(servr)  #run a vusiliation, file html java-vsuiliation json-data
# create the JSON object to feed the visualization:
json <- createJSON(phi = news_for_LDA$phi, 
                   theta = news_for_LDA$theta, 
                   doc.length = news_for_LDA$doc.length, 
                   vocab = news_for_LDA$vocab, 
                   term.frequency = news_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE) #json=data,out.dir='vis', out.dir

