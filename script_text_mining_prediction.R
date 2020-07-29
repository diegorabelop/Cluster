library(readxl)
#load("sessions/TEXT_MININ_16-01.RData")

random_data <- read_excel("data/data_no_duplicate.xlsx")

# Dataset a ser predito
# Dataset rotulado
# Existem artigos do dataset rotulado que estao no dataset a ser predito?



data1 <- random_data[, c("Title", "Abstract")]

#coombining Abstract and Title
data1$Text <- paste(data1$Title, data1$Abstract)

data <- data1[, c("Text")]

# Preparing the data
data <- as.data.frame(data)

data$Text = as.character(data$Text)

head(data$Text)
str(data)

# Text mining packages 
# install.packages('tm', dependencies=TRUE, repos='http://cran.rstudio.com/')
library(tm)
library(SnowballC)

# Step 1 - Create a corpus text
corpus = Corpus(VectorSource(data$Text))
corpus[[1]][1]
data$Final_decision[1]
data$Text[1]

## Step 2 - Conversion to Lowercase
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[1]][1]

#Step 3 - Removing Punctuation
corpus = tm_map(corpus, removePunctuation)
corpus[[1]][1]

#Step 4 - Removing Stopwords and other words
corpus = tm_map(corpus, removeWords, c("objective","background", "introduction", "purpose", "method", "result", "conclusion", "limitation", 
                                       stopwords("english")))

corpus[[1]][1]  

# Step 5 - Stemming: reducing the number of inflectional forms of words
corpus = tm_map(corpus, stemDocument)
corpus[[1]][1]  

# Step 6 - Create Document Term Matrix
frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995) #remove sparse terms
tSparse = as.data.frame(as.matrix(sparse)) #convert into data frame

colnames(tSparse) = make.names(colnames(tSparse)) #all the variable names R-friendly
tSparse$final_decision = data$Final_decision #add the outcome

saveRDS(tSparse, file = "cache/data_no_duplicated_preprocessed.rds")

# Importar o modelo ----
columns_training <- colnames(rose_outside$trainingData)[-1]
columns_training

library(purrr)
percentage <- map_dbl(rose_outside$trainingData, function(x){(length(x) - table(x)[1]) / length(x)})
percentage <- percentage[-1]
percentage <- percentage * 100

frequency <- map_dbl(rose_outside$trainingData, function(x){(length(x) - table(x)[1])})
frequency <- frequency[-1]

m <- match(columns_training, colnames(tSparse))

df <- data.frame(columns_training, m, percentage, frequency)
df_notfound <- df[is.na(df$m), ]
df_notfound

m <- m[!is.na(m)]

new_data <- tSparse[, m]

data.frame(colnames(new_data), columns_training)


