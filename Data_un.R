# Name: Tosin Williams
# Topic: Data understanding

#First lets view the dataset iris
view(iris)

#Display the summary statistic

head(iris,4) #First 4 rows of the dataset

tail(iris,4) #Last 4 rows of the dataset

summary(iris) #It displays a summary of the dataset

summary(iris$Sepal.Length) #It displays a summary of the sepal.length row

sum(is.na(iris)) #It checks if there is any missing data

install.packages("skimr") #Install the skim package
library(skimr)
skim(iris) #This shows a more detailed summary about iris

#Lets use the skim command and group according to species
iris %>%
  dplyr::group_by(Species) %>% 
  skim()

