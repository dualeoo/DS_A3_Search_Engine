# Goal
The goal of application is to allow users to find articles contains the _query_ they are interested in.
To achieve this goal, we would treat the _query_ as a document and find documents similar to the _query_.   

#### So how to evaluate the similarity of any two documents?
To compare the similarity of any two documents, there are several ways. The way we use here is to convert 
each document to a vector and compare the similarity of the two vectors using cosine similarity.

#### How to convert a document to a vector?
There are several ways to convert a document to a vector. 
Regardless of the method we use, the process is called **document modelling**.
For this project, we model the document using three methods: bag of words (doc2bow)+ tfidf + lsi

#### Why tfidf?
Our common sense telling us that if a word appears many times in the document, that word is significant to that document.
This follows that the word '*the*' is a significant word to a document because that word appears many times in 
virtually almost every document.
Obviously, the word '*the*' is not a significant word because it appears in **many** documents.
To take account this fact, we re-define the term *significance* this way:
A significant word is a word appears many times in the document *but* few times in other documents.
Tfidf is based on this definition of *significance*. 

#### Why LSI?
Our corpus contains about 190,000 different words.
After modelling each document in our corpus using doc2bow and tfidf, each document is represent by a vector 
of size 190,000.
Given that we have 18,000 documents in the corpus, we need a matrix of size (18,000, 190,000). 
Assuming each cell takes 8 bytes then we need at least 18K * 190K * 8B = 27360 MB to store it @@
That's not to say, when I need to find a document most similar to the query, I would need to do 18K * 190K computations
which take a **HUGE** amount of time.
We use two approaches to solve this problem:
1. For those words whose corpus frequency is less than 5 or more than 9,000 (50% of corpus size), we remove it. 
In particular, there are 168062 words satisfied that criteria. So now the matrix size becomes (18,000, 20,000)
2. 20,000 features is still a lot. We then use LSI to further reduces it to (18,000, 10,000) (dimensionality reduction). 

So now, our comparision is more tractable, we only need to make 18K * 10K computations. 