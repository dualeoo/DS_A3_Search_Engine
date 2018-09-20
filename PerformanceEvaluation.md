#Performance Evaluation
1. successfully retrieve sentences that includes the word in it (100%)
2. still return result even though the word doesn't appear in the database (yoda)
3. search for a string => even if the sentence doesnt contain the whole string, 
it stills return a value that contains part of the string ("this is war")
4. some cases like "what the hell" return "what are you stupid?" as highest similarity and 
"^^^^^^ what the hell ios a 'wovie' ?? (wovy (sp))??" as 2nd highest

### Conclusion
- Let w be a word from the query. The higher the frequency of the w in a document, 
the higher the final score of the document
- The final score doesn't depend on the total occurrence of a word on the total number of words in the sentence in
