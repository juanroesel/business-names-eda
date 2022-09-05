# Business Names EDA Summary

## Objective
What can we learn about business names? How can we use these insights? For example: Are there structural differences in business names correlated with company size? Can
I figure out what type of company it is? etc.

## Key insights

* Business naming tends to include patterns such as words associated with business structure (e.g., pvt. ltd, group, corporation) words that reference the type of business it is (e.g., spa, photography, agency), or words that conform to a particular lexical structure (e.g., X and Y, X & Y).

* About 3 millions businesses have their names registered in English.

* London is the city mostly associated with business names with over 68k records, followed by New York (41k), Paris (26k), Toronto (20k) and Sao Paulo (19k).

* Taking London business names as our sample, we found that the average words of a given business name tends to decrease over time.

* Using cosine similarity with BERT vectors, we found that one-word business names tend to be closer to each other in the semantic space than business names with more words, regardless of language.
