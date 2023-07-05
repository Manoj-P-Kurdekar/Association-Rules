#######################################Problem1#######################################
# Kitabi Duniya , a famous book store in India, which was established before Independence, 
# the growth of the company was incremental year by year, but due to online selling of books
# and wide spread Internet access its annual growth started to collapse, seeing sharp 
# downfalls, you as a Data Scientist help this heritage book store gain its popularity 
# back and increase footfall of customers and provide ways the business can improve 
# exponentially, apply Association Rule Algorithm, explain the rules, and visualize 
# the graphs for clear understanding of solution.
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

books = pd.read_csv(r"C:\Users\manoj\OneDrive\Desktop\360DigiTMG\Data Science\Assignment Question\Associstion Rules\Association_Rules-Assignment_Datasets\book.csv")
print(type(books))
books

# Itemsets
a_books = apriori(books, min_support = 0.075, max_len = 4, use_colnames = True)
a_books

# Most Frequent item sets based on support 
a_books.sort_values('support', ascending = False, inplace = True)
a_books

# Association Rules
rules = association_rules(a_books, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

new_rules = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
new_rules
new_rules = new_rules.apply(sorted)
new_rules
rules_sets = list(new_rules)
rules_sets

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    index_rules
    
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy

# Sorting them with respect to list and getting top 10 rules 
rules10 = rules_no_redudancy.sort_values('lift', ascending = False).head(10)
rules10 

##############################Problem 2#########################################
# A film distribution company wants to target audience based on their likes and 
# dislikes, you as a Chief Data Scientist Analyze the data and come up with different 
# rules of movie list so that the business objective is achieved.
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

my_movies = pd.read_csv(r"C:\Users\manoj\OneDrive\Desktop\360DigiTMG\Data Science\Assignment Question\Associstion Rules\Association_Rules-Assignment_Datasets\my_movies.csv")
my_movies = my_movies.iloc[:,5:]
print(type(my_movies))
my_movies

# Itemsets
a_movies = apriori(my_movies, min_support = 0.002, max_len = 4, use_colnames = True)
a_movies

# Most Frequent item sets based on support 
a_movies.sort_values('support', ascending = False, inplace = True)
a_movies

# Association Rules
rules = association_rules(a_movies, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

new_rules = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
new_rules
new_rules = new_rules.apply(sorted)
new_rules
rules_sets = list(new_rules)
rules_sets

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    index_rules
    
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy

# Sorting them with respect to list and getting top 10 rules 
rules10 = rules_no_redudancy.sort_values('lift', ascending = False).head(10)
rules10 

##########################################Problem 3###############################
# A Mobile Phone manufacturing company wants to launch its three brand new phone into 
# the market, but before going with its traditional marketing approach this time it want 
# to analyze the data of its previous model sales in different regions and you have been 
# hired as an Data Scientist to help them out, use the Association rules concept and provide
# your insights to the company’s marketing team to improve its sales.

import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

my_phone = pd.read_csv(r"C:\Users\manoj\OneDrive\Desktop\360DigiTMG\Data Science\Assignment Question\Associstion Rules\Association_Rules-Assignment_Datasets\myphonedata.csv")
my_phone = my_phone.iloc[:,3:]
print(type(my_phone))
my_phone

# Itemsets
a_phone = apriori(my_phone , min_support = 0.02 , max_len = 4 , use_colnames = True)
a_phone

# Most Frequent item sets based on support 
a_phone.sort_values('support', ascending = False, inplace = True)
a_phone

# Association Rules
rules = association_rules(a_phone, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

new_rules = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
new_rules
new_rules = new_rules.apply(sorted)
new_rules
rules_sets = list(new_rules)
rules_sets

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    index_rules
    
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy

# Sorting them with respect to list and getting top 10 rules 
rules10 = rules_no_redudancy.sort_values('lift', ascending = False).head(10)
rules10 

##########################Problem 4######################################################
# A retail store in India, has its transaction data, and it would like to know the 
# buying pattern of the consumers in its locality, you have been assigned this task 
# to provide the manager with rules on how the placement of products needs to be there 
# in shelves so that it can improve the buying patterns of consumes and increase customer 
# footfall.

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules,apriori
import warnings
warnings.filterwarnings("ignore")

retails = []
with open(r"C:\Users\manoj\OneDrive\Desktop\360DigiTMG\Data Science\Assignment Question\Associstion Rules\Association_Rules-Assignment_Datasets\transactions_retail.csv") as f:
    retails = f.read()
    
retails = retails.split("\n")

retails = retails[:100]

retails_list = []
for i in retails:
    retails_list.append(i.split(","))
print(retails_list)

all_retails_list = [i for item in retails_list for i in item]

new_all_retails_list =[]
for i in all_retails_list:
    if i != "NA":
        new_all_retails_list.append(i)
        
from collections import Counter # ,OrderedDict 

item_frequencies = Counter(new_all_retails_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])
item_frequencies

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# Creating Data Frame for the transactions data
retail_series = pd.DataFrame(pd.Series(retails_list))
retail_series

retail_series.columns = ["transactions"]
retail_series

X = retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')
X
X = X.drop(['NA'] , axis =1)
X
# Itemsets
frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets

# Association Rules
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

new_rules = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
new_rules
new_rules = new_rules.apply(sorted)
new_rules
rules_sets = list(new_rules)
rules_sets
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    index_rules
    
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
rules_no_redudancy 

################################END#####################################################