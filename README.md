# cpi-csi

## todo

- create option to append results to csv file containing past results in the same format as the df that is used to train the model. Using the -o flag. Will try to read the path given by -o if it exists, append result to it, if it doesn't create new file and write the whole df to it with the added column of predicted results.
- create script to test the models accuracy using past data to train it and try to predict the past year's monthly CPIs based on the CSIs. Could be a file called test.py or something.
- use matplot lib to create script to create graphs for the website.
    - last year graph
    - last 5 or 10 years
    - all of the data
