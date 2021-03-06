<h1>Overview</h1>

This project is about solving a data science project using the CRISP-DM framework. CRISP-DM stands for Cross Industry Standard Process for Data Mining and is a widely accepted framework for data exploration and analysis in the data science community.

The overall objective is to analyze Airbnb Bostons listings data and find answers to the questions below :
  1. How do listing prices change across locations and time ?
  2. What are the main drivers of listing price ?
  3. Which features of a listings influence customer satisfaction the most ?

This repo contains data and jupyter notebook for all analysis done for this data science blog [This is how you can analyze any dataset with the CRISP-DM framework](https://medium.com/@gangulym23/this-is-how-you-can-analyze-any-dataset-with-the-crisp-dm-framework-cc9353f4dabe) on Medium.

<h2>Data Description</h2>

These are 3 datasets from Airbnb Boston that have been analyzed :
1. listings.csv — details on listing features like location, amenities and reviews
2. calendar.csv — availability calendar of listings over a one year period
3. reviews.csv — reviews posted by customers after their stay
These dataset can be found [here](http://insideairbnb.com/get-the-data.html) as well inside the data folder of this repo.

<h2>Libraries</h2>

* pandas
* numpy
* seaborn
* matplotlib
* xgboost
* sklearn

<h2> Main steps of Analysis </h2>

* Convert columns to appropriate data type
* Missing value imputation
* Categoical feature encoding
* Analyze univariate and bivariate plots to understand feature distributions and relationships
* Build machine learning model that predicts price of a listing based on listing features like location, amenities and review scores
* Build machine learning model that predicts review rating score of a listing based on listing features like location, amenities and price


All these details can be found in this [notebook](https://github.com/maddyg/Data-Scientist-Nanodegree/blob/master/Project1-Write_a_Data_Science_Blog/Codes/Analyzing%20Airbnb%20Listings%20in%20Boston.ipynb)


<h2>Key findings</h2>

<h3>How do listing prices change across locations and time ?</h3>

* Prices vary across locations, suburbs like Mattapan and Hyde Park are economical while neighbourhoods like Chinatown and Leather District that close to Downtown Boston are more expensive.
* November to March is the off-season for tourism and business picks up and prices increase April onwards with July to octover being the peak season. 

<h3> What are the main drivers of listing price ?</h3>

* Main drivers of price are room type, cleaning fee, no. of bedrooms, no. of beds, no. of bathrooms and location.
* Amenities like TV, cable, heating, internet, pets allowed on property and elevator influence listing price.

<h3> Which features of a listings influence customer satisfaction the most ?</h3>
  
* Cleanliness, value for money, listing description online matches real listing, communication with host, check-in experience are the top factors that influence customer satisfaction.
* Amenities that seem to influence customer satisfaction are Internet, Washer, Dryer, Cable, Gym and Elevator

These are explained with vizualizations in the [blog](https://medium.com/@gangulym23/this-is-how-you-can-analyze-any-dataset-with-the-crisp-dm-framework-cc9353f4dabe).




