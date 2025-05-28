from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from login.views import *
from statsmodels.tsa.arima.model import ARIMA
from django.shortcuts import render

#========== IMPORT LIBRARIES ==========
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
# Turn off pandas warning
pd.set_option('mode.chained_assignment', None)

# For Getting Dataset From YahooFinance
import yfinance as yf



# Get today's date
from datetime import datetime, timedelta
import pytz
today = datetime.now(tz=pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')

# For Graphs
import plotly.offline as opy
from plotly.graph_objs import Scatter
import plotly.graph_objects as go

#========== READ DATA ==========
Df = yf.download('GC=F', '2008-01-01', today, auto_adjust=True)
# Only keep close columns
Df = Df[['Close']]
# Drop rows with missing values
Df = Df.dropna()

print(Df.head())

# Plot the closing price of GLD
x_data = Df.index
y_data = Df['Close']
ClosingPricePlot_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data, mode='lines', name='test', opacity=0.8, marker_color='green')],
    'layout': {'title': 'Gold ETF Price Series', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
}, output_type='div')

#========== DEFINE EXPLANATORY VARIABLES ==========
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
y = Df['next_day_price']

#========== TRAIN AND TEST DATASET ==========
t = 0.8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]

#========== LINEAR REGRESSION MODEL ==========
linear = LinearRegression().fit(X_train, y_train)
RegressionModelFormula = "Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) + %.2f * 9 Days Moving Average (x2) + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_)

#========== PREDICTING GOLD ETF PRICES ==========
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
# Attach y_tese series to dataframe
predicted_price['close'] = y_test
# Plot graph
x_data = predicted_price.index
y_data_predicted = predicted_price['price']
y_data_actual = predicted_price['close']
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_data, y=y_data_predicted,
                    mode='lines',
                    name='Predicted Price'))
fig.add_trace(go.Scatter(x=x_data, y=y_data_actual,
                    mode='lines',
                    name='Actual Price'))
PredictionPlot_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data_predicted, mode='lines', name='Predicted Price', opacity=0.8),
    Scatter(x=x_data, y=y_data_actual, mode='lines', name='Actual Price', opacity=0.8)],
    'layout': {'title': 'Predicted VS Actual Price', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
}, auto_open=False, output_type='div')

#========== CUMULATIVE RETURNS ==========
gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price['price']
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)
    
gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)
    
gold['strategy_returns'] = gold.signal * gold['gold_returns']
x_data = gold.index
y_data = ((gold['strategy_returns']+1).cumprod()).values
CumulativeReturns_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data, mode='lines', name='test', opacity=0.8, marker_color='green')],
    'layout': {'title': 'Cumulative Returns', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Cumulative Returns (X 100%)'}}
}, output_type='div')






#========== PREDICT DAILY MOVES ==========
data = yf.download('GC=F', '2008-06-01', today, auto_adjust=True)
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
data['predicted_gold_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(1) < data.predicted_gold_price,"Buy","No Position")


# Return to Context functions
def PlotClosingPrice():
    return ClosingPricePlot_div

def RegressionModel():
    return RegressionModelFormula

def PredictionPlot():
    return PredictionPlot_div

def r2_scoreCalculate():
    # R square
    r2_score = linear.score(X[t:], y[t:])*100
    r2_score = float("{0:.2f}".format(r2_score))
    return r2_score

def CumulativeReturns():
    return CumulativeReturns_div

def SharpeRatioCalculate():
    return '%.2f' % (gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5))

USD_TO_INR_RATE = 83.5  # 1 USD = 83.5 INR

USD_TO_INR_RATE = 83.5  # (example, use current rate)
TROY_OUNCE_TO_GRAM = 28.3495

def MovingAverage_S3():
    usd_per_ounce = data['S_3'].iloc[-1]
    usd_per_gram = usd_per_ounce / TROY_OUNCE_TO_GRAM
    inr_per_gram = usd_per_gram * USD_TO_INR_RATE
    inr_for_10_grams = inr_per_gram * 10
    return round(inr_for_10_grams, 2)

def MovingAverage_S9():
    usd_per_ounce = data['S_9'].iloc[-1]
    usd_per_gram = usd_per_ounce / TROY_OUNCE_TO_GRAM
    inr_per_gram = usd_per_gram * USD_TO_INR_RATE
    inr_for_10_grams = inr_per_gram * 10
    return round(inr_for_10_grams, 2)

def GetSignal():
    return data['signal'].iloc[-1]

def GetPredictedPrice():
    return round(data['predicted_gold_price'].iloc[-1], 2)

def GetClosingPrice():
    return round(data["Close"].iloc[-1], 2)

def GetClosingPriceDate():
    return data.index[-1].strftime("%d/%m/%y")

#Ended
# @login_required
def home(request):
    #Added

    context = {
        'ClosingPricePlot_div' : PlotClosingPrice(),
        'PredictionPlot_div' : PredictionPlot(),
        'CumulativeReturns_div' : CumulativeReturns(),
        'SharpeRatio' : SharpeRatioCalculate(),
        'S_3' : MovingAverage_S3(),
        'S_9' : MovingAverage_S9(),
        'Signal' : GetSignal(),
        'PredictedPrice' : GetPredictedPrice(),
        'ClosingPrice' : GetClosingPrice(),
        'ClosingDate' : GetClosingPriceDate(),
    }

    #Ended
    return render(request, 'GoldPricePrediction/home.html', context)

# def base(request):
#     return render(request, 'GoldPricePrediction/base.html')

def information(request):
    context = {
        'RegressionModelFormula' : 'Gold ETF Price (y) = 1.17 * 3 Days Moving Average (x1) + -0.18 * 9 Days Moving Average (x2) + 0.34 (constant)' , 'r2_score' : 97.52 ,
    }
    return render(request, 'GoldPricePrediction/information.html', context)



def plots_view(request):
    context = {
        'ClosingPricePlot_div' : PlotClosingPrice(),
        'PredictionPlot_div' : PredictionPlot(),
        'CumulativeReturns_div' : CumulativeReturns(),
        'SharpeRatio' : SharpeRatioCalculate(),
        'S_3' : MovingAverage_S3(),
        'S_9' : MovingAverage_S9(),
        'Signal' : GetSignal(),
        'PredictedPrice' : GetPredictedPrice(),
        'ClosingPrice' : GetClosingPrice(),
        'ClosingDate' : GetClosingPriceDate(),
    }

    # Your logic to generate content for the 'plots' page
    # For example, rendering a template named 'plots.html'
    return render(request, 'GoldPricePrediction/plots.html',context )



import requests
from django.shortcuts import render, redirect

# The main function to fetch and convert the gold price to INR
def gold_price(request):
    api_key = 'goldapi-4fv4sm9xskcnp-io'  # Replace with your actual API key 
    
    gold_api_url = 'https://www.goldapi.io/api/XAU/INR'  # API to fetch gold price in USD
    headers = {'x-access-token': api_key}

    try:
        response = requests.get(gold_api_url, headers=headers)
        response.raise_for_status()  # Raise error for unsuccessful status codes

        if response.status_code == 200:
            gold_data = response.json()
            gold_price = gold_data['price']  # Gold price in USD

            # Convert USD price to INR
            gold_price_in_inr = gold_price // 2.83495

            # Prepare the context with the converted INR price
            context = {'gold_price': gold_price_in_inr}
            return render(request, 'GoldPricePrediction/gold_price.html', context)

    except requests.exceptions.RequestException as e:
        # Log the error and print a message
        print(f"API Error: {e}")
    
    # In case of error or failure, redirect to home page or display an error message
    return redirect('home')








from django.shortcuts import render
from django.http import HttpResponse

# Function to set a cookie
def set_cookie(request):
    response = render(request, 'GoldPricePrediction/home.html')
    # Set a cookie named 'gold_prediction' with a value
    response.set_cookie('gold_prediction', 'predicted_value')
    return response


from django.shortcuts import render
from django.http import HttpResponse

# Function to get a cookie
def get_cookie(request):
    # Get the value of the 'gold_prediction' cookie
    predicted_value = request.COOKIES.get('gold_prediction')
    return HttpResponse(f'Predicted value from cookie: {predicted_value}')

# Include this function in your views where needed








from django.http import HttpResponse

def set_user_timezone(request):
    # Get user's preferred timezone (for example, let's assume 'US/Eastern')
    user_timezone = 'US/Eastern'  # You might fetch this from user settings

    response = render(request, 'GoldPricePrediction/home.html')

    # Set 'user_timezone' cookie with the user's preferred timezone
    response.set_cookie('user_timezone', user_timezone)

    return response

def get_user_timezone(request):
    # Retrieve 'user_timezone' cookie value
    user_timezone = request.COOKIES.get('user_timezone')

    return HttpResponse(f"User's Timezone: {user_timezone}")


def info_page(request):
    # Add any context data or logic needed to render Info.html
    return render(request, 'GoldPricePrediction/info.html')


def info_page(request):
    # Fetch gold price data from your database or an API
    # Example data - modify this with your actual data retrieval
    gold_prices = [
        {'date': '2023-12-01', 'price': 1500},
        # ... (Fetch data for other dates)
    ]

    # Send data to the template
    context = {
        'gold_prices': gold_prices,
        # Add other context data needed for your page
    }
    return render(request, 'GoldPricePrediction/info.html', context)


# =======================================


# Function to get gold price data for the graph and table
def gold_price_view(request):
    # Retrieve data from Yahoo Finance from December 1st, 2023, to current date
    gold_data = yf.download('GOLD', start='2023-12-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    
    # Process the data to format for the graph
    gold_prices = [{'date': str(index.date()), 'price': price} for index, price in zip(gold_data.index, gold_data['Close'])]

    # Retrieve data for the past 15 days for the table
    past_15_days = gold_data.tail(15)
    past_15_days_data = [{'date': str(index.date()), 'price': price} for index, price in zip(past_15_days.index, past_15_days['Close'])]

    return render(request, 'GoldPricePrediction/info.html', {'gold_prices': gold_prices, 'past_15_days': past_15_days_data})


from datetime import datetime, timedelta
import yfinance as yf

def info_page(request):
    # Fetch gold price data from Yahoo Finance for the past 15 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
    
    gold_data = yf.download('GLD', start=start_date, end=end_date, progress=False)
    gold_prices_past_15_days = gold_data['Close'].reset_index().tail(15).values.tolist()

    # Fetch gold price data from Yahoo Finance from 2023-12-01 to the current date for the graph
    gold_data_full = yf.download('GLD', start='2023-12-01', end=end_date, progress=False)
    gold_prices_full = gold_data_full['Close'].reset_index().values.tolist()

    context = {
        'gold_prices_past_15_days': gold_prices_past_15_days,
        'gold_prices_full': gold_prices_full,
    }
    return render(request, 'GoldPricePrediction/info.html', context)









# def login_page(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             auth_login(request, user)
#             return redirect('home')  # Redirect to home page after successful login
#     else:
#         form = AuthenticationForm()
#     return render(request, 'login.html', {'form': form})




# def login_page(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         User.objects.create(username=username, password=password)
#         # Here, you might want to add authentication logic
#         # For simplicity, this example just saves the user directly to the database
#         return redirect('home')  # Redirect to the home page after login

#     return render(request, 'login.html')
import random
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Function to generate a random date
def generate_random_date(start_year=2015, end_year=2025):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date

# Fetch gold price data (handling future dates)
def fetch_gold_data():
    ticker = yf.Ticker('GLD')
    data = ticker.history(period="10y")  # Fetch 5 years of data
    return data

# Train the Random Forest model
def train_predictive_model(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal values

    # Features and target variable
    X = data[['Date']]  # Independent variable (date)
    y = data['Close']   # Dependent variable (gold price)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from django.shortcuts import render

# Load data from CSV file
def load_data_from_csv():
    # Replace 'gold_prices.csv' with the actual path to your CSV file
    df = pd.read_csv('./gold3.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day  # Extract the day part
    return df

# Prepare the data (load from CSV)
df = load_data_from_csv()

# Prepare features (Day, Month, and Year) and target (Price)
X = df[['Year', 'Month', 'Day']]  # Features: Year, Month, Day
y = df['Price']  # Target: Gold Price

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# View function to predict gold price
def predict_gold_price(request):
    predicted_price = None
    year = None
    month = None
    day = None

    if request.method == 'POST':
        # Get the year, month, and day from the form
        year = int(request.POST.get('year'))
        month = int(request.POST.get('month'))
        day = int(request.POST.get('day'))

        # Use the trained model to predict the price for the entered date
        prediction = model.predict([[year, month, day]])  # Including the day in the prediction
        predicted_price = prediction[0]

    return render(request, 'GoldPricePrediction/price.html', {
        'predicted_price': predicted_price,
        'year': year,
        'month': month,
        'day': day
    })
