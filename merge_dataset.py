<<<<<<< HEAD
import pandas as pd

# Load datasets
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")
users = pd.read_csv("data/users.csv")

# Merge flights + users
flight_data = pd.merge(flights, users, left_on="userCode", right_on="code")

# Merge hotels + users
hotel_data = pd.merge(hotels, users, left_on="userCode", right_on="code")

print("Flight Data Shape:", flight_data.shape)
=======
import pandas as pd

# Load datasets
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")
users = pd.read_csv("data/users.csv")

# Merge flights + users
flight_data = pd.merge(flights, users, left_on="userCode", right_on="code")

# Merge hotels + users
hotel_data = pd.merge(hotels, users, left_on="userCode", right_on="code")

print("Flight Data Shape:", flight_data.shape)
>>>>>>> c1622b5dc05837c452c7f26cd55fbdd3f18f79fc
print("Hotel Data Shape:", hotel_data.shape)