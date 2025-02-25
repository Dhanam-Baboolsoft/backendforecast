from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from keras.layers import LSTM, Dense
from keras.models import Sequential
from pandas import read_csv
from pmdarima.arima import ADFTest, auto_arima
from rest_framework import permissions, views, viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from .models import Client, Employee, Sales
from .serializers import ClientSerializer, EmployeeSerializer, SalesSerializer


class ClientView(viewsets.ModelViewSet):
    queryset = Client.objects.all()
    serializer_class = ClientSerializer


class EmployeeView(viewsets.ModelViewSet):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer


class SalesView(viewsets.ModelViewSet):
    queryset = Sales.objects.all()
    serializer_class = SalesSerializer
    allowed_methods = ['post']


# class FileUploadView(views.APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request, format=None):  # Removed 'filename' parameter
#         if 'file' not in request.data:
#             return Response({"error": "No file provided"}, status=400)

#         file_obj = request.data['file']
#         sales = read_csv(file_obj)

#         # Convert 'Month' column to datetime and set it as index
#         sales['Month'] = pd.to_datetime(sales['Month'], errors='coerce')
#         sales.set_index('Month', inplace=True)

#         # Check stationarity
#         adf_test = ADFTest(alpha=0.05)
#         adf_test.should_diff(sales)

#         # Train-test split
#         train, test = train_test_split(sales, test_size=0.2, shuffle=False)

#         # Auto ARIMA model
#         arima_model = auto_arima(train, start_p=0, d=1, start_q=0,
#                                  max_p=5, max_d=5, max_q=5, start_P=0,
#                                  D=1, start_Q=0, max_P=5, max_D=5,
#                                  max_Q=5, m=12, seasonal=True,
#                                  error_action='warn', trace=True,
#                                  suppress_warnings=True, stepwise=True,
#                                  random_state=20, n_fits=50)

#         # ARIMA Prediction
#         prediction = pd.DataFrame(arima_model.predict(n_periods=len(test)), index=test.index)
#         prediction.columns = ['predicted_sales']
#         prediction.reset_index(inplace=True)

#         # Future ARIMA Forecast
#         index_future_dates = pd.date_range(start='2021-10-01', end='2023-01-01', freq='M')
#         prediction_arima = pd.DataFrame(arima_model.predict(n_periods=len(index_future_dates)), index=index_future_dates)
#         prediction_arima.columns = ['predicted_sales']

#         ################################################################################
#         # LSTM Forecasting
#         rnn_train = sales[:93]
#         rnn_test = sales[93:]

#         scaler = MinMaxScaler()
#         scaler.fit(rnn_train)

#         scaled_train = scaler.transform(rnn_train)
#         scaled_test = scaler.transform(rnn_test)

#         # Define LSTM model
#         n_input = 12
#         n_features = 1
#         generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

#         model = Sequential([
#             LSTM(100, activation='relu', input_shape=(n_input, n_features)),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(generator, epochs=50)

#         # Make LSTM predictions
#         test_predictions = []
#         current_batch = scaled_train[-n_input:].reshape((1, n_input, n_features))

#         for _ in range(len(rnn_test)):
#             current_pred = model.predict(current_batch)[0]
#             test_predictions.append(current_pred)
#             current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

#         # Convert predictions back to original scale
#         true_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
#         rnn_test = rnn_test.copy()  # Avoid modifying a slice
#         rnn_test['Predictions'] = true_predictions

#         return Response({
#             "data1": prediction.to_json(),
#             "data2": rnn_test.to_json(),
#             "data3": prediction_arima.to_json()
#         })



# class FileUploadView(APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request, format=None):
#         if 'file' not in request.data:
#             return Response({"error": "No file provided"}, status=400)

#         file_obj = request.data['file']
#         sales = pd.read_csv(file_obj)

#         sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
#         sales.set_index('Date', inplace=True)

#         if 'Revenue' not in sales.columns:
#             return Response({"error": "CSV file must contain 'Revenue' column"}, status=400)

#         revenue = sales['Revenue']

#         # ✅ Use full data
#         train = revenue  

#         # ✅ Use manual ARIMA order
#         model = ARIMA(train, order=(1,1,1))
#         model_fit = model.fit()

#         # ✅ Print summary to debug
#         print(model_fit.summary())

#         # ✅ Forecast only 3 months (not 12)
#         future_forecast = model_fit.forecast(steps=3)
#         print(future_forecast)  # Debugging

#         # ✅ Generate future dates correctly
#         index_future_dates = pd.date_range(start=sales.index[-1] + pd.DateOffset(years=1), periods=3, freq='M')

#         future_forecast_df = pd.DataFrame(future_forecast.values, index=index_future_dates, columns=["Forecasted Revenue"])

#         return Response({
#               "past_data": sales[['Revenue']].to_json(date_format="iso"),  # Include past revenue data

#             "future_forecast": future_forecast_df.to_json(date_format="iso")
#         })



# third



class FileUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        if 'file' not in request.data:
            return Response({"error": "No file provided"}, status=400)

        file_obj = request.data['file']
        sales = pd.read_csv(file_obj)

        sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
        sales.set_index('Date', inplace=True)

        if 'Revenue' not in sales.columns or 'Expenses' not in sales.columns:
            return Response({"error": "CSV file must contain 'Revenue' and 'Expenses' columns"}, status=400)

        revenue = sales['Revenue']
        expenses = sales['Expenses']

        # ✅ Train ARIMA models separately for Revenue and Expenses
        model_revenue = ARIMA(revenue, order=(1,1,1)).fit()
        model_expenses = ARIMA(expenses, order=(1,1,1)).fit()

        # ✅ Forecast for the next 3 months
        revenue_forecast = model_revenue.forecast(steps=3)
        expenses_forecast = model_expenses.forecast(steps=3)

        # ✅ Generate future dates
        index_future_dates = pd.date_range(start=sales.index[-1] + pd.DateOffset(years=1), periods=3, freq='M')

        # ✅ Convert forecasts to DataFrames
        revenue_forecast_df = pd.DataFrame(revenue_forecast.values, index=index_future_dates, columns=["Forecasted Revenue"])
        expenses_forecast_df = pd.DataFrame(expenses_forecast.values, index=index_future_dates, columns=["Forecasted Expenses"])

        return Response({
            "past_data": sales[['Revenue', 'Expenses']].to_json(date_format="iso"),  # Past Data
            "revenue_forecast": revenue_forecast_df.to_json(date_format="iso"),  # Separate Revenue Forecast
            "expenses_forecast": expenses_forecast_df.to_json(date_format="iso")  # Separate Expenses Forecast
        })

# import pandas as pd
# from pmdarima import auto_arima  # ✅ Automatically selects best (p,d,q)
# from rest_framework.parsers import FormParser, MultiPartParser
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # class FileUploadView(APIView):
# #     parser_classes = [MultiPartParser, FormParser]

# #     def post(self, request, format=None):
# #         if 'file' not in request.data:
# #             return Response({"error": "No file provided"}, status=400)

# #         file_obj = request.data['file']
# #         sales = pd.read_csv(file_obj)

# #         sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
# #         sales.set_index('Date', inplace=True)

# #         if 'Revenue' not in sales.columns or 'Expenses' not in sales.columns:
# #             return Response({"error": "CSV file must contain 'Revenue' and 'Expenses' columns"}, status=400)

# #         revenue = sales['Revenue']
# #         expenses = sales['Expenses']

# #         # ✅ Automatically determine the best ARIMA order
# #         best_order_revenue = auto_arima(revenue, seasonal=True, stepwise=True).order
# #         best_order_expenses = auto_arima(expenses, seasonal=True, stepwise=True).order

# #         # ✅ Train ARIMA models with the best order
# #         model_revenue = ARIMA(revenue, order=best_order_revenue).fit()
# #         model_expenses = ARIMA(expenses, order=best_order_expenses).fit()

# #         # ✅ Forecast for the next 3 months
# #         revenue_forecast = model_revenue.forecast(steps=3)
# #         expenses_forecast = model_expenses.forecast(steps=3)

# #         # ✅ Generate future dates
# #         index_future_dates = pd.date_range(start=sales.index[-1] + pd.DateOffset(years=1), periods=3, freq='M')

# #         # ✅ Convert forecasts to DataFrames
# #         revenue_forecast_df = pd.DataFrame(revenue_forecast.values, index=index_future_dates, columns=["Forecasted Revenue"])
# #         expenses_forecast_df = pd.DataFrame(expenses_forecast.values, index=index_future_dates, columns=["Forecasted Expenses"])

# #         return Response({
# #             "past_data": sales[['Revenue', 'Expenses']].to_json(date_format="iso"),  # Past Data
# #             "revenue_forecast": revenue_forecast_df.to_json(date_format="iso"),  # Dynamic Revenue Forecast
# #             "expenses_forecast": expenses_forecast_df.to_json(date_format="iso"),  # Dynamic Expenses Forecast
# #             "arima_orders": {
# #                 "revenue_order": best_order_revenue,
# #                 "expenses_order": best_order_expenses
# #             }
# #         })


# class FileUploadView(APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request, format=None):
#         if 'file' not in request.data:
#             return Response({"error": "No file provided"}, status=400)

#         file_obj = request.data['file']
#         sales = pd.read_csv(file_obj)

#         sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
#         sales.set_index('Date', inplace=True)

#         if 'Revenue' not in sales.columns or 'Expenses' not in sales.columns:
#             return Response({"error": "CSV file must contain 'Revenue' and 'Expenses' columns"}, status=400)

#         revenue = sales['Revenue']
#         expenses = sales['Expenses']

#         # Handle missing data
#         sales = sales.dropna(subset=['Revenue', 'Expenses'])

#         # Automatically determine the best ARIMA order (including seasonal parameters)
#         best_order_revenue = auto_arima(revenue, seasonal=False, stepwise=True).order
#         best_order_expenses = auto_arima(expenses, seasonal=False, stepwise=True).order

#         # Debug: print the best orders
#         print(f"Best order for revenue: {best_order_revenue}")
#         print(f"Best order for expenses: {best_order_expenses}")

#         # Check if the orders have enough parameters
#         if len(best_order_revenue) < 3:
#             return Response({"error": "ARIMA model failed to find suitable parameters for Revenue"}, status=400)
#         if len(best_order_expenses) < 3:
#             return Response({"error": "ARIMA model failed to find suitable parameters for Expenses"}, status=400)

#         try:
#             # Try to fit ARIMA model
#             model_revenue = SARIMAX(revenue, order=best_order_revenue[:3]).fit(disp=False)
#             model_expenses = SARIMAX(expenses, order=best_order_expenses[:3]).fit(disp=False)

#             print("model_revenue",model_revenue)
#             print("model_expenses",model_expenses)


#             # Forecast for the next 3 months
#             revenue_forecast = model_revenue.forecast(steps=3)
#             expenses_forecast = model_expenses.forecast(steps=3)

#             # Generate future dates
#             index_future_dates = pd.date_range(start=sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='M')

#             # Convert forecasts to DataFrames
#             revenue_forecast_df = pd.DataFrame(revenue_forecast, index=index_future_dates, columns=["Forecasted Revenue"])
#             expenses_forecast_df = pd.DataFrame(expenses_forecast, index=index_future_dates, columns=["Forecasted Expenses"])

#             return Response({
#                 "past_data": sales[['Revenue', 'Expenses']].to_json(date_format="iso"),
#                 "revenue_forecast": revenue_forecast_df.to_json(date_format="iso"),
#                 "expenses_forecast": expenses_forecast_df.to_json(date_format="iso"),
#                 "sarima_orders": {
#                     "revenue_order": best_order_revenue,
#                     "expenses_order": best_order_expenses
#                 }
#             })
#         except Exception as e:
#             return Response({"error": str(e)}, status=400)
