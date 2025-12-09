from gridstatusio import GridStatusClient
import pandas as pd
# Recommended: set GRIDSTATUS_API_KEY as an
# environment variable instead of hardcoding
# client = GridStatusClient("9b4452518ea34945bc6d06a7da29b20e")
# # Fetch data as pandas DataFrame
# df = client.get_dataset(
#   dataset="pjm_outages_daily",
#   start="2009-11-28",
#   end="2025-12-01",
#   publish_time="latest",
#   timezone="market",
# )



# print (df.to_csv("Outage_Dataset" , index = False))

df = pd.read_csv("./Outage_Dataset.csv")

print(df.info())