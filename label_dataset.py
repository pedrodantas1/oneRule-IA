import pandas as pd

df = pd.read_csv("dataset.csv")

# Creditability
df["Creditability"] = df["Creditability"].replace({0: "Bad", 1: "Good"})
# print(df["Creditability"])

# Account_Balance
df["Account_Balance"] = df["Account_Balance"].replace(
    {1: "No account", 2: "No balance", 3: "Some Balance", 4: "Some Balance"}
)
# print(df["Account_Balance"])

# Payment_Status_of_Previous_Credit
df["Payment_Status_of_Previous_Credit"] = df[
    "Payment_Status_of_Previous_Credit"
].replace(
    {
        0: "Some Problems",
        1: "Some Problems",
        2: "Paid Up",
        3: "No Problems",
        4: "No Problems",
    }
)
# print(df["Payment_Status_of_Previous_Credit"])

# Credit amount
bins = [0, 2500, 5000, 10000, float("inf")]
labels = ["Low", "Medium", "High", "Very High"]
df["Credit_Amount"] = pd.cut(df["Credit_Amount"], bins=bins, labels=labels)
# print(df["Credit_Amount"])

# Value_Savings_Stocks
df["Value_Savings_Stocks"] = df["Value_Savings_Stocks"].replace(
    {
        1: "None",
        2: "Below 50 EUR",
        3: "50 to 500 EUR",
        4: "500 to 2000 EUR",
        5: "500 to 2000 EUR",
    }
)
# print(df["Value_Savings_Stocks"])

# Length_of_current_employment
df["Length_of_current_employment"] = df["Length_of_current_employment"].replace(
    {
        1: "Below 1 year",
        2: "1 to 4 years",
        3: "4 to 7 years",
        4: "Above 7 years",
        5: "Above 7 years",
    }
)
# print(df["Length_of_current_employment"])

# Guarantors
df["Guarantors"] = df["Guarantors"].replace({1: "None", 2: "Yes", 3: "Yes"})
# print(df["Guarantors"])

# Concurrent_Credits
df["Concurrent_Credits"] = df["Concurrent_Credits"].replace(
    {1: "None", 2: "Other Banks or Dept Stores", 3: "Other Banks or Dept Stores"}
)
# print(df["Concurrent_Credits"])

# No_of_Credits_at_this_Bank
df["No_of_Credits_at_this_Bank"] = df["No_of_Credits_at_this_Bank"].replace(
    {1: "One", 2: "More than one", 3: "None", 4: "None"}
)
# print(df["No_of_Credits_at_this_Bank"])

df.to_csv("dataset_labeled.csv", index=False)
