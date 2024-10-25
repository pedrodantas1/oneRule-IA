import pandas as pd

df = pd.read_csv("creditability.csv")
df_novo = df.drop(
    [
        "Duration_of_Credit_monthly",
        "Instalment_per_cent",
        "Duration_in_Current_address",
        "Most_valuable_available_asset",
        "Age_years",
        "Type_of_apartment",
        "Occupation",
        "No_of_dependents",
        "Telephone",
        "Foreign_Worker",
        "Guarantors",
        "Sex_Marital_Status",
    ],
    axis=1,
)

df_novo.to_csv("dataset.csv", index=False)
