import pandas as pd

def load_and_prepare_data():

    data1 = pd.read_csv("CrimesOnWomenData.csv")

    data1 = data1.rename(columns={"Unnamed: 0": "NO"})

    # Streamlit-friendly names
    data1.columns = [
        "NO",
        "State",
        "Year",
        "Rape",
        "Kidnap",
        "Dowry",
        "Assault",
        "Modesty",
        "DomesticViolence",
        "Trafficking"
    ]



    data1["State"] = (
        data1["State"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    crime_columns = list(data1.columns[3:])

    data1["Total Crimes"] = data1[crime_columns].sum(axis=1)


    long_df = data1.melt(
        id_vars=["NO", "State", "Year"],
        value_vars=crime_columns,
        var_name="Type of Crime",
        value_name="Crime Count"
    )


    return data1, long_df, crime_columns