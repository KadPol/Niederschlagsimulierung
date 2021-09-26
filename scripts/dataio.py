import pandas as pd


def load_forcings(path: str) -> pd.DataFrame:
    """
    Loads CAMELS forcing data from raw text files

    Parameters
    ----------
    path: str
        Path to the raw text file containing forcing data for a certain basin

    Returns
    -------
    pd.DataFrame
        DataFrame containing DateTime indexed forcing data for a basin

    """
    colnames = pd.read_csv(path, sep=' ', skiprows=3, nrows=1, header=None)
    df = pd.read_csv(path, sep='\t', skiprows=4, header=None, decimal='.',
                     names=colnames.iloc[0, 3:])
    dates = df.iloc[:, 0]
    df = df.drop(columns=df.columns[0])
    df["date"] = pd.to_datetime(dates.str.split(expand=True)
                                .drop([3], axis=1)
                                .rename(columns={0: "year", 1: "month", 2: "day"}))
    df = df.set_index("date")
    return df


def load_streamflow(path: str) -> pd.DataFrame:
    """
    Loads CAMELS streamflow data from raw text files

    Parameters
    ----------
    path: str
        Path to the raw text file containing streamflow data for a certain basin

    Returns
    -------
    pd.DataFrame
        DataFrame containing DateTime indexed streamflow data for a basin

    """
    df = pd.read_csv(path, delim_whitespace=True, header=None, decimal='.', na_values=["-999.00"],
                     names=["gauge_id", "year", "month", "day", "streamflow", "qc_flag"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(columns=["year", "month", "day"]).set_index("date")
    return df
