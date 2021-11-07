import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine

# pg_cred = "POSTGRES_URI"
engine = create_engine(pg_cred)

def query_sql(query):
    with engine.connect() as conn:
        return pd.read_sql(query, con=conn)


def insert(df, table):
    with engine.connect() as conn:
        df.to_sql(table, con=conn, if_exists="append", index=False, method='multi')


def insert_batched(df, table, batch_size=5000):
    with engine.connect() as conn:
        i = 0
        total = len(df)
        pbar = tqdm(total=len(df))

        while i <= total:
            df[i:i+batch_size].to_sql(table, con=conn,
                                    if_exists="append", index=False, method='multi')
            i += batch_size
            pbar.update(batch_size)

        pbar.close()
