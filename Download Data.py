import pandas as pd
from ucimlrepo import fetch_ucirepo

data_set = fetch_ucirepo(id=17)

pd.DataFrame(data_set.data.features).to_csv("features.csv", index=False)
pd.DataFrame(data_set.data.targets).to_csv("targets.csv", index=False)

# downloaded to speed up running