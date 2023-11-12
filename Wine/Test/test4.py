import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("jim-hopsworks-ai"))
   def f():
       g()


def generate_wine(stats):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df_dict = {}

    for row in stats:
        df_dict[row['column']] = [random.gauss(row['mean'], row['stdDev'])]

    df = pd.DataFrame(df_dict)
    return df



def g():
    import hopsworks
    import pandas as pd
    import pandas as pd
    import random

    project = hopsworks.login()
    fs = project.get_feature_store()

    quality = 0
    print("Adding wine of quality", quality)
    name = "wine_subset_" + str(quality)
    stat_fg = fs.get_feature_group(name=name, version=1)
    stats = stat_fg.get_statistics().content['columns']

    wine = generate_wine(stats)

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine)
    

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
