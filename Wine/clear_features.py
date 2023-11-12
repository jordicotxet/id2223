import hopsworks



project = hopsworks.login()
fs = project.get_feature_store()

try:
    wine_fg = fs.get_feature_group(name="wine", version=1)
    query = wine_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="wine",
                                    version=1,
                                    description="Read from wine dataset",
                                    labels=["quality"],
                                    query=query)

    feature_view.delete()
except:
    print("Featureview deletion unsuccessful")

wine_fg = fs.get_feature_group(
    name="wine",
    version=1)
wine_fg.delete()

for key in [0, 1, 2]:
    fg = fs.get_feature_group(
    name="wine_subset_" + str(int(key)),
    version=1)
    fg.delete()





