from utils.load import load
from utils.data_transformation import get_correct_countries



def run():
    metrics = ['Cos', 'Euc', 'Man', 'Het']
    categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']
    continents = ["africa","americas","asia","europe","oceania"]
    for met in metrics:
        for cat in categories:
            for continent in continents:
                path = f"data/fb_data/FB{met}Dist_{cat}.csv"
                df = load(path,",")
                iso_path = f"data/iso_codes/{continent}.txt"
                df = get_correct_countries(iso_path,df)
                df.to_csv(f"data/fb_data_continents_new/FB{met}Dist_{cat}_{continent}.csv",index = False)